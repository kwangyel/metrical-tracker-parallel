from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parallel_batching import (  # noqa: E402
    ChunkSpec,
    expected_owned_frames,
    format_frame_id,
    owned_frame_ids,
    plan_chunks_for_range,
    resolve_effective_end_frame,
    schedule_chunks,
    validate_batching_params,
)


# Edit these defaults directly in a Colab cell, or override them with CLI flags.
CFG_FILE = "./configs/actors/duda.yml"
SAVE_FOLDER = "./output_parallel/"
BATCH_FRAMES = 150
OVERLAP_FRAMES = 20
NUM_WORKERS = 2
DEVICE = "cuda:0"
RUN_PREPROCESS = True
RENDER_VIDEO = False
START_FRAME = 0
END_FRAME = -1

CHECKPOINT_SUFFIX = ".frame"


def ensure_repo_root() -> None:
    import os

    os.chdir(REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Metrical tracker in overlapping Colab chunks.")
    parser.add_argument("--cfg", default=CFG_FILE, help="Path to tracker actor config YAML.")
    parser.add_argument("--save_folder", default=SAVE_FOLDER, help="Base output folder for chunk and merged outputs.")
    parser.add_argument("--batch_frames", type=int, default=BATCH_FRAMES)
    parser.add_argument("--overlap_frames", type=int, default=OVERLAP_FRAMES)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--skip_preprocess", action="store_true", help="Use existing actor/images, kpt, and kpt_dense files.")
    parser.add_argument("--render_video", action="store_true", default=RENDER_VIDEO,
                        help="Render video from merged/<actor>/video/ (requires video frames; not created by checkpoint merge).")
    parser.add_argument("--start_frame", type=int, default=START_FRAME,
                        help="0-based first global frame index to process (default: 0).")
    parser.add_argument("--end_frame", type=int, default=END_FRAME,
                        help="0-based inclusive last frame to process; -1 means last frame in dataset.")
    return parser.parse_args()


def save_folder_with_slash(path: Path) -> str:
    return str(path) + "/"


def chunk_save_folder(base_save_folder: Path, chunk: ChunkSpec) -> Path:
    return base_save_folder / f"chunk_{chunk.chunk_id:03d}"


def print_chunk_table(chunks: Iterable[ChunkSpec], num_workers: int) -> None:
    chunks = list(chunks)
    waves = schedule_chunks(chunks, num_workers)

    print("\nChunk plan")
    print("chunk_id | process_start | process_end | owned_start | owned_end | frames")
    for chunk in chunks:
        print(
            f"{chunk.chunk_id:8d} | {chunk.start_frame:13d} | {chunk.end_frame:11d} | "
            f"{chunk.owned_start:11d} | {chunk.owned_end:9d} | {chunk.frame_count:6d}"
        )

    print("\nQueue waves")
    for wave_id, wave in enumerate(waves):
        ids = ", ".join(str(chunk.chunk_id) for chunk in wave)
        print(f"wave {wave_id}: chunks {ids}")


def load_config(cfg_file: str, save_folder: Path | None = None, start_frame: int | None = None, end_frame: int | None = None):
    from configs.config import parse_cfg

    cfg = parse_cfg(cfg_file)
    if save_folder is not None:
        cfg.save_folder = save_folder_with_slash(save_folder)
    if start_frame is not None:
        cfg.start_frame = int(start_frame)
    if end_frame is not None:
        cfg.end_frame = int(end_frame)
    return cfg


def preprocess_and_count_frames(cfg_file: str, save_folder: Path, run_preprocess: bool) -> tuple[str, int, int]:
    ensure_repo_root()

    cfg = load_config(cfg_file, save_folder=save_folder)
    actor_dir = Path(cfg.actor)
    identity_path = actor_dir / "identity.npy"
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing required identity file: {identity_path}")

    if run_preprocess:
        from datasets.generate_dataset import GeneratorDataset

        GeneratorDataset(cfg.actor, cfg).run()

    from datasets.image_dataset import ImagesDataset

    dataset = ImagesDataset(cfg)
    total_frames = len(dataset)
    if total_frames == 0:
        raise ValueError(f"No frames found in {actor_dir / 'images'}")

    return cfg.config_name, total_frames, int(cfg.fps)


def run_tracker_chunk(cfg_file: str, base_save_folder: str, chunk_payload: Dict, device: str) -> Dict:
    ensure_repo_root()

    chunk = ChunkSpec(**chunk_payload)
    save_folder = chunk_save_folder(Path(base_save_folder), chunk)
    cfg = load_config(
        cfg_file,
        save_folder=save_folder,
        start_frame=chunk.start_frame,
        end_frame=chunk.end_frame,
    )

    from tracker import Tracker

    print(
        f"[chunk {chunk.chunk_id}] processing frames {chunk.start_frame}-{chunk.end_frame}; "
        f"owning {chunk.owned_start}-{chunk.owned_end}; output={save_folder}"
    )
    Tracker(cfg, device=device).run()

    return {
        "chunk_id": chunk.chunk_id,
        "start_frame": chunk.start_frame,
        "end_frame": chunk.end_frame,
        "owned_start": chunk.owned_start,
        "owned_end": chunk.owned_end,
        "save_folder": str(save_folder),
    }


def run_chunks_parallel(cfg_file: str, save_folder: Path, chunks: List[ChunkSpec], num_workers: int, device: str) -> List[Dict]:
    results: List[Dict] = []
    # CUDA is incompatible with forked subprocesses; enforce spawn for Colab/Linux workers.
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        futures = [
            executor.submit(run_tracker_chunk, cfg_file, str(save_folder), asdict(chunk), device)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[chunk {result['chunk_id']}] finished")

    return sorted(results, key=lambda item: item["chunk_id"])


def copy_checkpoint(src_actor_dir: Path, dst_checkpoint_dir: Path, frame_id: int) -> None:
    frame_name = format_frame_id(frame_id) + CHECKPOINT_SUFFIX
    src = src_actor_dir / "checkpoint" / frame_name
    dst = dst_checkpoint_dir / frame_name

    if not src.exists():
        raise FileNotFoundError(f"Missing expected checkpoint: {src}")

    dst_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_chunk_checkpoints(base_save_folder: Path, actor_name: str, chunks: List[ChunkSpec]) -> Path:
    expected = expected_owned_frames(chunks)
    merged_actor_dir = base_save_folder / "merged" / actor_name
    merged_checkpoint_dir = merged_actor_dir / "checkpoint"
    merged_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        src_actor_dir = chunk_save_folder(base_save_folder, chunk) / actor_name
        for frame_id in owned_frame_ids(chunk):
            copy_checkpoint(src_actor_dir, merged_checkpoint_dir, frame_id)

    missing = sorted(
        frame_id for frame_id in expected
        if not (merged_checkpoint_dir / (format_frame_id(frame_id) + CHECKPOINT_SUFFIX)).exists()
    )
    if missing:
        raise ValueError(
            f"Merged checkpoint missing {len(missing)} frame(s); first missing index: {missing[0]}"
        )

    return merged_actor_dir


def render_merged_video(merged_actor_dir: Path, fps: int) -> None:
    import util

    util.images_to_video(str(merged_actor_dir), fps)


def main():
    args = parse_args()
    save_folder = Path(args.save_folder)
    run_preprocess = RUN_PREPROCESS and not args.skip_preprocess

    validate_batching_params(
        total_frames=0,
        batch_frames=args.batch_frames,
        overlap_frames=args.overlap_frames,
        num_workers=args.num_workers,
    )

    actor_name, total_frames, fps = preprocess_and_count_frames(args.cfg, save_folder, run_preprocess)
    validate_batching_params(total_frames, args.batch_frames, args.overlap_frames, args.num_workers)

    start_frame = int(args.start_frame)
    effective_end = resolve_effective_end_frame(total_frames, args.end_frame)
    if start_frame > effective_end:
        raise ValueError(
            f"start_frame ({start_frame}) must be <= end_frame ({effective_end})"
        )

    chunks = plan_chunks_for_range(
        total_frames,
        args.batch_frames,
        args.overlap_frames,
        start_frame=start_frame,
        end_frame=effective_end,
    )
    expected = expected_owned_frames(chunks)

    print(f"Actor: {actor_name}")
    print(f"Total frames in dataset: {total_frames}")
    print(f"Processing range: {start_frame}..{effective_end} (inclusive, {len(expected)} owned frames)")
    print(f"Batch frames: {args.batch_frames}")
    print(f"Overlap frames: {args.overlap_frames}")
    print(f"Workers: {args.num_workers}")
    print_chunk_table(chunks, args.num_workers)

    results = run_chunks_parallel(args.cfg, save_folder, chunks, args.num_workers, args.device)
    print(f"\nFinished {len(results)} chunks. Merging owned checkpoints...")
    merged_actor_dir = merge_chunk_checkpoints(save_folder, actor_name, chunks)

    if args.render_video:
        video_dir = merged_actor_dir / "video"
        if not video_dir.exists() or not any(video_dir.glob("*.jpg")):
            print(
                "Warning: --render_video skipped; merged output has checkpoints only "
                "(no video/ frames under merged actor dir)."
            )
        else:
            render_merged_video(merged_actor_dir, fps)

    print(
        f"\nPASS: merged {len(expected)} checkpoints into "
        f"{merged_actor_dir / 'checkpoint'}"
    )


if __name__ == "__main__":
    main()
