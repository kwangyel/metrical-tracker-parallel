from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parallel_batching import (  # noqa: E402
    ChunkSpec,
    format_frame_id,
    owned_frame_ids,
    plan_chunks,
    resolve_merge_owner,
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


ARTIFACT_EXTENSIONS = {
    "checkpoint": ".frame",
    "mesh": ".ply",
    "depth": ".png",
    "video": ".jpg",
    "input": ".png",
}


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
    parser.add_argument("--render_video", action="store_true", default=RENDER_VIDEO)
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
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_tracker_chunk, cfg_file, str(save_folder), asdict(chunk), device)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[chunk {result['chunk_id']}] finished")

    return sorted(results, key=lambda item: item["chunk_id"])


def copy_frame_artifact(src_actor_dir: Path, dst_actor_dir: Path, folder_name: str, frame_id: int) -> None:
    suffix = ARTIFACT_EXTENSIONS[folder_name]
    frame_name = format_frame_id(frame_id) + suffix
    src = src_actor_dir / folder_name / frame_name
    dst = dst_actor_dir / folder_name / frame_name

    if not src.exists():
        raise FileNotFoundError(f"Missing expected artifact: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_chunk_outputs(base_save_folder: Path, actor_name: str, chunks: List[ChunkSpec], total_frames: int) -> Path:
    owners = resolve_merge_owner(chunks)
    expected_frames = set(range(total_frames))
    if set(owners.keys()) != expected_frames:
        missing = sorted(expected_frames - set(owners.keys()))
        extra = sorted(set(owners.keys()) - expected_frames)
        raise ValueError(f"Merge ownership does not match frame range. missing={missing[:10]} extra={extra[:10]}")

    merged_actor_dir = base_save_folder / "merged" / actor_name
    if merged_actor_dir.exists():
        shutil.rmtree(merged_actor_dir)
    merged_actor_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        src_actor_dir = chunk_save_folder(base_save_folder, chunk) / actor_name
        for frame_id in owned_frame_ids(chunk):
            for folder_name in ARTIFACT_EXTENSIONS:
                copy_frame_artifact(src_actor_dir, merged_actor_dir, folder_name, frame_id)

        canonical = src_actor_dir / "canonical.obj"
        if chunk.chunk_id == 0 and canonical.exists():
            shutil.copy2(canonical, merged_actor_dir / "canonical.obj")

    checkpoint_count = len(list((merged_actor_dir / "checkpoint").glob("*.frame")))
    if checkpoint_count != total_frames:
        raise ValueError(f"Merged checkpoint count {checkpoint_count} != total frame count {total_frames}")

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

    chunks = plan_chunks(total_frames, args.batch_frames, args.overlap_frames)
    print(f"Actor: {actor_name}")
    print(f"Total frames: {total_frames}")
    print(f"Batch frames: {args.batch_frames}")
    print(f"Overlap frames: {args.overlap_frames}")
    print(f"Workers: {args.num_workers}")
    print_chunk_table(chunks, args.num_workers)

    results = run_chunks_parallel(args.cfg, save_folder, chunks, args.num_workers, args.device)
    print(f"\nFinished {len(results)} chunks. Merging owned frame artifacts...")
    merged_actor_dir = merge_chunk_outputs(save_folder, actor_name, chunks, total_frames)

    if args.render_video:
        render_merged_video(merged_actor_dir, fps)

    print(f"\nPASS: merged {total_frames} frames into {merged_actor_dir}")


if __name__ == "__main__":
    main()
