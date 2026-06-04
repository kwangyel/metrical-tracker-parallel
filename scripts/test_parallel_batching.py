from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parallel_batching import (  # noqa: E402
    expected_owned_frames,
    plan_chunks,
    plan_chunks_for_range,
    resolve_merge_owner,
    schedule_chunks,
    validate_batching_params,
)


BATCH_FRAMES = 150
OVERLAP_FRAMES = 20
NUM_WORKERS = 2
TOTAL_FRAMES = 500


def assert_chunk(chunk, chunk_id, start_frame, end_frame, owned_start, owned_end):
    assert chunk.chunk_id == chunk_id
    assert chunk.start_frame == start_frame
    assert chunk.end_frame == end_frame
    assert chunk.owned_start == owned_start
    assert chunk.owned_end == owned_end


def test_single_chunk():
    chunks = plan_chunks(total_frames=100, batch_frames=150, overlap_frames=20)
    assert len(chunks) == 1
    assert_chunk(chunks[0], 0, 0, 99, 0, 99)


def test_two_chunks_with_overlap():
    chunks = plan_chunks(total_frames=200, batch_frames=150, overlap_frames=20)
    assert len(chunks) == 2
    assert_chunk(chunks[0], 0, 0, 149, 0, 149)
    assert_chunk(chunks[1], 1, 130, 199, 150, 199)


def test_user_overlap_example():
    chunks = plan_chunks(total_frames=300, batch_frames=150, overlap_frames=20)
    owners = resolve_merge_owner(chunks)

    assert_chunk(chunks[0], 0, 0, 149, 0, 149)
    assert_chunk(chunks[1], 1, 130, 279, 150, 279)
    assert_chunk(chunks[2], 2, 260, 299, 280, 299)

    for frame_id in range(130, 150):
        assert owners[frame_id] == 0
    for frame_id in range(150, 280):
        assert owners[frame_id] == 1


def test_partial_tail():
    chunks = plan_chunks(total_frames=281, batch_frames=150, overlap_frames=20)
    assert len(chunks) == 3
    assert_chunk(chunks[-1], 2, 260, 280, 280, 280)


def test_parameter_validation():
    invalid_cases = [
        {"total_frames": -1, "batch_frames": 150, "overlap_frames": 20, "num_workers": 1},
        {"total_frames": 10, "batch_frames": 0, "overlap_frames": 0, "num_workers": 1},
        {"total_frames": 10, "batch_frames": 150, "overlap_frames": -1, "num_workers": 1},
        {"total_frames": 10, "batch_frames": 150, "overlap_frames": 150, "num_workers": 1},
        {"total_frames": 10, "batch_frames": 150, "overlap_frames": 20, "num_workers": 0},
    ]

    for kwargs in invalid_cases:
        try:
            validate_batching_params(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {kwargs}")


def test_merge_ownership_has_full_coverage():
    total_frames = 500
    chunks = plan_chunks(total_frames=total_frames, batch_frames=150, overlap_frames=20)
    owners = resolve_merge_owner(chunks)

    assert set(owners.keys()) == set(range(total_frames))
    assert min(owners) == 0
    assert max(owners) == total_frames - 1


def test_queue_scheduling():
    chunks = plan_chunks(total_frames=600, batch_frames=150, overlap_frames=20)
    waves = schedule_chunks(chunks, num_workers=2)

    assert len(chunks) == 5
    assert all(len(wave) <= 2 for wave in waves)
    assert [[chunk.chunk_id for chunk in wave] for wave in waves] == [[0, 1], [2, 3], [4]]


def test_plan_chunks_from_start_frame():
    chunks = plan_chunks_for_range(500, 150, 20, start_frame=200)
    assert len(chunks) == 3
    assert_chunk(chunks[0], 0, 200, 349, 200, 349)
    assert_chunk(chunks[1], 1, 330, 479, 350, 479)
    assert_chunk(chunks[2], 2, 460, 499, 480, 499)


def test_plan_chunks_with_end_frame():
    chunks = plan_chunks_for_range(500, 150, 20, start_frame=200, end_frame=349)
    assert len(chunks) == 1
    assert_chunk(chunks[0], 0, 200, 349, 200, 349)


def test_expected_owned_frames_subset():
    start_frame = 200
    end_frame = 499
    chunks = plan_chunks_for_range(500, 150, 20, start_frame=start_frame, end_frame=end_frame)
    owned = expected_owned_frames(chunks)

    assert owned == set(range(start_frame, end_frame + 1))
    assert owned.isdisjoint(set(range(start_frame)))


def test_start_frame_validation():
    try:
        plan_chunks_for_range(500, 150, 20, start_frame=500)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected validation failure for start_frame >= total_frames")

    try:
        plan_chunks_for_range(500, 150, 20, start_frame=400, end_frame=200)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected validation failure for start_frame > end_frame")


def test_custom_params(total_frames, batch_frames, overlap_frames, num_workers):
    validate_batching_params(total_frames, batch_frames, overlap_frames, num_workers)
    chunks = plan_chunks(total_frames, batch_frames, overlap_frames)
    waves = schedule_chunks(chunks, num_workers)
    owners = resolve_merge_owner(chunks)

    if total_frames == 0:
        assert chunks == []
        assert waves == []
        assert owners == {}
        return

    assert set(owners.keys()) == set(range(total_frames))
    assert all(len(wave) <= num_workers for wave in waves)


def print_chunk_table(total_frames, batch_frames, overlap_frames, num_workers):
    chunks = plan_chunks(total_frames, batch_frames, overlap_frames)
    waves = schedule_chunks(chunks, num_workers)

    print("\nChunk plan")
    print("chunk_id | process_start | process_end | owned_start | owned_end")
    for chunk in chunks:
        print(
            f"{chunk.chunk_id:8d} | {chunk.start_frame:13d} | {chunk.end_frame:11d} | "
            f"{chunk.owned_start:11d} | {chunk.owned_end:9d}"
        )

    print("\nQueue waves")
    for wave_id, wave in enumerate(waves):
        ids = ", ".join(str(chunk.chunk_id) for chunk in wave)
        print(f"wave {wave_id}: chunks {ids}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test parallel tracker batching and queue planning.")
    parser.add_argument("--batch_frames", type=int, default=BATCH_FRAMES)
    parser.add_argument("--overlap_frames", type=int, default=OVERLAP_FRAMES)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--total_frames", type=int, default=TOTAL_FRAMES)
    return parser.parse_args()


def main():
    args = parse_args()

    test_single_chunk()
    test_two_chunks_with_overlap()
    test_user_overlap_example()
    test_partial_tail()
    test_parameter_validation()
    test_merge_ownership_has_full_coverage()
    test_queue_scheduling()
    test_plan_chunks_from_start_frame()
    test_plan_chunks_with_end_frame()
    test_expected_owned_frames_subset()
    test_start_frame_validation()
    test_custom_params(args.total_frames, args.batch_frames, args.overlap_frames, args.num_workers)

    print_chunk_table(args.total_frames, args.batch_frames, args.overlap_frames, args.num_workers)
    print("\nPASS: batching, overlap ownership, and queue scheduling tests passed.")


if __name__ == "__main__":
    main()
