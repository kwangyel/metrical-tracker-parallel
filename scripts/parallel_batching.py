from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    start_frame: int
    end_frame: int
    overlap_start: Optional[int]
    owned_start: int
    owned_end: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def owned_count(self) -> int:
        if self.owned_end < self.owned_start:
            return 0
        return self.owned_end - self.owned_start + 1


def validate_batching_params(total_frames: int, batch_frames: int, overlap_frames: int, num_workers: int = 1) -> None:
    if total_frames < 0:
        raise ValueError("total_frames must be >= 0")
    if batch_frames < 1:
        raise ValueError("batch_frames must be >= 1")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")
    if overlap_frames >= batch_frames:
        raise ValueError("overlap_frames must be smaller than batch_frames")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")


def plan_chunks(total_frames: int, batch_frames: int, overlap_frames: int) -> List[ChunkSpec]:
    validate_batching_params(total_frames, batch_frames, overlap_frames)
    if total_frames == 0:
        return []

    chunks: List[ChunkSpec] = []
    stride = batch_frames - overlap_frames
    last_frame = total_frames - 1
    start_frame = 0

    while start_frame <= last_frame:
        end_frame = min(start_frame + batch_frames - 1, last_frame)
        previous_end = chunks[-1].end_frame if chunks else None
        owned_start = 0 if previous_end is None else previous_end + 1
        chunk = ChunkSpec(
            chunk_id=len(chunks),
            start_frame=start_frame,
            end_frame=end_frame,
            overlap_start=None if previous_end is None else start_frame,
            owned_start=owned_start,
            owned_end=end_frame,
        )
        chunks.append(chunk)

        if end_frame == last_frame:
            break
        start_frame += stride

    return chunks


def schedule_chunks(chunks: Iterable[ChunkSpec], num_workers: int) -> List[List[ChunkSpec]]:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    chunk_list = list(chunks)
    return [chunk_list[i:i + num_workers] for i in range(0, len(chunk_list), num_workers)]


def resolve_merge_owner(chunks: Iterable[ChunkSpec]) -> Dict[int, int]:
    owners: Dict[int, int] = {}
    for chunk in chunks:
        for frame_id in range(chunk.start_frame, chunk.end_frame + 1):
            owners.setdefault(frame_id, chunk.chunk_id)
    return owners


def owned_frame_ids(chunk: ChunkSpec) -> range:
    if chunk.owned_count == 0:
        return range(0)
    return range(chunk.owned_start, chunk.owned_end + 1)


def format_frame_id(frame_id: int) -> str:
    return str(frame_id).zfill(5)
