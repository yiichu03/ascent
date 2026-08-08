"""Conservative global lifecycle for submap-local frontier candidates."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ascent.submaps.types import FrontierRecord, FrontierStatus


class FrontierRegistry:
    """Registry that never merges candidates using untrusted global XY alone."""

    def __init__(self) -> None:
        self._records: Dict[str, FrontierRecord] = {}
        self._next_index_by_submap: Dict[str, int] = {}

    @property
    def records(self) -> Dict[str, FrontierRecord]:
        return dict(self._records)

    def register_submap_frontiers(
        self,
        submap_id: str,
        frontiers_local: np.ndarray,
        creation_step: int,
        scores: Optional[Iterable[float]] = None,
    ) -> List[FrontierRecord]:
        frontiers = np.asarray(frontiers_local, dtype=np.float64)
        if frontiers.size == 0:
            return []
        if frontiers.ndim != 2 or frontiers.shape[1] != 2:
            raise ValueError(f"Invalid frontiers shape {frontiers.shape}")
        score_values = (
            [0.0] * len(frontiers)
            if scores is None
            else [float(score) for score in scores]
        )
        if len(score_values) != len(frontiers):
            raise ValueError("One score is required for each frontier")

        next_index = self._next_index_by_submap.get(submap_id, 0)
        created = []
        for point, score in zip(frontiers, score_values):
            frontier_id = f"{submap_id}:f{next_index:04d}"
            next_index += 1
            record = FrontierRecord(
                frontier_id=frontier_id,
                source_submap_id=submap_id,
                local_xy=point,
                creation_step=int(creation_step),
                score=score,
                last_update_step=int(creation_step),
            )
            self._records[frontier_id] = record
            created.append(record)
        self._next_index_by_submap[submap_id] = next_index
        return created

    def get(self, frontier_id: str) -> FrontierRecord:
        return self._records[frontier_id]

    def eligible(self, source_submap_id: Optional[str] = None) -> List[FrontierRecord]:
        records = [
            record
            for record in self._records.values()
            if record.eligible
            and (
                source_submap_id is None
                or record.source_submap_id == source_submap_id
            )
        ]
        return sorted(
            records,
            key=lambda record: (
                -record.score,
                record.creation_step,
                record.frontier_id,
            ),
        )

    def mark_selected(self, frontier_id: str, step: int) -> None:
        record = self.get(frontier_id)
        if not record.eligible:
            raise RuntimeError(f"Cannot select inactive frontier {frontier_id}")
        record.status = FrontierStatus.SELECTED
        record.selection_count += 1
        record.last_update_step = int(step)

    def release_selected(self, frontier_id: str, step: int) -> None:
        """Return an unattempted selected frontier to the eligible pool."""

        record = self.get(frontier_id)
        if record.status is not FrontierStatus.SELECTED:
            raise RuntimeError(
                f"Cannot release non-selected frontier {frontier_id}"
            )
        record.status = FrontierStatus.ACTIVE
        record.last_update_step = int(step)

    def mark_attempted(self, frontier_id: str, step: int) -> None:
        record = self.get(frontier_id)
        if not record.eligible:
            raise RuntimeError(f"Cannot attempt inactive frontier {frontier_id}")
        record.status = FrontierStatus.ATTEMPTED
        record.attempt_count += 1
        record.last_update_step = int(step)

    def mark_resolved(
        self, frontier_id: str, resolved_by: str, step: int
    ) -> None:
        record = self.get(frontier_id)
        record.status = FrontierStatus.RESOLVED
        record.resolved_by = resolved_by
        record.last_update_step = int(step)

    def mark_superseded(
        self, frontier_id: str, superseded_by: str, step: int
    ) -> None:
        record = self.get(frontier_id)
        record.status = FrontierStatus.SUPERSEDED
        record.superseded_by = superseded_by
        record.last_update_step = int(step)

    def mark_disabled(self, frontier_id: str, step: int) -> None:
        record = self.get(frontier_id)
        record.status = FrontierStatus.DISABLED
        record.last_update_step = int(step)

    def resolve_near_gateway(
        self,
        source_submap_id: str,
        gateway_local_xy: np.ndarray,
        destination_submap_id: str,
        radius_m: float,
        step: int,
    ) -> List[str]:
        """Resolve only source-frontiers directly covered by a split gateway."""

        gateway = np.asarray(gateway_local_xy, dtype=np.float64)
        if gateway.shape != (2,) or not np.isfinite(gateway).all():
            raise ValueError("gateway_local_xy must be a finite XY point")
        resolved = []
        for record in self.eligible(source_submap_id):
            if np.linalg.norm(record.local_xy - gateway) <= float(radius_m):
                record.lineage = (
                    f"gateway:{source_submap_id}->{destination_submap_id}"
                )
                self.mark_resolved(record.frontier_id, destination_submap_id, step)
                resolved.append(record.frontier_id)
        return resolved
