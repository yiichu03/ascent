"""Conservative global lifecycle for submap-local frontier candidates."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ascent.submaps.types import FrontierRecord, FrontierStatus


class FrontierRegistry:
    """Registry that never merges candidates using untrusted global XY alone."""

    def __init__(self, evidence_maturity_enabled: bool = False) -> None:
        self._records: Dict[str, FrontierRecord] = {}
        self._next_index_by_submap: Dict[str, int] = {}
        self.evidence_maturity_enabled = bool(evidence_maturity_enabled)

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

    def mark_attempted(self, frontier_id: str, step: int) -> None:
        record = self.get(frontier_id)
        if not record.eligible:
            raise RuntimeError(f"Cannot attempt inactive frontier {frontier_id}")
        record.status = FrontierStatus.ATTEMPTED
        record.attempt_count += 1
        record.last_update_step = int(step)

    def mark_remote_failure(
        self,
        frontier_id: str,
        *,
        submap_id: str,
        view_local_xy: np.ndarray,
        step: int,
        reason: str,
    ) -> FrontierStatus:
        """Record bounded negative evidence from remote-frontier execution.

        With evidence maturity disabled this is exactly the v1.2 ATTEMPTED
        transition.  With it enabled, the first negative observation is only
        tentative; failure of the sole reconsideration is globally final.
        """

        record = self.get(frontier_id)
        if not self.evidence_maturity_enabled:
            self.mark_attempted(frontier_id, step)
            return record.status
        view = np.asarray(view_local_xy, dtype=np.float64)
        if view.shape != (2,) or not np.isfinite(view).all():
            raise ValueError("view_local_xy must be a finite XY point")
        if record.status is FrontierStatus.RECONSIDERING:
            self.confirm_retired(
                frontier_id,
                step=step,
                reason=f"reconsideration_failed:{reason}",
                count_attempt=True,
            )
            return record.status
        if record.status not in {
            FrontierStatus.ACTIVE,
            FrontierStatus.SELECTED,
        }:
            raise RuntimeError(
                f"Cannot suppress frontier {frontier_id} from {record.status}"
            )
        record.status = FrontierStatus.TENTATIVE_SUPPRESSION
        record.attempt_count += 1
        record.negative_support_submap_id = str(submap_id)
        record.negative_support_view_local_xy = view.copy()
        record.negative_support_step = int(step)
        record.negative_support_reason = str(reason)
        record.last_update_step = int(step)
        return record.status

    def tentative(self) -> List[FrontierRecord]:
        return sorted(
            (
                record
                for record in self._records.values()
                if record.status is FrontierStatus.TENTATIVE_SUPPRESSION
            ),
            key=lambda record: (
                -record.score,
                record.creation_step,
                record.frontier_id,
            ),
        )

    def mark_reconsidering(
        self, frontier_id: str, *, submap_id: str, step: int
    ) -> None:
        record = self.get(frontier_id)
        if record.status is not FrontierStatus.TENTATIVE_SUPPRESSION:
            raise RuntimeError(
                f"Cannot reconsider frontier {frontier_id} from {record.status}"
            )
        if record.reconsideration_count != 0:
            raise RuntimeError(
                f"Frontier {frontier_id} already used its reconsideration"
            )
        if record.negative_support_submap_id == str(submap_id):
            raise RuntimeError(
                "Frontier reconsideration requires a different submap"
            )
        record.status = FrontierStatus.RECONSIDERING
        record.reconsideration_count = 1
        record.reconsideration_submap_id = str(submap_id)
        record.selection_count += 1
        record.last_update_step = int(step)

    def confirm_retired(
        self,
        frontier_id: str,
        *,
        step: int,
        reason: str,
        count_attempt: bool = False,
    ) -> None:
        record = self.get(frontier_id)
        if record.status is FrontierStatus.CONFIRMED_RETIRED:
            return
        if count_attempt:
            record.attempt_count += 1
        record.status = FrontierStatus.CONFIRMED_RETIRED
        record.confirmed_retirement_step = int(step)
        record.confirmed_retirement_reason = str(reason)
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
