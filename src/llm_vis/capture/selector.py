"""Deterministic representative-block selection."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .types import (
    RepresentativeCandidate,
    RepresentativeKind,
    RepresentativeSelection,
)

DEFAULT_KIND_ORDER: Tuple[RepresentativeKind, ...] = (
    RepresentativeKind.DENSE,
    RepresentativeKind.FULL_ATTENTION,
    RepresentativeKind.LINEAR_STATE,
)


class RepresentativeSelector:
    """Select the earliest deterministic supported instance of each block kind."""

    def __init__(
        self,
        kinds: Sequence[RepresentativeKind] = DEFAULT_KIND_ORDER,
    ) -> None:
        self._kinds = tuple(RepresentativeKind(kind) for kind in kinds)
        if len(set(self._kinds)) != len(self._kinds):
            raise ValueError("representative kinds must be unique")

    @staticmethod
    def _sort_key(candidate: RepresentativeCandidate) -> Tuple[int, str, str]:
        layer_index = candidate.layer_index
        return (
            layer_index if layer_index is not None else 2**63 - 1,
            candidate.module_path,
            candidate.instance_id,
        )

    def select(
        self,
        candidates: Iterable[RepresentativeCandidate],
    ) -> Tuple[RepresentativeSelection, ...]:
        grouped = {kind: [] for kind in self._kinds}
        for candidate in candidates:
            if candidate.kind in grouped and candidate.supported:
                grouped[candidate.kind].append(candidate)

        selected: List[RepresentativeSelection] = []
        for kind in self._kinds:
            matches = grouped[kind]
            if not matches:
                continue
            candidate = min(matches, key=self._sort_key)
            selected.append(
                RepresentativeSelection(
                    kind=candidate.kind,
                    instance_id=candidate.instance_id,
                    module_path=candidate.module_path,
                    definition_id=candidate.definition_id,
                    layer_index=candidate.layer_index,
                )
            )
        return tuple(selected)


def select_representatives(
    candidates: Iterable[RepresentativeCandidate],
) -> Tuple[RepresentativeSelection, ...]:
    return RepresentativeSelector().select(candidates)
