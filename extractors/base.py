from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


class FeatureExtractor:
    name: str

    def required_keys(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def extract(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None
