import os
from typing import Any, Dict, Iterable, Optional


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def variant() -> str:
    return _clean(os.environ.get("ASCENT_ZS_VARIANT")).upper()


def case_id() -> str:
    return _clean(
        os.environ.get("ASCENT_ZS_CASE_ID")
        or os.environ.get("ASCENT_VISUAL_CAPTURE_ROW_ID")
    )


def enabled(*names: str) -> bool:
    active = variant()
    return bool(active) and active in {name.upper() for name in names}


def any_enabled(names: Iterable[str]) -> bool:
    active = variant()
    return bool(active) and active in {name.upper() for name in names}


def case_is(*names: str) -> bool:
    active_case = case_id()
    return bool(active_case) and active_case in set(names)


def enabled_for_case(name: str, *case_names: str) -> bool:
    return enabled(name) and case_is(*case_names)


def float_env(name: str, default: float) -> float:
    raw = _clean(os.environ.get(name))
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def int_env(name: str, default: int) -> int:
    raw = _clean(os.environ.get(name))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def active_metadata(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "variant": variant(),
        "case_id": case_id(),
    }
    if extra:
        data.update(extra)
    return data
