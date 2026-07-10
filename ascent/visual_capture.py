import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ascent.decision_trace import run_metadata_from_env


def _is_enabled(value: Optional[str]) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_slug(value: str, limit: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return value[:limit] or "window"


def _as_rgb_image(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _write_rgb(path: Path, image: Any) -> bool:
    rgb = _as_rgb_image(image)
    if rgb is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(path), bgr))


def _is_stop_action(action: Any) -> bool:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action)
    if arr.size == 0:
        return False
    try:
        return int(arr.reshape(-1)[0]) == 0
    except (TypeError, ValueError):
        return False


class VisualCaptureWriter:
    def __init__(
        self,
        enabled: bool,
        output_dir: Path,
        row_id: str,
        windows: List[Dict[str, Any]],
        interval: int = 3,
        dynamic_capture: bool = True,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.row_id = row_id
        self.windows = windows
        self.interval = max(int(interval), 1)
        self.dynamic_capture = dynamic_capture
        self._index_fh = None
        self._index_writer = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "frames").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
            self._index_fh = open(self.output_dir / "visual_capture_index.csv", "w", newline="", buffering=1)
            self._index_writer = csv.DictWriter(
                self._index_fh,
                fieldnames=[
                    "row_id",
                    "env",
                    "scene_id",
                    "episode_id",
                    "step",
                    "window_types",
                    "dynamic_reason",
                    "raw_rgb_path",
                    "annotated_rgb_path",
                    "target_bbox_rgb_path",
                    "metadata_path",
                ],
            )
            self._index_writer.writeheader()
            (self.output_dir / "visual_capture_config.json").write_text(
                json.dumps(
                    {
                        "row_id": self.row_id,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "interval": self.interval,
                        "dynamic_capture": self.dynamic_capture,
                        "windows": self.windows,
                        "run_metadata": run_metadata_from_env(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    @classmethod
    def from_env(cls) -> "VisualCaptureWriter":
        enabled = _is_enabled(os.environ.get("ASCENT_VISUAL_CAPTURE"))
        if not enabled:
            return cls(
                enabled=False,
                output_dir=Path("."),
                row_id="",
                windows=[],
                interval=1,
                dynamic_capture=False,
            )

        output_dir = Path(os.environ.get("ASCENT_VISUAL_CAPTURE_DIR", "visual_capture"))
        row_id = os.environ.get("ASCENT_VISUAL_CAPTURE_ROW_ID", "")
        interval = int(os.environ.get("ASCENT_VISUAL_CAPTURE_INTERVAL", "3"))
        dynamic_capture = _is_enabled(os.environ.get("ASCENT_VISUAL_CAPTURE_DYNAMIC", "1"))
        manifest = os.environ.get("ASCENT_VISUAL_CAPTURE_MANIFEST", "")
        windows = cls._load_windows(Path(manifest), row_id) if manifest else []
        return cls(
            enabled=enabled,
            output_dir=output_dir,
            row_id=row_id,
            windows=windows,
            interval=interval,
            dynamic_capture=dynamic_capture,
        )

    @staticmethod
    def _load_windows(path: Path, row_id: str) -> List[Dict[str, Any]]:
        if not path.exists() or not row_id:
            return []
        windows: List[Dict[str, Any]] = []
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("row_id") != row_id:
                    continue
                try:
                    start = int(float(row.get("start_step", "")))
                    end = int(float(row.get("end_step", "")))
                except ValueError:
                    continue
                windows.append(
                    {
                        "window_type": row.get("window_type", ""),
                        "priority": row.get("priority", ""),
                        "start_step": start,
                        "end_step": end,
                        "reason": row.get("reason", ""),
                    }
                )
        return windows

    def close(self) -> None:
        if self._index_fh is not None:
            self._index_fh.close()
            self._index_fh = None

    def _matching_windows(self, step: int) -> List[Dict[str, Any]]:
        return [w for w in self.windows if int(w["start_step"]) <= step <= int(w["end_step"])]

    def _dynamic_reason(self, policy_info: Dict[str, Any], action: Any, done: bool) -> str:
        trace = policy_info.get("decision_trace", {})
        reasons = []
        if trace.get("target_detected"):
            reasons.append("target_detected")
        if trace.get("stop_called"):
            reasons.append("policy_stop")
        if _is_stop_action(action):
            reasons.append("action_stop")
        if done:
            reasons.append("done")
        return ";".join(reasons)

    def _should_capture(
        self,
        step: int,
        windows: List[Dict[str, Any]],
        dynamic_reason: str,
    ) -> bool:
        if dynamic_reason and self.dynamic_capture:
            return True
        for window in windows:
            if step in {int(window["start_step"]), int(window["end_step"])}:
                return True
        return bool(windows) and step % self.interval == 0

    def write_step(
        self,
        episode: Any,
        policy_info: Dict[str, Any],
        info: Dict[str, Any],
        action: Any,
        done: bool,
        env_index: int = 0,
    ) -> None:
        if not self.enabled:
            return
        trace = policy_info.get("decision_trace", {})
        try:
            step = int(policy_info.get("num_steps", trace.get("step", -1)))
        except (TypeError, ValueError):
            return
        if step < 0:
            return
        windows = self._matching_windows(step)
        dynamic_reason = self._dynamic_reason(policy_info, action, done)
        if not self._should_capture(step, windows, dynamic_reason):
            return

        window_slug = _safe_slug("__".join(w.get("window_type", "") for w in windows) or dynamic_reason)
        scene_id = str(getattr(episode, "scene_id", ""))
        episode_id = str(getattr(episode, "episode_id", ""))
        identity_slug = _safe_slug(
            f"{Path(scene_id).stem}__ep={episode_id}__env={env_index}"
        )
        stem = f"{identity_slug}__step_{step:06d}__{window_slug}"
        frame_dir = self.output_dir / "frames"
        metadata_dir = self.output_dir / "metadata"
        raw_path = frame_dir / f"{stem}__raw_rgb.jpg"
        annotated_path = frame_dir / f"{stem}__annotated_rgb.jpg"
        target_bbox_path = frame_dir / f"{stem}__target_bbox_rgb.jpg"
        metadata_path = metadata_dir / f"{stem}.json"

        raw_ok = _write_rgb(raw_path, policy_info.get("raw_rgb"))
        annotated_ok = _write_rgb(annotated_path, policy_info.get("annotated_rgb"))
        target_bbox_ok = _write_rgb(target_bbox_path, policy_info.get("target_detection_annotated_rgb"))

        metadata = {
            "row_id": self.row_id,
            "env": int(env_index),
            "scene_id": scene_id,
            "episode_id": episode_id,
            "step": step,
            "action": _to_jsonable(action),
            "done": bool(done),
            "matching_windows": windows,
            "dynamic_reason": dynamic_reason,
            "metrics": {
                key: _to_jsonable(info.get(key))
                for key in ["distance_to_goal", "success", "spl", "soft_spl", "num_steps"]
                if key in info
            },
            "decision_trace": _to_jsonable(trace),
            "detector_trace": _to_jsonable(policy_info.get("detector_trace")),
            "saved_files": {
                "raw_rgb": str(raw_path) if raw_ok else "",
                "annotated_rgb": str(annotated_path) if annotated_ok else "",
                "target_bbox_rgb": str(target_bbox_path) if target_bbox_ok else "",
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        if self._index_writer is not None:
            self._index_writer.writerow(
                {
                    "row_id": self.row_id,
                    "env": int(env_index),
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "step": step,
                    "window_types": ";".join(w.get("window_type", "") for w in windows),
                    "dynamic_reason": dynamic_reason,
                    "raw_rgb_path": str(raw_path) if raw_ok else "",
                    "annotated_rgb_path": str(annotated_path) if annotated_ok else "",
                    "target_bbox_rgb_path": str(target_bbox_path) if target_bbox_ok else "",
                    "metadata_path": str(metadata_path),
                }
            )
