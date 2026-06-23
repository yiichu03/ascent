import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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


def _safe_scalar(info: Dict[str, Any], key: str) -> Optional[float]:
    value = info.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sequence_or_empty(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def _safe_name(value: str) -> str:
    value = _scene_name(value)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_DYNAMIC_INDICATOR_MIN = 3

MAP_OUTSIDE_BGR = (255, 255, 255)
MAP_TRAVERSABLE_BGR = (226, 246, 226)
MAP_OBSTACLE_BGR = (68, 68, 68)
MAP_DYNAMIC_AS_TRAVERSABLE_BGR = (232, 248, 232)
DEFAULT_OVERLAY_RENDER_SCALE = 2.0


class DecisionTraceWriter:
    def __init__(
        self,
        enabled: bool,
        output_dir: Path,
        save_topdown: bool = False,
        topdown_interval: int = 25,
        save_topdown_overlay: bool = False,
        save_map_arrays: bool = False,
        overlay_render_scale: float = DEFAULT_OVERLAY_RENDER_SCALE,
        identity_resolution_m: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.save_topdown = save_topdown
        self.topdown_interval = max(int(topdown_interval), 0)
        self.save_topdown_overlay = save_topdown_overlay
        self.save_map_arrays = save_map_arrays
        self.overlay_render_scale = max(float(overlay_render_scale), 1.0)
        self.identity_resolution_m = max(float(identity_resolution_m), 1e-3)
        self._fh = None
        self._seen_episodes = set()
        self._paths: Dict[Tuple[str, str], List[Tuple[float, float, int, str]]] = {}
        self._frontiers: Dict[Tuple[str, str], List[Tuple[float, float, int]]] = {}
        self._topdown_paths: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._last_topdown_maps: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "top_down").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "top_down_overlay").mkdir(parents=True, exist_ok=True)
            if self.save_map_arrays:
                (self.output_dir / "top_down_map_arrays").mkdir(parents=True, exist_ok=True)
            if self.save_topdown_overlay:
                self._save_topdown_overlay_legend()
            self._fh = open(self.output_dir / "decision_trace.jsonl", "a", buffering=1)
            self._write(
                {
                    "event": "trace_start",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_dir": str(self.output_dir),
                }
            )

    @classmethod
    def from_env(cls, num_envs: int) -> "DecisionTraceWriter":
        del num_envs
        enabled = _is_enabled(os.environ.get("ASCENT_DECISION_TRACE"))
        root = os.environ.get("ASCENT_DECISION_TRACE_DIR")
        if root is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            root = os.path.join("outputs", "decision_trace", timestamp)
        save_topdown = _is_enabled(os.environ.get("ASCENT_DECISION_TRACE_SAVE_TOPDOWN"))
        topdown_interval = int(os.environ.get("ASCENT_DECISION_TRACE_TOPDOWN_INTERVAL", "25"))
        save_topdown_overlay = _is_enabled(
            os.environ.get(
                "ASCENT_DECISION_TRACE_SAVE_TOPDOWN_OVERLAY",
                "1" if save_topdown else "0",
            )
        )
        save_map_arrays = _is_enabled(
            os.environ.get(
                "ASCENT_DECISION_TRACE_SAVE_MAP_ARRAYS",
                "1" if save_topdown_overlay else "0",
            )
        )
        identity_resolution_m = float(
            os.environ.get("ASCENT_DECISION_TRACE_ID_RESOLUTION_M", "1.0")
        )
        overlay_render_scale = float(
            os.environ.get(
                "ASCENT_DECISION_TRACE_OVERLAY_SCALE",
                str(DEFAULT_OVERLAY_RENDER_SCALE),
            )
        )
        return cls(
            enabled=enabled,
            output_dir=Path(root),
            save_topdown=save_topdown,
            topdown_interval=topdown_interval,
            save_topdown_overlay=save_topdown_overlay,
            save_map_arrays=save_map_arrays,
            overlay_render_scale=overlay_render_scale,
            identity_resolution_m=identity_resolution_m,
        )

    def close(self) -> None:
        if self._fh is not None:
            self._write({"event": "trace_end", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            self._fh.close()
            self._fh = None

    def _write(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        self._fh.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")

    def write_step(
        self,
        env_index: int,
        episode: Any,
        policy_info: Dict[str, Any],
        info: Dict[str, Any],
        action: Any,
        done: bool,
    ) -> None:
        if not self.enabled:
            return

        scene_id = str(getattr(episode, "scene_id", "unknown_scene"))
        episode_id = str(getattr(episode, "episode_id", "unknown_episode"))
        key = (scene_id, episode_id)

        if key not in self._seen_episodes:
            self._seen_episodes.add(key)
            self._paths[key] = []
            self._frontiers[key] = []
            self._topdown_paths[key] = []
            self._last_topdown_maps[key] = {}
            self._write(
                {
                    "event": "episode_start",
                    "env": env_index,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "goal": policy_info.get("target_object"),
                }
            )

        trace = policy_info.get("decision_trace", {})
        trace_for_record = deepcopy(trace)
        identity_trace = self._build_identity_trace(
            trace_for_record,
            target_object=policy_info.get("target_object"),
        )
        step = int(policy_info.get("num_steps", trace.get("step", -1)))
        robot_xy = trace.get("robot_xy")
        mode = str(trace.get("mode", "unknown"))
        if isinstance(robot_xy, np.ndarray):
            robot_xy = robot_xy.tolist()
        if isinstance(robot_xy, list) and len(robot_xy) >= 2:
            self._paths[key].append((float(robot_xy[0]), float(robot_xy[1]), step, mode))

        selected_frontier = trace.get("frontier_decision", {}).get("selected_frontier")
        if isinstance(selected_frontier, np.ndarray):
            selected_frontier = selected_frontier.tolist()
        if isinstance(selected_frontier, list) and len(selected_frontier) >= 2:
            self._frontiers[key].append((float(selected_frontier[0]), float(selected_frontier[1]), step))

        metrics = {
            name: _safe_scalar(info, name)
            for name in ["distance_to_goal", "success", "spl", "soft_spl", "num_steps"]
            if _safe_scalar(info, name) is not None
        }

        topdown_trace = self._extract_topdown_trace(info.get("top_down_map"), trace, step, action)
        if topdown_trace is not None:
            self._topdown_paths[key].append(topdown_trace)
            map_array = info.get("top_down_map", {}).get("map")
            if map_array is not None:
                floor_key = self._floor_key(topdown_trace)
                self._last_topdown_maps.setdefault(key, {})[floor_key] = np.asarray(map_array).copy()

        record = {
            "event": "step",
            "env": env_index,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "step": step,
            "action": _to_jsonable(action),
            "done": bool(done),
            "metrics": metrics,
            "trace": trace_for_record,
        }
        if identity_trace:
            record["identity_trace"] = identity_trace
        if topdown_trace is not None:
            record["top_down_trace"] = topdown_trace
        self._write(record)

        if (
            self.save_topdown
            and self.topdown_interval > 0
            and "top_down_map" in info
            and step % self.topdown_interval == 0
        ):
            self._save_topdown_snapshot(scene_id, episode_id, step, info["top_down_map"])

    def write_episode_end(
        self,
        env_index: int,
        episode: Any,
        episode_stats: Dict[str, Any],
        failure_cause: str,
    ) -> None:
        if not self.enabled:
            return
        scene_id = str(getattr(episode, "scene_id", "unknown_scene"))
        episode_id = str(getattr(episode, "episode_id", "unknown_episode"))
        self._write(
            {
                "event": "episode_end",
                "env": env_index,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "failure_cause": failure_cause,
                "metrics": episode_stats,
            }
        )
        self._save_xy_summary(scene_id, episode_id)
        if self.save_map_arrays:
            self._save_topdown_map_arrays(scene_id, episode_id)
        if self.save_topdown_overlay:
            self._save_topdown_overlay(scene_id, episode_id)

    def _extract_topdown_trace(
        self,
        top_down_map: Optional[Dict[str, Any]],
        trace: Dict[str, Any],
        step: int,
        action: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(top_down_map, dict):
            return None
        agent_coords = top_down_map.get("agent_map_coord")
        if not agent_coords:
            return None
        first_agent = agent_coords[0]
        if first_agent is None or len(first_agent) < 2:
            return None
        try:
            row = int(first_agent[0])
            col = int(first_agent[1])
        except (TypeError, ValueError):
            return None
        map_array = top_down_map.get("map")
        map_shape = list(np.asarray(map_array).shape[:2]) if map_array is not None else None
        explore_trace = trace.get("explore_trace", {}) if isinstance(trace, dict) else {}
        return {
            "step": int(step),
            "agent_map_coord": [row, col],
            "agent_angle": top_down_map.get("agent_angle"),
            "map_shape": map_shape,
            "floor_index": trace.get("floor_index"),
            "policy_floor_index": trace.get("floor_index"),
            "agent_height": _to_jsonable(top_down_map.get("agent_height")),
            "topdown_floor_index": _to_jsonable(top_down_map.get("topdown_floor_index")),
            "topdown_floor_height": _to_jsonable(top_down_map.get("topdown_floor_height")),
            "topdown_floor_heights": _to_jsonable(top_down_map.get("topdown_floor_heights")),
            "topdown_floor_match": _to_jsonable(top_down_map.get("topdown_floor_match")),
            "mode": trace.get("mode"),
            "target_detected": bool(trace.get("target_detected")),
            "stair_flag": trace.get("stair_flag"),
            "no_frontier": bool(explore_trace.get("no_frontier")),
            "no_frontier_action": explore_trace.get("no_frontier_action"),
            "stop_action": _to_jsonable(action) == 0,
        }

    def _build_identity_trace(
        self,
        trace: Dict[str, Any],
        target_object: Optional[str],
    ) -> Dict[str, Any]:
        if not isinstance(trace, dict):
            return {}

        floor_index = trace.get("floor_index")
        robot_xy = self._as_xy(trace.get("robot_xy"))
        nav_goal = self._as_xy(trace.get("nav_goal"))
        frontier_decision = trace.get("frontier_decision") or {}
        explore_trace = trace.get("explore_trace") or {}
        selected_frontier = self._as_xy(frontier_decision.get("selected_frontier"))

        robot_submap_id = self._submap_id(robot_xy, floor_index)
        selected_frontier_id = self._coord_id("frontier", selected_frontier, floor_index)
        selected_frontier_submap_id = self._submap_id(selected_frontier, floor_index)

        candidates = []
        topk = _sequence_or_empty(frontier_decision.get("frontier_topk"))
        values = _sequence_or_empty(frontier_decision.get("frontier_values"))
        selected_rank = None
        topk_xy = [self._as_xy(point) for point in topk]
        for rank, xy in enumerate(topk_xy):
            if xy is None:
                continue
            candidate_id = self._coord_id("frontier", xy, floor_index)
            if selected_frontier_id is not None and candidate_id == selected_frontier_id:
                selected_rank = rank
                break
        selected_value = self._as_optional_float(frontier_decision.get("selected_value"))
        if selected_value is None and selected_rank is not None and selected_rank < len(values):
            selected_value = self._as_optional_float(values[selected_rank])
        selected_distance = self._distance(robot_xy, selected_frontier)
        llm_trace = frontier_decision.get("llm_trace") or {}
        llm_prompt_indices = []
        for value in _sequence_or_empty(llm_trace.get("frontier_index_list")):
            try:
                llm_prompt_indices.append(int(value))
            except (TypeError, ValueError):
                continue
        llm_selected_rank = None
        try:
            llm_selected_rank = int(llm_trace.get("selected_frontier_rank"))
        except (TypeError, ValueError):
            llm_selected_rank = None
        last_frontier = self._as_xy(frontier_decision.get("last_frontier"))
        force_frontier = self._as_xy(frontier_decision.get("force_frontier"))

        for rank, xy in enumerate(topk_xy):
            if xy is None:
                continue
            value = values[rank] if rank < len(values) else None
            value_float = self._as_optional_float(value)
            candidate_id = self._coord_id("frontier", xy, floor_index)
            is_selected = selected_frontier_id is not None and candidate_id == selected_frontier_id
            distance_to_robot = self._distance(robot_xy, xy)
            prompt_order = None
            if rank in llm_prompt_indices:
                prompt_order = llm_prompt_indices.index(rank)
            candidates.append(
                {
                    "rank": rank,
                    "candidate_type": "frontier",
                    "candidate_id": candidate_id,
                    "submap_id": self._submap_id(xy, floor_index),
                    "xy": list(xy),
                    "value": _to_jsonable(value),
                    "value_delta_from_selected": (
                        float(value_float - selected_value)
                        if value_float is not None and selected_value is not None
                        else None
                    ),
                    "distance_to_robot": distance_to_robot,
                    "distance_delta_from_selected": (
                        float(distance_to_robot - selected_distance)
                        if distance_to_robot is not None and selected_distance is not None
                        else None
                    ),
                    "distance_to_selected_frontier": self._distance(xy, selected_frontier),
                    "rank_delta_from_selected": rank - selected_rank if selected_rank is not None else None,
                    "is_selected": bool(is_selected),
                    "same_submap_as_robot": self._submap_id(xy, floor_index) == robot_submap_id,
                    "same_submap_as_selected": self._submap_id(xy, floor_index) == selected_frontier_submap_id,
                    "same_as_last_frontier": self._same_xy(xy, last_frontier),
                    "same_as_force_frontier": self._same_xy(xy, force_frontier),
                    "in_llm_prompt": prompt_order is not None,
                    "llm_prompt_order": prompt_order,
                    "is_llm_response_rank": rank == llm_selected_rank if llm_selected_rank is not None else None,
                    "selection_source": frontier_decision.get("selection_source"),
                }
            )

        no_frontier = bool(explore_trace.get("no_frontier"))
        transition_active = bool(
            trace.get("stair_flag")
            or trace.get("reach_stair")
            or trace.get("reach_stair_centroid")
            or trace.get("mode") == "climb_stair"
            or no_frontier
        )
        transition_anchor = nav_goal if nav_goal is not None else robot_xy
        transition_id = (
            self._coord_id("transition", transition_anchor, floor_index)
            if transition_active
            else None
        )

        object_evidence_id = None
        if trace.get("target_detected") and target_object:
            target = str(target_object).split("|")[0]
            object_evidence_id = f"object:{target}@{robot_submap_id or 'unknown_submap'}"

        return {
            "schema": "identity_v0",
            "identity_resolution_m": self.identity_resolution_m,
            "floor_index": floor_index,
            "robot_submap_id": robot_submap_id,
            "nav_goal_submap_id": self._submap_id(nav_goal, floor_index),
            "selected_frontier_id": selected_frontier_id,
            "selected_frontier_submap_id": selected_frontier_submap_id,
            "selected_frontier_rank": selected_rank,
            "frontier_candidates": candidates,
            "transition_active": transition_active,
            "transition_id": transition_id,
            "object_evidence_id": object_evidence_id,
        }

    def _as_xy(self, value: Any) -> Optional[Tuple[float, float]]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    def _as_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, dict)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _same_xy(
        self,
        a: Optional[Tuple[float, float]],
        b: Optional[Tuple[float, float]],
    ) -> bool:
        distance = self._distance(a, b)
        return bool(distance is not None and distance <= self.identity_resolution_m * 0.5)

    def _coord_id(
        self,
        prefix: str,
        xy: Optional[Tuple[float, float]],
        floor_index: Any,
    ) -> Optional[str]:
        if xy is None:
            return None
        floor = "unknown" if floor_index is None else str(floor_index)
        x_bin = int(np.round(xy[0] / self.identity_resolution_m))
        y_bin = int(np.round(xy[1] / self.identity_resolution_m))
        return f"{prefix}:floor={floor}:x={x_bin}:y={y_bin}"

    def _submap_id(
        self,
        xy: Optional[Tuple[float, float]],
        floor_index: Any,
    ) -> Optional[str]:
        return self._coord_id("submap", xy, floor_index)

    def _distance(
        self,
        a: Optional[Tuple[float, float]],
        b: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def _save_topdown_snapshot(self, scene_id: str, episode_id: str, step: int, top_down_map: Dict[str, Any]) -> None:
        map_array = top_down_map.get("map") if isinstance(top_down_map, dict) else None
        if map_array is None:
            return
        image = np.asarray(map_array)
        if image.ndim == 2:
            image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        filename = f"{_safe_name(scene_id)}_ep={_safe_name(episode_id)}_step={step:04d}.png"
        cv2.imwrite(str(self.output_dir / "top_down" / filename), image)

    def _save_topdown_overlay(self, scene_id: str, episode_id: str) -> None:
        key = (scene_id, episode_id)
        maps_by_floor = self._last_topdown_maps.get(key, {})
        points = self._topdown_paths.get(key, [])
        if not maps_by_floor or len(points) < 2:
            return
        max_step = max(int(point["step"]) for point in points)
        background_panels = []
        overlay_panels = []

        floor_keys = sorted(
            maps_by_floor,
            key=lambda item: (item == "unknown", item),
        )
        for floor_key in floor_keys:
            floor_points = [point for point in points if self._floor_key(point) == floor_key]
            if len(floor_points) < 2:
                continue
            canvas = self._render_overlay_background(maps_by_floor[floor_key])
            self._draw_floor_title(
                canvas,
                f"floor={floor_key} steps={floor_points[0]['step']}-{floor_points[-1]['step']}",
            )
            background_panels.append(canvas.copy())

            for prev, cur in zip(floor_points[:-1], floor_points[1:]):
                prev_coord = prev.get("agent_map_coord")
                cur_coord = cur.get("agent_map_coord")
                if not prev_coord or not cur_coord:
                    continue
                color = self._step_color(int(cur["step"]), max_step)
                cv2.line(
                    canvas,
                    self._scale_map_pixel(prev_coord),
                    self._scale_map_pixel(cur_coord),
                    color,
                    thickness=max(1, int(round(1.3 * self.overlay_render_scale))),
                    lineType=cv2.LINE_AA,
                )

            for point in floor_points:
                coord = point.get("agent_map_coord")
                if not coord:
                    continue
                pixel = self._scale_map_pixel(coord)
                if point.get("target_detected"):
                    cv2.drawMarker(canvas, pixel, (220, 40, 220), cv2.MARKER_CROSS, self._scaled_size(9), 1)
                if point.get("stair_flag") or point.get("mode") == "climb_stair":
                    cv2.drawMarker(canvas, pixel, (180, 60, 190), cv2.MARKER_TRIANGLE_UP, self._scaled_size(9), 1)
                if point.get("no_frontier"):
                    cv2.drawMarker(canvas, pixel, (20, 20, 20), cv2.MARKER_TILTED_CROSS, self._scaled_size(8), 1)
                if point.get("stop_action"):
                    half = self._scaled_size(4)
                    cv2.rectangle(canvas, (pixel[0] - half, pixel[1] - half), (pixel[0] + half, pixel[1] + half), (40, 40, 230), -1)

            start = floor_points[0]["agent_map_coord"]
            end = floor_points[-1]["agent_map_coord"]
            cv2.circle(canvas, self._scale_map_pixel(start), self._scaled_size(5), (40, 180, 40), -1)
            cv2.circle(canvas, self._scale_map_pixel(end), self._scaled_size(5), (40, 40, 230), -1)
            overlay_panels.append(canvas)

        if not overlay_panels:
            return

        background = self._stack_panels(background_panels)
        bg_filename = f"{_safe_name(scene_id)}_ep={_safe_name(episode_id)}_topdown_background_white_bg.png"
        cv2.imwrite(str(self.output_dir / "top_down_overlay" / bg_filename), background)

        canvas = self._add_colorbar_sidebar(self._stack_panels(overlay_panels), max_step)

        filename = f"{_safe_name(scene_id)}_ep={_safe_name(episode_id)}_topdown_step_color_white_bg.png"
        cv2.imwrite(str(self.output_dir / "top_down_overlay" / filename), canvas)

    def _save_topdown_map_arrays(self, scene_id: str, episode_id: str) -> None:
        key = (scene_id, episode_id)
        maps_by_floor = self._last_topdown_maps.get(key, {})
        if not maps_by_floor:
            return

        floor_keys = sorted(
            maps_by_floor,
            key=lambda item: (item == "unknown", item),
        )
        payload: Dict[str, Any] = {
            "floor_keys": np.array(floor_keys, dtype=str),
        }
        for idx, floor_key in enumerate(floor_keys):
            payload[f"map_floor_{idx}"] = np.asarray(maps_by_floor[floor_key], dtype=np.uint8)

        filename = f"{_safe_name(scene_id)}_ep={_safe_name(episode_id)}_topdown_map_arrays.npz"
        np.savez_compressed(self.output_dir / "top_down_map_arrays" / filename, **payload)

    def _draw_floor_title(self, canvas: np.ndarray, title: str) -> None:
        cv2.putText(
            canvas,
            title,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    def _floor_key(self, point: Dict[str, Any]) -> str:
        floor_index = point.get("topdown_floor_index")
        if floor_index is None:
            floor_index = point.get("floor_index")
        return "unknown" if floor_index is None else str(floor_index)

    def _semantic_background_map(self, map_array: np.ndarray) -> np.ndarray:
        raw = np.asarray(map_array)
        if raw.ndim == 3:
            gray = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = raw
        gray = gray.astype(np.uint8, copy=False)

        canvas = np.full((*gray.shape, 3), MAP_OUTSIDE_BGR, dtype=np.uint8)
        traversable = gray == MAP_VALID_POINT
        obstacle_or_border = gray == MAP_BORDER_INDICATOR
        dynamic_annotation = gray >= MAP_DYNAMIC_INDICATOR_MIN

        canvas[traversable] = MAP_TRAVERSABLE_BGR
        canvas[dynamic_annotation] = MAP_DYNAMIC_AS_TRAVERSABLE_BGR
        canvas[obstacle_or_border] = MAP_OBSTACLE_BGR
        return canvas

    def _render_overlay_background(self, map_array: np.ndarray) -> np.ndarray:
        canvas = self._semantic_background_map(map_array)
        if self.overlay_render_scale <= 1.01:
            return canvas
        height, width = canvas.shape[:2]
        size = (
            int(round(width * self.overlay_render_scale)),
            int(round(height * self.overlay_render_scale)),
        )
        return cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST)

    def _scale_map_pixel(self, coord: Any) -> Tuple[int, int]:
        scale = self.overlay_render_scale
        return (
            int(round((float(coord[1]) + 0.5) * scale)),
            int(round((float(coord[0]) + 0.5) * scale)),
        )

    def _scaled_size(self, value: int) -> int:
        return max(1, int(round(value * self.overlay_render_scale)))

    def _step_color(self, step: int, max_step: int) -> Tuple[int, int, int]:
        value = int(np.clip(step / max(max_step, 1), 0, 1) * 255)
        color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        return int(color[0]), int(color[1]), int(color[2])

    def _stack_panels(self, panels: List[np.ndarray]) -> np.ndarray:
        max_width = max(panel.shape[1] for panel in panels)
        padded = []
        for panel in panels:
            if panel.shape[1] < max_width:
                pad = np.full((panel.shape[0], max_width - panel.shape[1], 3), 255, dtype=np.uint8)
                panel = np.concatenate([panel, pad], axis=1)
            padded.append(panel)
        if len(padded) == 1:
            return padded[0]
        separator = np.full((16, max_width, 3), 255, dtype=np.uint8)
        stacked = []
        for idx, panel in enumerate(padded):
            if idx:
                stacked.append(separator)
            stacked.append(panel)
        return np.concatenate(stacked, axis=0)

    def _add_colorbar_sidebar(self, canvas: np.ndarray, max_step: int) -> np.ndarray:
        height, width = canvas.shape[:2]
        sidebar_width = 72
        out = np.full((height, width + sidebar_width, 3), 255, dtype=np.uint8)
        out[:, :width] = canvas

        bar_width = 8
        x0 = width + 14
        y0 = 28
        bar_height = max(36, min(150, height - 56))
        y1 = min(y0 + bar_height, height - 18)
        for y in range(y0, y1):
            ratio = 1.0 - (y - y0) / max(y1 - y0, 1)
            color = self._step_color(int(ratio * max_step), max_step)
            out[y, x0 : x0 + bar_width] = color
        cv2.rectangle(out, (x0, y0), (x0 + bar_width, y1), (30, 30, 30), 1)
        cv2.putText(out, "step", (width + 8, max(y0 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        cv2.putText(out, str(max_step), (width + 34, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1)
        cv2.putText(out, "0", (width + 34, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1)
        return out

    def _save_topdown_overlay_legend(self) -> None:
        legend_path = self.output_dir / "top_down_overlay" / "topdown_overlay_legend.png"
        canvas = make_topdown_overlay_legend()
        cv2.imwrite(str(legend_path), canvas)

    def _save_xy_summary(self, scene_id: str, episode_id: str) -> None:
        key = (scene_id, episode_id)
        points = self._paths.get(key, [])
        if len(points) < 2:
            return
        frontiers = self._frontiers.get(key, [])
        canvas = np.full((800, 800, 3), 255, dtype=np.uint8)

        xy = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        if frontiers:
            frontier_xy = np.array([(p[0], p[1]) for p in frontiers], dtype=np.float32)
            all_xy = np.concatenate([xy, frontier_xy], axis=0)
        else:
            all_xy = xy
        mins = all_xy.min(axis=0)
        maxs = all_xy.max(axis=0)
        span = np.maximum(maxs - mins, 1e-3)

        def project(point: Tuple[float, float]) -> Tuple[int, int]:
            norm = (np.array(point, dtype=np.float32) - mins) / span
            px = int(60 + norm[0] * 680)
            py = int(740 - norm[1] * 680)
            return px, py

        for frontier in frontiers:
            cv2.circle(
                canvas,
                project((frontier[0], frontier[1])),
                3,
                (190, 150, 95),
                1,
                cv2.LINE_AA,
            )

        max_step = max((int(p[2]) for p in points), default=len(points))
        for prev, cur in zip(points[:-1], points[1:]):
            cv2.line(
                canvas,
                project((prev[0], prev[1])),
                project((cur[0], cur[1])),
                self._step_color(int(cur[2]), max_step),
                2,
                cv2.LINE_AA,
            )
        start = project((points[0][0], points[0][1]))
        end = project((points[-1][0], points[-1][1]))
        cv2.circle(canvas, start, 6, (0, 170, 0), -1)
        cv2.circle(canvas, end, 6, (0, 0, 220), -1)

        title = f"{_scene_name(scene_id)} ep={episode_id} steps={len(points)}"
        cv2.putText(canvas, title[:90], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(canvas, "path: blue/purple early -> orange/red late; pale orange=selected frontier", (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        filename = f"{_safe_name(scene_id)}_ep={_safe_name(episode_id)}_trajectory_xy.png"
        cv2.imwrite(str(self.output_dir / "figures" / filename), canvas)


def _legend_text(canvas: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), thickness, cv2.LINE_AA)


def _legend_step_color(step: int, max_step: int) -> Tuple[int, int, int]:
    value = int(np.clip(step / max(max_step, 1), 0, 1) * 255)
    color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def make_topdown_overlay_legend() -> np.ndarray:
    """Return a BGR legend image for decision-trace top-down overlays."""
    canvas = np.full((700, 1040, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1039, 699), (220, 220, 220), 2)

    _legend_text(canvas, "Top-down overlay legend", 28, 42, scale=0.95, thickness=2)
    _legend_text(canvas, "File is copied into every top_down_overlay/ folder generated by Line B traces.", 30, 75, scale=0.52)

    # Map background swatches.
    y = 122
    _legend_text(canvas, "Map/background", 30, y, scale=0.68, thickness=2)
    for label, color in (
        ("white: invalid / outside saved top-down map", MAP_OUTSIDE_BGR),
        ("light green: traversable top-down cells", MAP_TRAVERSABLE_BGR),
        ("dark gray: Habitat border / obstacle context", MAP_OBSTACLE_BGR),
    ):
        y += 38
        cv2.rectangle(canvas, (42, y - 20), (90, y + 8), color, -1)
        cv2.rectangle(canvas, (42, y - 20), (90, y + 8), (90, 90, 90), 1)
        _legend_text(canvas, label, 110, y, scale=0.54)

    # Path and colorbar.
    y = 285
    _legend_text(canvas, "Trajectory/time", 30, y, scale=0.68, thickness=2)
    x0 = 52
    prev = (x0, y + 42)
    for idx, step in enumerate(range(0, 501, 50)):
        x = x0 + idx * 28
        pt = (x, y + 42 + int(16 * np.sin(idx / 1.2)))
        if idx:
            cv2.line(canvas, prev, pt, _legend_step_color(step, 500), 5, cv2.LINE_AA)
        prev = pt
    cv2.circle(canvas, (x0, y + 42), 7, (40, 180, 40), -1)
    cv2.circle(canvas, prev, 8, (40, 40, 230), -1)
    _legend_text(canvas, "colored line: agent path, blue/purple early -> orange/red late", 360, y + 28, scale=0.54)
    _legend_text(canvas, "green circle: start    red circle: episode end", 360, y + 58, scale=0.54)

    bar_x, bar_y0, bar_y1 = 890, 118, 308
    for yy in range(bar_y0, bar_y1):
        ratio = 1.0 - (yy - bar_y0) / max(bar_y1 - bar_y0, 1)
        canvas[yy, bar_x : bar_x + 24] = _legend_step_color(int(ratio * 500), 500)
    cv2.rectangle(canvas, (bar_x, bar_y0), (bar_x + 24, bar_y1), (30, 30, 30), 1)
    _legend_text(canvas, "step", bar_x - 4, bar_y0 - 12, scale=0.48)
    _legend_text(canvas, "late", bar_x + 34, bar_y0 + 16, scale=0.42)
    _legend_text(canvas, "early", bar_x + 34, bar_y1 - 8, scale=0.42)

    # Event markers.
    y = 405
    _legend_text(canvas, "Event markers", 30, y, scale=0.68, thickness=2)
    marker_rows = [
        ("target_detected", "magenta cross: target/object evidence seen", "cross"),
        ("stair / climb_stair", "purple triangle: stair flag or climb-stair mode", "triangle"),
        ("no_frontier", "black tilted cross: no active frontier / fallback context", "tilted"),
        ("stop_action", "red square: STOP action issued by policy", "square"),
        ("panel label", "text: floor id and first-last step range for that panel", "label"),
    ]
    for idx, (_, label, kind) in enumerate(marker_rows):
        row_y = y + 42 + idx * 38
        center = (62, row_y - 8)
        if kind == "cross":
            cv2.drawMarker(canvas, center, (220, 40, 220), cv2.MARKER_CROSS, 18, 2)
        elif kind == "triangle":
            cv2.drawMarker(canvas, center, (180, 60, 190), cv2.MARKER_TRIANGLE_UP, 18, 2)
        elif kind == "tilted":
            cv2.drawMarker(canvas, center, (20, 20, 20), cv2.MARKER_TILTED_CROSS, 18, 2)
        elif kind == "square":
            cv2.rectangle(canvas, (center[0] - 8, center[1] - 8), (center[0] + 8, center[1] + 8), (40, 40, 230), -1)
        else:
            cv2.putText(canvas, "floor=0 steps=1-500", (42, row_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        label_x = 250 if kind == "label" else 130
        _legend_text(canvas, label, label_x, row_y, scale=0.54)

    _legend_text(canvas, "Notes:", 30, 660, scale=0.48, thickness=1)
    _legend_text(canvas, "multi-floor episodes are stacked as separate panels; axes are image/map pixels, not metric coordinates.", 95, 660, scale=0.48)
    return canvas
