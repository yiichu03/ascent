import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ascent import zero_shot_controls as zs

_STATE: Dict[Tuple[str, int], Dict[str, Any]] = {}

_ALIASES = {
    '20260706_I01_FRONTIER_OBJECT_REASONER': '20260706_I01D_FOM_STYLE_FRONTIER_OBJECT_REASONER',
    '20260706_I01D_FRONTIER_OBJECT_PATH_REASONER': '20260706_I01D_FOM_STYLE_FRONTIER_OBJECT_REASONER',
    '20260706_I01_FOM_STYLE_FRONTIER_OBJECT_REASONER': '20260706_I01D_FOM_STYLE_FRONTIER_OBJECT_REASONER',
}

_META = {
    'I01': ('20260706_I01_FOM_STYLE_FRONTIER_OBJECT_REASONER', 'FrontierObjectReasoner', 'FrontierObjectMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'frontier-object-path structured reasoning'),
    'I02': ('20260706_I02_WORLD_MODEL_LOOKAHEAD_REASONER', 'WorldModelLookaheadReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'world-model lookahead'),
    'I03': ('20260706_I03_ANTICIPATED_LAYOUT_BELLMAN_PLANNER', 'AscentReasoner', 'AscentMemory', 'AnticipatedBellmanFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'Bellman-style frontier planning'),
    'I04': ('20260706_I04_CONTINUOUS_TRANSITION_SPACE_ABLATION', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'LocalTransitionGraph', 'None', 'None', 'None', 'local continuous transition planning'),
    'I05': ('20260706_I05_OBJECT_CENTRIC_SEMANTIC_MEMORY_REPLACE_MSS', 'ObjectCentricReasoner', 'ObjectCardMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'object-centric semantic memory'),
    'I06': ('20260706_I06_MLLM_EXECUTE_REVIEW_POLICY_REPLACE_ASCENT_LLM', 'ExecuteReviewReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'execute-review high-level policy'),
    'I07': ('20260706_I07_CAUSE_TYPED_RECOVERY_MANAGER', 'CauseTypedReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'CauseTypedRecovery', 'None', 'None', 'cause-aware recovery'),
    'I08': ('20260706_I08_TOPOLOGICAL_FRONTIER_GRAPH_MEMORY', 'AscentReasoner', 'TopologicalObjectGraphMemory', 'TopoGraphFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'topological semantic frontier planning'),
    'I09': ('20260706_I09_ACTIVE_TARGET_VERIFICATION_OPTION', 'VerificationReasoner', 'ObjectCardMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'ActiveTargetVerification', 'None', 'active target verification option'),
    'I10': ('20260706_I10_DISAGREEMENT_TRIGGERED_DELIBERATION', 'DisagreementDeliberationReasoner', 'ObjectCardMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'None', 'DisagreementTriggeredDeliberation', 'adaptive high-level reasoning scheduler'),
}

REPRESENTATIVE_VARIANTS = [
    '20260706_I01D_FOM_STYLE_FRONTIER_OBJECT_REASONER',
    '20260706_I02D_WORLD_MODEL_RISK_AWARE',
    '20260706_I03D_ANTICIPATED_BELLMAN_TOP3',
    '20260706_I04B_LOCAL_TRANSITION_GRAPH',
    '20260706_I05C_MSS_PLUS_OBJECT_CARDS',
    '20260706_I06C_MLLM_PLAN_REVIEW',
    '20260706_I07C_CAUSE_TYPED_RECOVERY',
    '20260706_I08D_TOPO_GRAPH_OBJECT_ANCHORS',
    '20260706_I09C_VERIFY_APPROACH_OPTION',
    '20260706_I10C_FRONTIER_OBJECT_DISAGREE_DELIBERATION',
]

BATCH2_VARIANTS = [
    '20260707_B01_CURRENT_TARGET_STOP_090',
    '20260707_B02_CURRENT_TARGET_STOP_080',
    '20260707_B03_CURRENT_TARGET_STOP_100',
    '20260707_B04_RECENT_TARGET_STOP_090',
    '20260707_B05_RECENT_TARGET_STOP_110',
    '20260707_B06_APPROACH_THEN_STOP_070',
    '20260707_B07_APPROACH_THEN_STOP_085',
    '20260707_B08_TARGET_SCAN_THEN_STOP_090',
    '20260707_B09_RELAXED_BLIP_TARGET_STOP_095',
    '20260707_B10_MIN_DISTANCE_PLATEAU_STOP_100',
]

BATCH3_VARIANTS = [
    '20260707_C01_UPSTAIR_IF_VISIBLE_FLOOR3',
    '20260707_C02_UPSTAIR_AFTER_INIT_FLOOR3',
    '20260707_C03_UPSTAIR_IGNORE_TARGET_FLOOR3',
    '20260707_C04_UPSTAIR_UNTIL_FLOOR2_THEN_ASCENT',
    '20260707_C05_UPSTAIR_UNTIL_FLOOR3_THEN_ASCENT',
    '20260707_C06_UPSTAIR_WITH_FAST_STAIR_RECOVERY',
    '20260707_C07_MARK_LOW_FLOORS_EXPLORED_THEN_UP',
    '20260707_C08_UPSTAIR_SCAN_LOOKUP_BIAS',
    '20260707_C09_UPSTAIR_RETRY_IF_STUCK',
    '20260707_C10_FLOOR3_FIRST_THEN_TARGET_STOP',
]

BATCH4_VARIANTS = [
    '20260707_D01_FAST_FIRST_UPSTAIR',
    '20260707_D02_FORCE_SECOND_UPSTAIR',
    '20260707_D03_MARK_FLOOR_EXPLORED_AFTER_60',
    '20260707_D04_IGNORE_LOW_FLOOR_TARGET_AND_UP',
    '20260707_D05_UPPER_FLOOR_SECOND_STAIR_PRIORITY',
    '20260707_D06_AGGRESSIVE_STAIR_PROGRESS',
    '20260707_D07_UPPER_FLOOR_QUICK_COMPLETE_THEN_UP',
    '20260707_D08_BLOCK_DOWNSTAIRS_WHILE_BELOW_TARGET',
    '20260707_D09_RETRY_SECOND_STAIR_IF_STUCK',
    '20260707_D10_FLOOR2_FAST_SEARCH_AND_STOP',
]

BATCH5_VARIANTS = [
    '20260707_E01_FAST_FIRST_UPSTAIR_EXT700',
    '20260707_E02_IGNORE_LOW_TARGET_FAST_UP_EXT700',
    '20260707_E03_UPPER_UNSEEN_FRONTIER_EXT700',
    '20260707_E04_UPPER_BED_PRIOR_FRONTIER_EXT700',
    '20260707_E05_UPPER_FAR_FRONTIER_EXT700',
    '20260707_E06_FLOOR3_FIRST_EXT900',
    '20260707_E07_FLOOR3_FAST_STAIR_EXT900',
    '20260707_E08_UPPER_NEAREST_COVERAGE_EXT700',
    '20260707_E09_UPPER_ONLY_TARGET_STOP_EXT700',
    '20260707_E10_ORACLE_BED_FRONTIER_DIAGNOSTIC_EXT700',
]

_META.update({
    'B01': ('20260707_B01_CURRENT_TARGET_STOP_090', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'TargetSeenStop090', 'None', 'current target evidence stop at 0.90m'),
    'B02': ('20260707_B02_CURRENT_TARGET_STOP_080', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'TargetSeenStop080', 'None', 'current target evidence stop at 0.80m'),
    'B03': ('20260707_B03_CURRENT_TARGET_STOP_100', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'TargetSeenStop100', 'None', 'current target evidence stop at 1.00m'),
    'B04': ('20260707_B04_RECENT_TARGET_STOP_090', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'RecentTargetStop090', 'None', 'recent target evidence stop at 0.90m'),
    'B05': ('20260707_B05_RECENT_TARGET_STOP_110', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'RecentTargetStop110', 'None', 'recent target evidence stop at 1.10m'),
    'B06': ('20260707_B06_APPROACH_THEN_STOP_070', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'ApproachThenStop070', 'None', 'force approach then stop at 0.70m'),
    'B07': ('20260707_B07_APPROACH_THEN_STOP_085', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'ApproachThenStop085', 'None', 'force approach then stop at 0.85m'),
    'B08': ('20260707_B08_TARGET_SCAN_THEN_STOP_090', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'ScanThenStop090', 'None', 'one-step scan before close target stop'),
    'B09': ('20260707_B09_RELAXED_BLIP_TARGET_STOP_095', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'RelaxedBlipStop095', 'None', 'relaxed BLIP target stop at 0.95m'),
    'B10': ('20260707_B10_MIN_DISTANCE_PLATEAU_STOP_100', 'AscentReasoner', 'RecentTargetMemory', 'AscentFrontierScore', 'AscentTransition', 'None', 'MinDistancePlateauStop100', 'None', 'stop after recent target distance stops improving'),
})

_META.update({
    'D01': ('20260707_D01_FAST_FIRST_UPSTAIR', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'fast first upstairs transition'),
    'D02': ('20260707_D02_FORCE_SECOND_UPSTAIR', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'ForceSecondUpstairs', 'None', 'None', 'None', 'prioritize second upstairs transition'),
    'D03': ('20260707_D03_MARK_FLOOR_EXPLORED_AFTER_60', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'QuickFloorComplete', 'None', 'None', 'None', 'mark floors explored after 60 steps to expose stairs'),
    'D04': ('20260707_D04_IGNORE_LOW_FLOOR_TARGET_AND_UP', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'IgnoreLowFloorTargetAndUp', 'None', 'IgnoreLowFloorTarget', 'None', 'ignore low-floor target evidence and continue upstairs'),
    'D05': ('20260707_D05_UPPER_FLOOR_SECOND_STAIR_PRIORITY', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpperFloorSecondStairPriority', 'None', 'None', 'None', 'on upper floor prioritize next upstairs frontier'),
    'D06': ('20260707_D06_AGGRESSIVE_STAIR_PROGRESS', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'AggressiveStairProgress', 'AggressiveStairRecovery', 'None', 'None', 'force progress when approaching upstairs stalls'),
    'D07': ('20260707_D07_UPPER_FLOOR_QUICK_COMPLETE_THEN_UP', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpperFloorQuickComplete', 'None', 'None', 'None', 'quickly complete upper floor and go upstairs again'),
    'D08': ('20260707_D08_BLOCK_DOWNSTAIRS_WHILE_BELOW_TARGET', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'BlockDownstairsBelowTarget', 'None', 'None', 'None', 'block downstairs while target floor not reached'),
    'D09': ('20260707_D09_RETRY_SECOND_STAIR_IF_STUCK', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'RetrySecondStairIfStuck', 'RetryStairRecovery', 'None', 'None', 'retry upstairs if floor index does not progress'),
    'D10': ('20260707_D10_FLOOR2_FAST_SEARCH_AND_STOP', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'Floor2FastSearch', 'None', 'UpperFloorTargetStop', 'None', 'upper-floor search and stop after fast upstairs'),
})

_META.update({
    'C01': ('20260707_C01_UPSTAIR_IF_VISIBLE_FLOOR3', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsIfVisible', 'None', 'None', 'None', 'go upstairs whenever up-stair frontier is visible'),
    'C02': ('20260707_C02_UPSTAIR_AFTER_INIT_FLOOR3', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsAfterInit', 'None', 'None', 'None', 'after initial scan, prioritize upstairs'),
    'C03': ('20260707_C03_UPSTAIR_IGNORE_TARGET_FLOOR3', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsIgnoreLowFloorTarget', 'None', 'IgnoreLowFloorTarget', 'None', 'ignore low-floor target detections and prioritize upstairs'),
    'C04': ('20260707_C04_UPSTAIR_UNTIL_FLOOR2_THEN_ASCENT', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsUntilFloor2', 'None', 'None', 'None', 'go upstairs until floor index at least 1'),
    'C05': ('20260707_C05_UPSTAIR_UNTIL_FLOOR3_THEN_ASCENT', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsUntilFloor3', 'None', 'None', 'None', 'go upstairs until floor index at least 2'),
    'C06': ('20260707_C06_UPSTAIR_WITH_FAST_STAIR_RECOVERY', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsFastRecovery', 'FastStairRecovery', 'None', 'None', 'go upstairs with aggressive stair retry/recovery'),
    'C07': ('20260707_C07_MARK_LOW_FLOORS_EXPLORED_THEN_UP', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'LowFloorExploredThenUp', 'None', 'None', 'None', 'deprioritize current low floor to force upstairs exploration'),
    'C08': ('20260707_C08_UPSTAIR_SCAN_LOOKUP_BIAS', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsLookUpScan', 'None', 'None', 'None', 'look upward/scan before pursuing upstairs'),
    'C09': ('20260707_C09_UPSTAIR_RETRY_IF_STUCK', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'UpstairsRetryIfStuck', 'RetryRecovery', 'None', 'None', 'retry upstairs when floor index fails to increase'),
    'C10': ('20260707_C10_FLOOR3_FIRST_THEN_TARGET_STOP', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'Floor3First', 'None', 'Floor3TargetStop', 'None', 'prioritize floor3, then allow target stop'),
})

_META.update({
    'E01': ('20260707_E01_FAST_FIRST_UPSTAIR_EXT700', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'diagnostic: D01-style first-upstairs policy with 700-step horizon'),
    'E02': ('20260707_E02_IGNORE_LOW_TARGET_FAST_UP_EXT700', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastFirstUpstairs', 'None', 'IgnoreLowFloorTarget', 'None', 'ignore low-floor target evidence, then fast first upstairs, 700-step horizon'),
    'E03': ('20260707_E03_UPPER_UNSEEN_FRONTIER_EXT700', 'AscentReasoner', 'AscentMemory', 'UpperUnseenFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'after first upstairs, prefer frontiers with large local unseen area'),
    'E04': ('20260707_E04_UPPER_BED_PRIOR_FRONTIER_EXT700', 'AscentReasoner', 'AscentMemory', 'UpperBedPriorFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'after first upstairs, prefer bedroom/bed-related frontier evidence'),
    'E05': ('20260707_E05_UPPER_FAR_FRONTIER_EXT700', 'AscentReasoner', 'AscentMemory', 'UpperFarFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'after first upstairs, prefer long-range upper-floor expansion'),
    'E06': ('20260707_E06_FLOOR3_FIRST_EXT900', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'Floor3First', 'None', 'None', 'None', 'diagnostic: continue upstairs toward internal floor index 2 with 900-step horizon'),
    'E07': ('20260707_E07_FLOOR3_FAST_STAIR_EXT900', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'Floor3FastStair', 'FastStairRecovery', 'None', 'None', 'diagnostic: floor3-first plus fast stair recovery with 900-step horizon'),
    'E08': ('20260707_E08_UPPER_NEAREST_COVERAGE_EXT700', 'AscentReasoner', 'AscentMemory', 'UpperNearestCoverageScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'after first upstairs, prefer efficient nearby upper-floor coverage'),
    'E09': ('20260707_E09_UPPER_ONLY_TARGET_STOP_EXT700', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastFirstUpstairs', 'None', 'UpperOnlyTargetStop', 'None', 'ignore target evidence before first upstairs, then allow normal target stop'),
    'E10': ('20260707_E10_ORACLE_BED_FRONTIER_DIAGNOSTIC_EXT700', 'AscentReasoner', 'OracleBedMemory', 'OracleBedFrontierScore', 'FastFirstUpstairs', 'None', 'None', 'None', 'diagnostic only: after first upstairs, bias frontier choice toward known bed goal xz positions'),
})


def normalize_variant(raw: Optional[str] = None) -> str:
    value = (raw if raw is not None else zs.variant() or '').strip().upper()
    return _ALIASES.get(value, value)


def family(raw: Optional[str] = None) -> str:
    value = normalize_variant(raw)
    for pattern in (r'20260706_(I\d\d)', r'20260707_([BCDE]\d\d)'):
        m = re.search(pattern, value)
        if m:
            return m.group(1)
    m = re.search(r'\b(I\d\d|B\d\d|C\d\d|D\d\d|E\d\d)\b', value)
    return m.group(1) if m else ''


def is_paradigm_variant(raw: Optional[str] = None) -> bool:
    return family(raw) in _META


def reset_env(env: int) -> None:
    for key in list(_STATE.keys()):
        if key[1] == env:
            del _STATE[key]


def _state(name: str, env: int) -> Dict[str, Any]:
    return _STATE.setdefault((name, env), {})


def variant_metadata(raw: Optional[str] = None) -> Dict[str, Any]:
    fam = family(raw)
    data: Dict[str, Any] = {
        'variant_id': normalize_variant(raw),
        'case_id': zs.case_id(),
        'family': fam,
        'active': fam in _META,
    }
    if fam in _META:
        idea, reasoner, memory, scorer, transition, recovery, verification, deliberation, paradigm = _META[fam]
        data.update({
            'idea_id': idea,
            'reasoner_adapter': reasoner,
            'map_memory_adapter': memory,
            'frontier_scoring_adapter': scorer,
            'transition_adapter': transition,
            'recovery_adapter': recovery,
            'verification_adapter': verification,
            'deliberation_adapter': deliberation,
            'paradigm': paradigm,
        })
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8', errors='replace')).hexdigest()[:12]


def _same_xy(a: Any, b: Any, eps: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    try:
        aa = np.asarray(a, dtype=float).reshape(-1)[:2]
        bb = np.asarray(b, dtype=float).reshape(-1)[:2]
        return bool(np.linalg.norm(aa - bb) <= eps)
    except Exception:
        return False


def _room_objects(object_map: Any, env: int, step: Any) -> Tuple[str, List[str]]:
    if step is None:
        return 'unknown', []
    try:
        env_map = object_map[env]
        room_value = env_map.each_step_rooms.get(step, 'unknown')
        object_value = env_map.each_step_objects.get(step, [])
    except Exception:
        return 'unknown', []
    room = ', '.join(str(x) for x in list(room_value)[:5]) if isinstance(room_value, (list, tuple, set)) else str(room_value or 'unknown')
    if isinstance(object_value, str):
        objects = [x.strip() for x in object_value.split(',') if x.strip()]
    elif isinstance(object_value, (list, tuple, set)):
        objects = [str(x) for x in list(object_value)]
    else:
        objects = []
    return room or 'unknown', objects[:12]


def _unseen_area_proxy(obstacle_map: Any, env: int, frontier: np.ndarray) -> float:
    try:
        omap = obstacle_map[env]
        px = omap._xy_to_px(np.atleast_2d(frontier[:2]))[0]
        x, y = int(px[0]), int(px[1])
        explored = np.asarray(omap.explored_area)
        radius = max(int(getattr(omap, 'pixels_per_meter', 20) * 1.5), 8)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, explored.shape[0])
        x0, x1 = max(x - radius, 0), min(x + radius + 1, explored.shape[1])
        patch = explored[y0:y1, x0:x1]
        return float(np.count_nonzero(patch == 0)) / float(max(patch.size, 1))
    except Exception:
        return 0.0


def _object_memory(object_map: Any, env: int, target: str) -> Dict[str, Any]:
    try:
        env_map = object_map[env]
    except Exception:
        return {'target': target, 'this_floor_objects': [], 'this_floor_rooms': [], 'object_cards': []}
    rooms = sorted(str(x) for x in getattr(env_map, 'this_floor_rooms', set()))[:20]
    objects = sorted(str(x) for x in getattr(env_map, 'this_floor_objects', set()))[:40]
    cards = []
    for name, cloud in list(getattr(env_map, 'clouds', {}).items())[:16]:
        arr = np.asarray(cloud)
        cards.append({'category': str(name), 'point_count': int(arr.shape[0]) if arr.ndim else 0, 'target_match': str(name).split('|')[0] == target})
    return {'target': target, 'this_floor_rooms': rooms, 'this_floor_objects': objects, 'object_cards': cards}


def _frontier_cards(obs_cache: List[dict], obstacle_map: Any, object_map: Any, sorted_pts: np.ndarray, sorted_values: List[float], env: int, topk: int) -> List[Dict[str, Any]]:
    robot_xy = np.asarray(obs_cache[env].get('robot_xy', np.zeros(2)))[:2]
    cards: List[Dict[str, Any]] = []
    for idx, frontier in enumerate(list(sorted_pts[: max(topk, 1)])):
        xy = np.asarray(frontier, dtype=float).reshape(-1)[:2]
        try:
            step, _ = obstacle_map[env].extract_frontiers_with_image(xy)
        except Exception:
            step = None
        room, objects = _room_objects(object_map, env, step)
        try:
            repeat_count = int(obstacle_map[env]._best_frontier_selection_count.get(tuple(xy), 0))
            disabled = tuple(xy) in obstacle_map[env]._disabled_frontiers
        except Exception:
            repeat_count, disabled = 0, False
        cards.append({
            'id': idx + 1,
            'sorted_index': idx,
            'xy': [round(float(xy[0]), 3), round(float(xy[1]), 3)],
            'mval': _safe_float(sorted_values[idx] if idx < len(sorted_values) else 0.0),
            'distance_to_robot': round(float(np.linalg.norm(xy - robot_xy)), 3),
            'scene_step': step,
            'room': room,
            'objects': objects,
            'repeat_count': repeat_count,
            'disabled': bool(disabled),
            'expected_unseen_area': round(_unseen_area_proxy(obstacle_map, env, xy), 4),
        })
    return cards


def _format_cards(cards: List[Dict[str, Any]]) -> str:
    rows = []
    for c in cards:
        rows.append(
            f"{c['id']}. xy={c['xy']}, mval={c['mval']:.4f}, dist={c['distance_to_robot']:.2f}, "
            f"room={c['room']}, objects={', '.join(c['objects'][:8]) or 'none'}, "
            f"repeat={c['repeat_count']}, unseen={c['expected_unseen_area']:.3f}"
        )
    return '\n'.join(rows)


def _extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not text or text == '-1':
        return None, True
    cleaned = str(text).replace('\n', ' ').replace('\r', ' ').strip()
    candidates = [cleaned]
    left, right = cleaned.find('{'), cleaned.rfind('}')
    if left >= 0 and right > left:
        candidates.insert(0, cleaned[left:right + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, False
    return None, True


def _select_index(parsed: Optional[Dict[str, Any]], n: int) -> Tuple[int, bool]:
    if not parsed:
        return 0, True
    raw = parsed.get('Index') or parsed.get('index') or parsed.get('frontier_id') or parsed.get('choice')
    try:
        idx = int(str(raw).strip()) - 1
    except (TypeError, ValueError):
        return 0, True
    if idx < 0 or idx >= max(n, 1):
        return 0, True
    return idx, False


def _call_llm(planner: Any, prompt: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    trace = {'llm_called': 1, 'prompt_hash': _short_hash(prompt), 'prompt_chars': len(prompt), 'parse_failure': 0, 'fallback': 0}
    try:
        response = planner._llm.chat(prompt)
    except Exception as exc:
        trace.update({'raw_output_hash': '', 'raw_output_excerpt': repr(exc)[:500], 'parse_failure': 1, 'fallback': 1})
        return None, trace
    parsed, failed = _extract_json(str(response))
    trace.update({'raw_output_hash': _short_hash(str(response)), 'raw_output_excerpt': str(response)[:500], 'parse_failure': int(failed), 'fallback': int(failed)})
    return parsed, trace


def _prompt_for(fam: str, target: str, cards: List[Dict[str, Any]], memory: Dict[str, Any], original_id: int) -> str:
    if fam == 'I02':
        instruction = 'Predict likely next region, target relevance, redundancy risk, transition risk, and wrong-stop risk for each frontier; choose the best future outcome.'
        schema = '{"Index":"1","Reason":"...","Predictions":[{"frontier":1,"next_region":"...","target_relevance":"...","risk":"..."}]}'
    elif fam == 'I05':
        instruction = 'Use object-centric memory plus Mval. Prefer stable object/room evidence related to the target.'
        schema = '{"Index":"1","Reason":"...","ObjectEvidence":"..."}'
    elif fam == 'I06':
        instruction = 'Act as a high-level execute-review controller. Review likely progress and choose the next frontier option.'
        schema = '{"Index":"1","Plan":"...","Review":"...","Reason":"..."}'
    elif fam == 'I10':
        instruction = 'Cheap frontier score and memory signals may disagree. Deliberate and decide whether to keep or switch the frontier.'
        schema = '{"Index":"1","Reason":"...","DisagreementResolution":"..."}'
    else:
        instruction = 'Compare frontier-object-path evidence. Avoid repeated low-value exploration and choose the most useful frontier for finding the target.'
        schema = '{"Index":"1","Reason":"...","GoalRelation":"..."}'
    memory_text = json.dumps(_jsonable(memory), ensure_ascii=False)[:1600]
    return (
        'You are a high-level ObjectNav frontier selector.\n'
        f'Target object: {target}\n'
        f'Original ASCENT selected candidate id: {original_id}\n'
        f'Memory summary: {memory_text}\n'
        'Candidate frontiers:\n'
        f'{_format_cards(cards)}\n'
        f'Instruction: {instruction}\n'
        'Return only valid JSON with schema:\n'
        f'{schema}'
    )


def _bellman_choice(cards: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
    scored = []
    for c in cards[:3]:
        q = _safe_float(c.get('mval')) + 0.8 * _safe_float(c.get('expected_unseen_area')) - 0.015 * _safe_float(c.get('distance_to_robot')) - 0.2 * _safe_float(c.get('repeat_count'))
        item = dict(c)
        item['bellman_q'] = round(q, 5)
        scored.append(item)
    if not scored:
        return 0, []
    return max(range(len(scored)), key=lambda i: scored[i]['bellman_q']), scored


def _topo_choice(cards: List[Dict[str, Any]], env: int, target: str) -> Tuple[int, List[Dict[str, Any]]]:
    st = _state('topology', env)
    visited = st.setdefault('visited_cells', {})
    scored = []
    for c in cards:
        xy = c.get('xy', [0, 0])
        cell = f"{round(float(xy[0]) / 2.0)}:{round(float(xy[1]) / 2.0)}"
        novelty = 0.0 if cell in visited else 1.0
        object_anchor = 1.0 if target.lower() in {str(x).lower() for x in c.get('objects', [])} else 0.0
        score = _safe_float(c.get('mval')) + 0.45 * novelty + 0.3 * _safe_float(c.get('expected_unseen_area')) + 0.25 * object_anchor - 0.15 * _safe_float(c.get('repeat_count'))
        item = dict(c)
        item.update({'topo_cell': cell, 'topo_novelty': novelty, 'object_anchor': object_anchor, 'topo_score': round(score, 5)})
        scored.append(item)
    if not scored:
        return 0, []
    best = max(range(len(scored)), key=lambda i: scored[i]['topo_score'])
    visited[scored[best]['topo_cell']] = visited.get(scored[best]['topo_cell'], 0) + 1
    return best, scored


def _cause_choice(cards: List[Dict[str, Any]], original_idx: int, frontier_stick_step: int) -> Tuple[int, Dict[str, Any]]:
    cause = 'low_gain_search'
    if frontier_stick_step >= 4 or any(_safe_float(c.get('repeat_count')) >= 2 for c in cards):
        cause = 'frontier_loop'
    chosen = original_idx
    if cause == 'frontier_loop' and len(cards) > 1:
        candidates = [(i, _safe_float(c.get('repeat_count')), -_safe_float(c.get('expected_unseen_area')), -_safe_float(c.get('mval'))) for i, c in enumerate(cards) if i != original_idx]
        chosen = min(candidates, key=lambda x: (x[1], x[2], x[3]))[0]
    return chosen, {'cause_label': cause, 'cause_confidence': 0.65 if cause == 'frontier_loop' else 0.45, 'recovery_event_count': int(cause == 'frontier_loop')}


def _disagreement(cards: List[Dict[str, Any]], original_idx: int) -> Tuple[bool, Dict[str, Any]]:
    if len(cards) < 2:
        return False, {'trigger_type': 'insufficient_candidates'}
    margin = abs(_safe_float(cards[0].get('mval')) - _safe_float(cards[1].get('mval')))
    repeated = _safe_float(cards[original_idx].get('repeat_count')) >= 1 if original_idx < len(cards) else False
    triggered = margin < 0.03 or repeated
    trigger_type = 'frontier_repeat' if repeated else ('low_mval_margin' if margin < 0.03 else 'none')
    return triggered, {'trigger_type': trigger_type, 'mval_margin': round(margin, 5), 'original_repeat': int(repeated)}


_ORACLE_BED_GOAL_XZ = [
    (5.69853, -10.20639),
    (0.49813, -10.18748),
    (-0.84827, -4.88821),
]


def _upper_floor_batch5_choice(cards: List[Dict[str, Any]], fam: str, target: str) -> Tuple[int, List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []
    target_terms = {target.lower(), 'bed', 'bedroom', 'bed room', 'sleeping'}
    for c in cards:
        distance = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        repeat = _safe_float(c.get('repeat_count'))
        room = str(c.get('room', '')).lower()
        objects = {str(x).lower() for x in c.get('objects', [])}
        object_hit = 1.0 if objects.intersection(target_terms) else 0.0
        room_hit = 1.0 if any(term in room for term in target_terms) else 0.0
        oracle_min = None
        if fam == 'E10':
            xy = c.get('xy', [0.0, 0.0])
            try:
                fx, fz = float(xy[0]), float(xy[1])
                oracle_min = min(math.hypot(fx - gx, fz - gz) for gx, gz in _ORACLE_BED_GOAL_XZ)
            except Exception:
                oracle_min = 99.0
        if fam == 'E03':
            score = 1.20 * unseen + 0.20 * mval - 0.015 * distance - 0.25 * repeat
        elif fam == 'E04':
            score = 1.40 * object_hit + 0.80 * room_hit + 0.50 * unseen + 0.20 * mval - 0.10 * repeat
        elif fam == 'E05':
            score = 0.08 * distance + 0.45 * unseen + 0.10 * mval - 0.20 * repeat
        elif fam == 'E08':
            score = -0.12 * distance + 0.55 * unseen + 0.20 * mval - 0.15 * repeat
        elif fam == 'E10':
            score = -0.18 * _safe_float(oracle_min) + 0.30 * unseen + 0.10 * mval - 0.12 * repeat
        else:
            score = mval
        item = dict(c)
        item.update({
            'batch5_score': round(float(score), 5),
            'target_object_hit': object_hit,
            'target_room_hit': room_hit,
            'oracle_bed_xz_min_distance': round(float(oracle_min), 3) if oracle_min is not None else None,
        })
        scored.append(item)
    if not scored:
        return 0, []
    return max(range(len(scored)), key=lambda i: scored[i]['batch5_score']), scored

def maybe_override_frontier(planner: Any, observations_cache: List[dict], obstacle_map: Any, value_map: Any, object_map: Any, sorted_pts: np.ndarray, sorted_values: List[float], env: int, topk: int, original_frontier: Any, original_value: float, selection_source: str, frontier_stick_step: Optional[List[int]] = None, current_floor_index: Optional[int] = None) -> Tuple[Any, float, str, Dict[str, Any]]:
    fam = family()
    if fam not in _META:
        return original_frontier, original_value, selection_source, {}
    if sorted_pts is None or len(sorted_pts) == 0:
        trace = {**variant_metadata(), 'adapter_stage': 'frontier_selection', 'fallback': 1, 'fallback_reason': 'no_sorted_frontiers'}
        return original_frontier, original_value, selection_source, trace
    target = str(planner._target_object[env]).split('|')[0]
    cards = _frontier_cards(observations_cache, obstacle_map, object_map, sorted_pts, sorted_values, env, topk)
    original_idx = 0
    for idx, frontier in enumerate(list(sorted_pts[: len(cards)])):
        if _same_xy(frontier, original_frontier):
            original_idx = idx
            break
    memory = _object_memory(object_map, env, target)
    trace: Dict[str, Any] = {**variant_metadata(), 'adapter_stage': 'frontier_selection', 'current_floor_index': int(current_floor_index) if current_floor_index is not None else None, 'original_selection_source': selection_source, 'original_frontier': _jsonable(original_frontier), 'original_value': _safe_float(original_value), 'original_frontier_id': original_idx + 1, 'frontier_cards': _jsonable(cards), 'object_memory': _jsonable(memory), 'llm_called': 0, 'parse_failure': 0, 'fallback': 0, 'override': 0}
    if not cards:
        trace.update({'fallback': 1, 'fallback_reason': 'no_frontier_cards'})
        return original_frontier, original_value, selection_source, trace
    selected_idx = original_idx
    source = selection_source
    if fam == 'I03':
        selected_idx, scored = _bellman_choice(cards)
        trace['frontier_bellman_scores'] = _jsonable(scored)
        source = 'anticipated_bellman'
    elif fam == 'I08':
        selected_idx, scored = _topo_choice(cards, env, target)
        trace['topological_scores'] = _jsonable(scored)
        source = 'topological_graph'
    elif fam == 'I07':
        stick = int(frontier_stick_step[env]) if frontier_stick_step is not None and env < len(frontier_stick_step) else 0
        selected_idx, cause_trace = _cause_choice(cards, original_idx, stick)
        trace.update(cause_trace)
        source = 'cause_typed_recovery'
    elif fam in {'I01', 'I02', 'I05', 'I06'}:
        parsed, llm_trace = _call_llm(planner, _prompt_for(fam, target, cards, memory, original_idx + 1))
        trace.update(llm_trace)
        trace['parsed_output'] = _jsonable(parsed or {})
        selected_idx, bad = _select_index(parsed, len(cards))
        if bad:
            selected_idx = original_idx
            trace.update({'fallback': 1, 'fallback_reason': 'bad_or_missing_index'})
        source = trace.get('reasoner_adapter', 'paradigm_reasoner')
    elif fam in {'E03', 'E04', 'E05', 'E08', 'E10'}:
        floor_idx = int(current_floor_index) if current_floor_index is not None else -1
        if floor_idx < 1:
            trace['batch5_upper_floor_inactive'] = 1
            return original_frontier, original_value, selection_source, trace
        selected_idx, scored = _upper_floor_batch5_choice(cards, fam, target)
        trace['batch5_upper_floor_scores'] = _jsonable(scored)
        source = 'batch5_upper_floor_frontier'
    elif fam == 'I10':
        triggered, trigger_trace = _disagreement(cards, original_idx)
        trace.update({'deliberation_triggered': int(triggered), **trigger_trace})
        if triggered:
            parsed, llm_trace = _call_llm(planner, _prompt_for(fam, target, cards, memory, original_idx + 1))
            trace.update(llm_trace)
            trace['parsed_output'] = _jsonable(parsed or {})
            selected_idx, bad = _select_index(parsed, len(cards))
            if bad:
                selected_idx = original_idx
                trace.update({'fallback': 1, 'fallback_reason': 'bad_or_missing_index'})
            source = 'disagreement_deliberation'
    selected_idx = max(0, min(selected_idx, len(cards) - 1))
    sorted_index = int(cards[selected_idx].get('sorted_index', selected_idx))
    try:
        selected_frontier = sorted_pts[sorted_index]
        selected_value = sorted_values[sorted_index] if sorted_index < len(sorted_values) else original_value
    except Exception:
        selected_frontier = original_frontier
        selected_value = original_value
        trace.update({'fallback': 1, 'fallback_reason': 'selected_index_mapping_failed'})
    trace.update({'selected_goal_type': 'frontier', 'selected_frontier_id': selected_idx + 1, 'selected_frontier': _jsonable(selected_frontier), 'selected_value': _safe_float(selected_value), 'override': int(not _same_xy(selected_frontier, original_frontier))})
    return selected_frontier, selected_value, source, trace




def floor_probe_policy(env: int, step: int, cur_floor_index: int, floor_num: int, floor_num_steps: int, has_up_stair: bool, up_frontier_count: int, target_detected: bool, climb_stair_over: bool) -> Optional[Dict[str, Any]]:
    fam = family()
    if fam not in ({f'C{i:02d}' for i in range(1, 11)} | {f'D{i:02d}' for i in range(1, 11)} | {f'E{i:02d}' for i in range(1, 11)}):
        return None
    st = _state('floor_probe', env)
    target_floor_index = 2
    if fam == 'C04':
        target_floor_index = 1
    elif fam == 'C10':
        target_floor_index = 2
    elif fam.startswith('E'):
        target_floor_index = 2 if fam in {'E06', 'E07'} else 1
    can_go_up = bool(climb_stair_over and has_up_stair and up_frontier_count > 0)
    event = None
    reason = 'no_override'
    if fam == 'C01' and can_go_up:
        event, reason = 'navigate_upstairs', 'up_stair_visible'
    elif fam == 'C02' and step >= 12 and can_go_up:
        event, reason = 'navigate_upstairs', 'after_initial_scan_up_stair_visible'
    elif fam == 'C03':
        if cur_floor_index < target_floor_index and can_go_up:
            event, reason = 'navigate_upstairs', 'ignore_low_floor_target_and_go_up'
        elif cur_floor_index < target_floor_index and target_detected:
            event, reason = 'ignore_target', 'low_floor_target_ignored'
    elif fam in {'C04', 'C05'} and cur_floor_index < target_floor_index and can_go_up:
        event, reason = 'navigate_upstairs', f'cur_floor_{cur_floor_index}_below_target_{target_floor_index}'
    elif fam == 'C06' and cur_floor_index < target_floor_index and can_go_up:
        event, reason = 'navigate_upstairs_fast_recovery', 'floor3_fast_upstairs'
    elif fam == 'C07' and cur_floor_index < target_floor_index:
        if can_go_up:
            event, reason = 'mark_current_floor_explored_and_up', 'force_low_floor_complete_then_up'
        elif floor_num_steps >= 25:
            event, reason = 'mark_current_floor_explored', 'force_low_floor_complete_wait_for_stair'
    elif fam == 'C08' and cur_floor_index < target_floor_index:
        if can_go_up:
            if not st.get('lookup_done'):
                st['lookup_done'] = True
                event, reason = 'look_up', 'scan_for_upstairs_before_stair'
            else:
                event, reason = 'navigate_upstairs', 'post_lookup_upstairs'
        elif step >= 12 and not st.get('scan_turn_done'):
            st['scan_turn_done'] = True
            event, reason = 'turn_left', 'scan_for_stair'
    elif fam == 'C09' and cur_floor_index < target_floor_index:
        last_floor = st.get('last_floor_index', cur_floor_index)
        last_progress_step = st.get('last_progress_step', step)
        if cur_floor_index != last_floor:
            st['last_floor_index'] = cur_floor_index
            st['last_progress_step'] = step
        stuck = int(step) - int(st.get('last_progress_step', step)) >= 80
        if can_go_up:
            event, reason = 'navigate_upstairs', 'retry_upstairs_if_available'
        elif stuck:
            event, reason = 'turn_left', 'stuck_without_floor_progress_scan'
    elif fam == 'C10':
        if cur_floor_index < target_floor_index and can_go_up:
            event, reason = 'navigate_upstairs', 'floor3_first'
        elif cur_floor_index < target_floor_index and target_detected:
            event, reason = 'ignore_target', 'target_before_floor3_ignored'
    elif fam == 'D01' and cur_floor_index < 1 and can_go_up:
        event, reason = 'navigate_upstairs', 'fast_first_upstairs'
    elif fam == 'D02' and cur_floor_index < 2 and can_go_up:
        event, reason = 'navigate_upstairs', 'force_second_upstairs_if_visible'
    elif fam == 'D03' and cur_floor_index < 2:
        if can_go_up:
            event, reason = 'navigate_upstairs', 'quick_complete_then_up'
        elif floor_num_steps >= 60:
            event, reason = 'mark_current_floor_explored', 'floor_steps_ge_60'
    elif fam == 'D04':
        if cur_floor_index < 2 and target_detected:
            event, reason = 'ignore_target', 'low_floor_target_ignored'
        elif cur_floor_index < 2 and can_go_up:
            event, reason = 'navigate_upstairs', 'ignore_target_and_upstairs'
    elif fam == 'D05' and cur_floor_index >= 1 and cur_floor_index < 2 and can_go_up:
        event, reason = 'navigate_upstairs', 'upper_floor_second_stair_priority'
    elif fam == 'D06' and cur_floor_index < 2 and can_go_up:
        event, reason = 'navigate_upstairs_fast_recovery', 'aggressive_stair_progress_upstairs'
    elif fam == 'D07' and cur_floor_index < 2:
        if can_go_up:
            event, reason = 'navigate_upstairs', 'upper_quick_complete_then_up'
        elif cur_floor_index >= 1 and floor_num_steps >= 35:
            event, reason = 'mark_current_floor_explored', 'upper_floor_steps_ge_35'
    elif fam == 'D08' and cur_floor_index < 2:
        event, reason = 'block_downstairs', 'block_downstairs_before_target_floor'
        if can_go_up:
            event, reason = 'block_downstairs_and_up', 'block_downstairs_then_up'
    elif fam == 'D09' and cur_floor_index < 2:
        if can_go_up:
            event, reason = 'navigate_upstairs_fast_recovery', 'retry_upstairs_if_available'
        elif floor_num_steps >= 45:
            event, reason = 'turn_left', 'scan_for_second_stair_after_stuck'
    elif fam == 'D10':
        if cur_floor_index < 1 and can_go_up:
            event, reason = 'navigate_upstairs', 'floor2_first'
        elif cur_floor_index == 1 and floor_num_steps >= 120 and can_go_up:
            event, reason = 'navigate_upstairs', 'late_second_upstairs'
    elif fam in {'E01', 'E03', 'E04', 'E05', 'E08', 'E10'} and cur_floor_index < 1 and can_go_up:
        event, reason = 'navigate_upstairs', 'batch5_fast_first_upstairs'
    elif fam in {'E02', 'E09'}:
        if cur_floor_index < 1 and target_detected:
            event, reason = 'ignore_target', 'batch5_ignore_low_floor_target'
        elif cur_floor_index < 1 and can_go_up:
            event, reason = 'navigate_upstairs', 'batch5_fast_first_upstairs_after_ignore'
    elif fam == 'E06' and cur_floor_index < 2 and can_go_up:
        event, reason = 'navigate_upstairs', 'batch5_floor3_first'
    elif fam == 'E07' and cur_floor_index < 2 and can_go_up:
        event, reason = 'navigate_upstairs_fast_recovery', 'batch5_floor3_fast_stair'
    if event is None:
        return None
    st['event_count'] = int(st.get('event_count', 0)) + 1
    return {
        **variant_metadata(),
        'adapter_stage': 'floor_probe',
        'floor_probe_event': event,
        'floor_probe_reason': reason,
        'floor_probe_event_count': st['event_count'],
        'cur_floor_index': int(cur_floor_index),
        'target_floor_index': int(target_floor_index),
        'floor_num': int(floor_num),
        'floor_num_steps': int(floor_num_steps),
        'has_up_stair': bool(has_up_stair),
        'up_frontier_count': int(up_frontier_count),
        'target_detected': bool(target_detected),
        'climb_stair_over': bool(climb_stair_over),
        'override': 1,
        'selected_goal_type': 'upstairs_probe',
    }



def stair_progress_policy(env: int, step: int, cur_floor_index: int, climb_stair_flag: int, get_close_step: int, frontier_stick_step: int, distance_to_stair: Any, reach_stair: bool, reach_stair_centroid: bool) -> Optional[Dict[str, Any]]:
    fam = family()
    if fam not in {'D02', 'D05', 'D06', 'D09', 'D10', 'E07'}:
        return None
    if climb_stair_flag != 1:
        return None
    d = _safe_float(distance_to_stair, None)
    threshold = 18
    if fam in {'D06', 'D09', 'E07'}:
        threshold = 8
    elif fam in {'D02', 'D05'}:
        threshold = 12
    event = None
    reason = 'no_override'
    if not reach_stair and (get_close_step >= threshold or frontier_stick_step >= threshold):
        event, reason = 'force_reach_upstairs', f'get_close_or_stick_ge_{threshold}'
    elif reach_stair and not reach_stair_centroid and fam in {'D06', 'D09', 'E07'} and get_close_step >= threshold:
        event, reason = 'force_reach_centroid', 'aggressive_centroid_skip'
    elif fam == 'D10' and cur_floor_index >= 1 and d is not None and d <= 1.4:
        event, reason = 'force_reach_upstairs', 'upper_floor_close_to_stair'
    if event is None:
        return None
    st = _state('stair_progress', env)
    st['event_count'] = int(st.get('event_count', 0)) + 1
    return {
        **variant_metadata(),
        'adapter_stage': 'stair_progress',
        'stair_progress_event': event,
        'stair_progress_reason': reason,
        'stair_progress_event_count': st['event_count'],
        'cur_floor_index': int(cur_floor_index),
        'climb_stair_flag': int(climb_stair_flag),
        'get_close_step': int(get_close_step),
        'frontier_stick_step': int(frontier_stick_step),
        'distance_to_stair': d,
        'reach_stair': bool(reach_stair),
        'reach_stair_centroid': bool(reach_stair_centroid),
        'override': 1,
        'selected_goal_type': 'upstairs_stair_progress',
    }


def maybe_target_policy_action(env: int, step: int, target_detected: bool, cur_distance: Any, blip_cosine: Any, navigate_step: int) -> Optional[Dict[str, Any]]:
    """Batch-2 hm3d_r0_006 target-evidence stop/approach interventions.

    Returns a trace with policy_action in {stop, move_forward, turn_left} when a
    default-off batch-2 variant wants to override ASCENT's normal target handling.
    """
    fam = family()
    if fam not in {f'B{i:02d}' for i in range(1, 11)}:
        return None
    d = _safe_float(cur_distance, None)
    st = _state('target_policy', env)
    prev_min = _safe_float(st.get('min_recent_distance'), None)
    last_seen_step = st.get('last_target_step')
    if target_detected and d is not None:
        st['last_target_step'] = int(step)
        st['last_target_distance'] = d
        if prev_min is None or d < prev_min:
            st['min_recent_distance'] = d
            st['min_recent_step'] = int(step)
    recent_age = None
    if st.get('last_target_step') is not None:
        recent_age = int(step) - int(st['last_target_step'])
    min_recent = _safe_float(st.get('min_recent_distance'), None)
    action = None
    reason = 'no_override'

    def current_stop(threshold: float) -> bool:
        return bool(target_detected and d is not None and d <= threshold)

    def recent_stop(threshold: float, grace: int) -> bool:
        return bool(recent_age is not None and recent_age <= grace and min_recent is not None and min_recent <= threshold)

    if fam == 'B01' and current_stop(0.90):
        action, reason = 'stop', 'current_target_distance_le_0.90'
    elif fam == 'B02' and current_stop(0.80):
        action, reason = 'stop', 'current_target_distance_le_0.80'
    elif fam == 'B03' and current_stop(1.00):
        action, reason = 'stop', 'current_target_distance_le_1.00'
    elif fam == 'B04' and recent_stop(0.90, 6):
        action, reason = 'stop', 'recent_target_min_distance_le_0.90'
    elif fam == 'B05' and recent_stop(1.10, 8):
        action, reason = 'stop', 'recent_target_min_distance_le_1.10'
    elif fam == 'B06':
        if recent_stop(0.70, 6):
            action, reason = 'stop', 'recent_target_min_distance_le_0.70'
        elif target_detected and d is not None and d <= 1.20:
            action, reason = 'move_forward', 'approach_target_until_0.70'
    elif fam == 'B07':
        if recent_stop(0.85, 6):
            action, reason = 'stop', 'recent_target_min_distance_le_0.85'
        elif target_detected and d is not None and d <= 1.30:
            action, reason = 'move_forward', 'approach_target_until_0.85'
    elif fam == 'B08':
        scanned = bool(st.get('scan_done'))
        if target_detected and d is not None and d <= 0.90 and not scanned:
            st['scan_done'] = True
            action, reason = 'turn_left', 'scan_once_before_stop'
        elif scanned and recent_stop(0.90, 4):
            action, reason = 'stop', 'post_scan_recent_target_stop_0.90'
    elif fam == 'B09' and target_detected and d is not None and d <= 0.95 and _safe_float(blip_cosine, 0.0) >= 0.12:
        action, reason = 'stop', 'target_distance_le_0.95_blip_ge_0.12'
    elif fam == 'B10':
        history = st.setdefault('distance_history', [])
        if d is not None:
            history.append((int(step), d, bool(target_detected)))
            del history[:-6]
        if recent_stop(1.00, 8):
            worsening = d is None or (min_recent is not None and d >= min_recent + 0.10)
            plateau = len(history) >= 3 and max(x[1] for x in history[-3:] if x[1] is not None) - min(x[1] for x in history[-3:] if x[1] is not None) < 0.06
            if worsening or plateau or not target_detected:
                action, reason = 'stop', 'recent_target_min_distance_plateau_or_worsen'

    if action is None:
        return None
    st['event_count'] = int(st.get('event_count', 0)) + 1
    return {
        **variant_metadata(),
        'adapter_stage': 'target_policy',
        'policy_action': action,
        'target_policy_reason': reason,
        'target_policy_event_count': st['event_count'],
        'target_detected': bool(target_detected),
        'cur_distance': _safe_float(cur_distance, None),
        'min_recent_distance': min_recent,
        'min_recent_step': st.get('min_recent_step'),
        'recent_target_age': recent_age,
        'blip_cosine': _safe_float(blip_cosine, None),
        'navigate_step': int(navigate_step),
        'override': 1,
        'selected_goal_type': 'target_object',
    }


def maybe_defer_stop_for_verification(env: int, step: int, goal: Any, cur_distance: float, blip_cosine: float, confirm_stop: bool) -> Optional[Dict[str, Any]]:
    if family() != 'I09' or not confirm_stop:
        return None
    st = _state('verification', env)
    try:
        key = tuple(round(float(x), 2) for x in np.asarray(goal).reshape(-1)[:2])
    except Exception:
        key = ('unknown',)
    if st.get('verified_goal') == key:
        return None
    if cur_distance <= 0.45 or step >= 490:
        st['verified_goal'] = key
        return {**variant_metadata(), 'adapter_stage': 'verification', 'verification_event': 'accept_without_defer', 'verify_event_count': int(st.get('verify_event_count', 0)), 'cur_distance': _safe_float(cur_distance), 'blip_cosine': _safe_float(blip_cosine)}
    st['verified_goal'] = key
    st['verify_event_count'] = int(st.get('verify_event_count', 0)) + 1
    return {**variant_metadata(), 'adapter_stage': 'verification', 'verification_event': 'defer_stop_for_active_verify', 'verify_event_count': st['verify_event_count'], 'cur_distance': _safe_float(cur_distance), 'blip_cosine': _safe_float(blip_cosine), 'override': 1, 'selected_goal_type': 'target_verify'}


def transition_event(env: int, direction: str, robot_xy: Any, candidates: Any, selected: Any) -> Dict[str, Any]:
    st = _state('transition', env)
    st['transition_attempt_count'] = int(st.get('transition_attempt_count', 0)) + 1
    return {**variant_metadata(), 'adapter_stage': 'transition', 'transition_adapter_active': int(family() == 'I04'), 'transition_direction': direction, 'transition_attempt_count': st['transition_attempt_count'], 'robot_xy': _jsonable(robot_xy), 'local_transition_graph_nodes': _jsonable(candidates), 'selected_transition': _jsonable(selected)}


def remember_policy_event(env: int, trace: Optional[Dict[str, Any]]) -> None:
    if trace:
        _state('policy_event', env)['last'] = trace


def compose_step_trace(env: int, frontier_decision: Dict[str, Any], mode: str, action: Any, stop_called: bool, final_distance: Any = None) -> Dict[str, Any]:
    trace = {**variant_metadata()}
    frontier_trace = frontier_decision.get('paradigm_sweep') if isinstance(frontier_decision, dict) else None
    policy_trace = _state('policy_event', env).get('last')
    trace.update({'mode': mode, 'action': _jsonable(action), 'stop_called': bool(stop_called), 'final_distance_observed': _safe_float(final_distance, None) if final_distance is not None else None, 'adapter_decisions': {}, 'llm_call_count': 0, 'fallback_count': 0, 'parse_fail_count': 0, 'transition_attempt_count': int(_state('transition', env).get('transition_attempt_count', 0)), 'verify_event_count': int(_state('verification', env).get('verify_event_count', 0)), 'recovery_event_count': 0})
    if frontier_trace:
        trace['adapter_decisions']['frontier'] = _jsonable(frontier_trace)
        trace['llm_call_count'] += int(frontier_trace.get('llm_called', 0))
        trace['fallback_count'] += int(frontier_trace.get('fallback', 0))
        trace['parse_fail_count'] += int(frontier_trace.get('parse_failure', 0))
        trace['recovery_event_count'] += int(frontier_trace.get('recovery_event_count', 0))
    if policy_trace:
        trace['adapter_decisions'][str(policy_trace.get('adapter_stage', 'policy'))] = _jsonable(policy_trace)
        trace['fallback_count'] += int(policy_trace.get('fallback', 0))
    return _jsonable(trace)

