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


def normalize_variant(raw: Optional[str] = None) -> str:
    value = (raw if raw is not None else zs.variant() or '').strip().upper()
    return _ALIASES.get(value, value)


def family(raw: Optional[str] = None) -> str:
    value = normalize_variant(raw)
    m = re.search(r'20260706_(I\d\d)', value)
    if m:
        return m.group(1)
    m = re.search(r'\b(I\d\d)\b', value)
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


def maybe_override_frontier(planner: Any, observations_cache: List[dict], obstacle_map: Any, value_map: Any, object_map: Any, sorted_pts: np.ndarray, sorted_values: List[float], env: int, topk: int, original_frontier: Any, original_value: float, selection_source: str, frontier_stick_step: Optional[List[int]] = None) -> Tuple[Any, float, str, Dict[str, Any]]:
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
    trace: Dict[str, Any] = {**variant_metadata(), 'adapter_stage': 'frontier_selection', 'original_selection_source': selection_source, 'original_frontier': _jsonable(original_frontier), 'original_value': _safe_float(original_value), 'original_frontier_id': original_idx + 1, 'frontier_cards': _jsonable(cards), 'object_memory': _jsonable(memory), 'llm_called': 0, 'parse_failure': 0, 'fallback': 0, 'override': 0}
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

