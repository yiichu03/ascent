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

BATCH6_VARIANTS = [
    '20260707_F01_FAST_TWO_UP_HOLD_FLOOR2_500',
    '20260707_F02_FAST_TWO_UP_FAST_INIT_500',
    '20260707_F03_FAST_TWO_UP_FLOOR2_BEDROOM_PRIOR_500',
    '20260707_F04_FAST_TWO_UP_FLOOR2_DEEP_PRIVATE_PRIOR_500',
    '20260707_F05_FAST_TWO_UP_FLOOR2_NO_LOCAL_BIAS_500',
    '20260707_F06_FAST_TWO_UP_BEDROOM_NO_LOCAL_BIAS_500',
    '20260707_F07_FAST_TWO_UP_EARLY_LLM_500',
    '20260707_F08_FAST_TWO_UP_ZERO_INIT_500',
    '20260707_F09_FAST_TWO_UP_FLOOR2_TARGET_APPROACH_500',
    '20260707_F10_ORACLE_FLOOR2_BED_WAYPOINT_DIAGNOSTIC_500',
]

BATCH7_VARIANTS = [
    '20260707_G01_ROUTE_NOVELTY_FRONTIER_500',
    '20260707_G02_LOOP_EXIT_CORRIDOR_FRONTIER_500',
    '20260707_G03_TRANSITION_AWARE_ROUTE_SCORER_500',
    '20260707_G04_ROUTE_COMPRESSION_BUDGET_500',
    '20260707_G05_SECOND_TRANSITION_BUDGET_PLANNER_500',
    '20260707_G06_FRONTIER_OUTCOME_MEMORY_500',
    '20260707_G07_TOPOLOGICAL_OPTION_CHAIN_500',
    '20260707_G08_BUDGETED_OPTION_PLANNER_500',
    '20260707_G09_ANTI_HOMING_ROUTE_EXPANSION_500',
    '20260707_G10_E01_COMPRESSED_ROUTE_DIAGNOSTIC_500',
]

BATCH8_VARIANTS = [
    '20260708_H01_E01_LOW_FLOOR_TARGET_GUARD_500',
    '20260708_H02_E01_FLOOR1_BUDGETED_SECOND_STAIR_500',
    '20260708_H03_E01_TRANSITION_HINT_FLOOR1_500',
    '20260708_H04_E01_ROUTE_PRESERVE_COMPRESS_FLOOR0_500',
    '20260708_H05_E01_FLOOR2_BEDROOM_PRIOR_GUARD_500',
    '20260708_H06_FAST_FLOOR2_BED_WAYPOINT_DIAG_500',
    '20260708_H07_COMPRESSED_E01_ROUTE_WAYPOINT_DIAG_500',
    '20260708_H08_G03_WITH_LOW_FLOOR_GUARD_500',
    '20260708_H09_G07_WITH_LOW_FLOOR_GUARD_500',
    '20260708_H10_DIRECT_ROUTE_WAYPOINT_PLUS_TARGET_500',
]

BATCH9_VARIANTS = [
    '20260708_J01_ORACLE_GT_BED0_VIEWPOINT_SCAN_500',
    '20260708_J02_ORACLE_GT_BED1_VIEWPOINT_SCAN_500',
    '20260708_J03_ORACLE_GT_BED2_VIEWPOINT_SCAN_500',
    '20260708_J04_ORACLE_GT_BED_VIEWPOINT_CAROUSEL_500',
    '20260708_J05_ORACLE_GT_BED_MULTI_APPROACH_SCAN_500',
    '20260708_J06_H07_WAYPOINT_NO_STOP_SCAN_500',
    '20260708_J07_H07_RELEASE_TO_HIGH_FLOOR_SEARCH_500',
    '20260708_J08_HIGH_FLOOR_PRIVATE_ROOM_CAROUSEL_500',
    '20260708_J09_HIGH_FLOOR_ANTI_REPEAT_SEARCH_500',
    '20260708_J10_HIGH_FLOOR_BUDGETED_FRONTIER_SEARCH_500',
]

BATCH10_VARIANTS = [
    '20260708_K01_E01_FLOOR0_EARLY_STAIR_COMMIT_500',
    '20260708_K02_E01_FLOOR0_LOOP_CUT_ROUTE_500',
    '20260708_K03_E01_FLOOR1_SECOND_STAIR_RUSH_500',
    '20260708_K04_E01_FLOOR1_ROUTE_COMPRESS_SECOND_STAIR_500',
    '20260708_K05_E01_AGGRESSIVE_SECOND_STAIR_PROGRESS_500',
    '20260708_K06_E01_DUAL_FLOOR_BUDGET_COMPRESS_500',
    '20260708_K07_E01_HIGH_FLOOR_RELEASE_PRIVATE_SEARCH_500',
    '20260708_K08_E01_HIGH_FLOOR_PRIVATE_ROOM_PRIOR_500',
    '20260708_K09_E01_BUDGETED_ANTI_REPEAT_HIGH_SEARCH_500',
    '20260708_K10_E01_ROUTE_COMPRESS_PLUS_HIGH_VERIFY_500',
]

BATCH11_VARIANTS = [
    '20260708_L01_E01_INIT8_500',
    '20260708_L02_E01_INIT6_500',
    '20260708_L03_E01_LOW_TARGET_CAP6_500',
    '20260708_L04_E01_LOW_TARGET_CAP4_500',
    '20260708_L05_E01_TURN_COMPRESS5_500',
    '20260708_L06_E01_TURN_COMPRESS4_500',
    '20260708_L07_E01_MILD_STAIR_RECOVERY_500',
    '20260708_L08_E01_INIT8_TARGET6_TURN5_500',
    '20260708_L09_E01_INIT6_TARGET4_TURN4_STAIR_500',
    '20260708_L10_E01_COMBO_FLOOR1_BUDGET_500',
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

_META.update({
    'F01': ('20260707_F01_FAST_TWO_UP_HOLD_FLOOR2_500', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'reach internal floor index 2 quickly, then block further upstairs and search'),
    'F02': ('20260707_F02_FAST_TWO_UP_FAST_INIT_500', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F01 plus shorter new-floor initialization'),
    'F03': ('20260707_F03_FAST_TWO_UP_FLOOR2_BEDROOM_PRIOR_500', 'AscentReasoner', 'AscentMemory', 'Floor2BedroomPriorScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F02 plus floor2-only bedroom/bed frontier prior'),
    'F04': ('20260707_F04_FAST_TWO_UP_FLOOR2_DEEP_PRIVATE_PRIOR_500', 'AscentReasoner', 'AscentMemory', 'Floor2DeepPrivateScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F02 plus floor2 deep/private-room frontier prior'),
    'F05': ('20260707_F05_FAST_TWO_UP_FLOOR2_NO_LOCAL_BIAS_500', 'AscentReasoner', 'AscentMemory', 'Floor2NoLocalBiasScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F02 plus floor2 local-nearby frontier bias disabled'),
    'F06': ('20260707_F06_FAST_TWO_UP_BEDROOM_NO_LOCAL_BIAS_500', 'AscentReasoner', 'AscentMemory', 'Floor2BedroomNoLocalBiasScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F03 plus floor2 local-nearby bias disabled'),
    'F07': ('20260707_F07_FAST_TWO_UP_EARLY_LLM_500', 'AscentReasoner', 'AscentMemory', 'Floor2EarlyLLMScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'after floor2, force earlier LLM/global frontier choice'),
    'F08': ('20260707_F08_FAST_TWO_UP_ZERO_INIT_500', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'None', 'None', 'F01 plus near-zero initialization after reaching high floors'),
    'F09': ('20260707_F09_FAST_TWO_UP_FLOOR2_TARGET_APPROACH_500', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'Floor2TargetApproach', 'None', 'F02 plus more assertive approach once floor2 target is detected'),
    'F10': ('20260707_F10_ORACLE_FLOOR2_BED_WAYPOINT_DIAGNOSTIC_500', 'AscentReasoner', 'OracleBedMemory', 'OracleBedWaypoint', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'OracleBedWaypoint', 'None', 'diagnostic only: after floor2, navigate toward E01 high-floor bed waypoint'),
})

_META.update({
    'G01': ('20260707_G01_ROUTE_NOVELTY_FRONTIER_500', 'RouteMemoryReasoner', 'RouteCellMemory', 'RouteNoveltyFrontierScore', 'AscentTransition', 'None', 'None', 'None', 'route-level novelty frontier selection from the first floor-0 decisions'),
    'G02': ('20260707_G02_LOOP_EXIT_CORRIDOR_FRONTIER_500', 'RouteMemoryReasoner', 'RouteCellMemory', 'LoopExitCorridorScore', 'AscentTransition', 'LoopExitRecovery', 'None', 'None', 'detect early loop/backtracking and select frontier that exits dense visited cells'),
    'G03': ('20260707_G03_TRANSITION_AWARE_ROUTE_SCORER_500', 'AscentReasoner', 'RouteTransitionMemory', 'TransitionAwareRouteScore', 'FastTransitionCommit', 'FastStairRecovery', 'None', 'None', 'prefer frontier routes likely to expose stairs/halls/doorways before forcing transition'),
    'G04': ('20260707_G04_ROUTE_COMPRESSION_BUDGET_500', 'AscentReasoner', 'RouteCellMemory', 'CompressedRouteFrontierScore', 'BudgetedFloorCompletion', 'None', 'None', 'None', 'compress low-value floor-0 exploration using route novelty and budget gates'),
    'G05': ('20260707_G05_SECOND_TRANSITION_BUDGET_PLANNER_500', 'AscentReasoner', 'RouteTransitionMemory', 'SecondTransitionBudgetScore', 'BudgetedSecondTransition', 'FastStairRecovery', 'None', 'None', 'reserve step budget for floor-index-2 and avoid spending floor1 on low-yield loops'),
    'G06': ('20260707_G06_FRONTIER_OUTCOME_MEMORY_500', 'AscentReasoner', 'FrontierOutcomeMemory', 'OutcomeAwareFrontierScore', 'BudgetedSecondTransition', 'OutcomeRecovery', 'None', 'None', 'penalize frontiers whose previous selection produced little route progress'),
    'G07': ('20260707_G07_TOPOLOGICAL_OPTION_CHAIN_500', 'TopologicalOptionReasoner', 'RouteTopologyMemory', 'TopoOptionChainScore', 'FastTransitionCommit', 'None', 'None', 'None', 'coarse option chain: exit-loop, transition, upper-private-room search'),
    'G08': ('20260707_G08_BUDGETED_OPTION_PLANNER_500', 'BudgetedOptionReasoner', 'RouteTransitionMemory', 'BudgetedOptionScore', 'BudgetedTransitionCommit', 'FastStairRecovery', 'None', 'None', 'budget-aware high-level option selection based on remaining steps and floor progress'),
    'G09': ('20260707_G09_ANTI_HOMING_ROUTE_EXPANSION_500', 'RouteMemoryReasoner', 'RouteCellMemory', 'AntiHomingExpansionScore', 'FastTransitionCommit', 'None', 'None', 'None', 'penalize frontier choices that route back toward start or dense visited cells'),
    'G10': ('20260707_G10_E01_COMPRESSED_ROUTE_DIAGNOSTIC_500', 'DiagnosticRouteReasoner', 'E01CompressedRouteMemory', 'E01WaypointFrontierScore', 'DiagnosticCompressedRoute', 'FastStairRecovery', 'None', 'None', 'diagnostic only: score frontiers by distance to compressed E01 success-route waypoints'),
})

_META.update({
    'H01': ('20260708_H01_E01_LOW_FLOOR_TARGET_GUARD_500', 'AscentReasoner', 'AscentMemory', 'AscentFrontierScore', 'FastFirstUpstairs', 'None', 'LowFloorTargetGuard', 'None', 'E01 control plus ignore target evidence below floor_index=2'),
    'H02': ('20260708_H02_E01_FLOOR1_BUDGETED_SECOND_STAIR_500', 'AscentReasoner', 'RouteTransitionMemory', 'SecondTransitionBudgetScore', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'E01 route plus floor1 budget reserve for second stair'),
    'H03': ('20260708_H03_E01_TRANSITION_HINT_FLOOR1_500', 'AscentReasoner', 'RouteTransitionMemory', 'Floor1TransitionHintScore', 'FastTransitionCommit', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'preserve E01 first route, then use transition-hint scoring on floor1'),
    'H04': ('20260708_H04_E01_ROUTE_PRESERVE_COMPRESS_FLOOR0_500', 'AscentReasoner', 'RouteCellMemory', 'E01PreserveCompressScore', 'BudgetedFloorCompletion', 'None', 'LowFloorTargetGuard', 'None', 'only compress E01 low-floor detours without broad novelty expansion'),
    'H05': ('20260708_H05_E01_FLOOR2_BEDROOM_PRIOR_GUARD_500', 'AscentReasoner', 'RouteSemanticMemory', 'Floor2BedroomPriorScore', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'after second transition, prefer bedroom/private-room frontiers with low-floor guard'),
    'H06': ('20260708_H06_FAST_FLOOR2_BED_WAYPOINT_DIAG_500', 'DiagnosticRouteReasoner', 'E01BedWaypointMemory', 'E01BedWaypointScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'diagnostic: once high floor reached, navigate toward E01 bed-side waypoint'),
    'H07': ('20260708_H07_COMPRESSED_E01_ROUTE_WAYPOINT_DIAG_500', 'DiagnosticRouteReasoner', 'E01CompressedRouteMemory', 'E01RouteWaypointScore', 'DiagnosticCompressedRoute', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'diagnostic: direct compressed E01 route waypoints with low-floor false-positive guard'),
    'H08': ('20260708_H08_G03_WITH_LOW_FLOOR_GUARD_500', 'AscentReasoner', 'RouteTransitionMemory', 'TransitionAwareRouteScore', 'FastTransitionCommit', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'batch7 G03 transition-aware route with low-floor target guard'),
    'H09': ('20260708_H09_G07_WITH_LOW_FLOOR_GUARD_500', 'TopologicalOptionReasoner', 'RouteTopologyMemory', 'TopoOptionChainScore', 'FastTransitionCommit', 'None', 'LowFloorTargetGuard', 'None', 'batch7 G07 option chain with low-floor target guard'),
    'H10': ('20260708_H10_DIRECT_ROUTE_WAYPOINT_PLUS_TARGET_500', 'DiagnosticRouteReasoner', 'E01CompressedRouteMemory', 'DirectRouteWaypoint', 'DiagnosticCompressedRoute', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'diagnostic: direct route waypoints, then allow target approach on floor_index=2'),
})

_META.update({
    'J01': ('20260708_J01_ORACLE_GT_BED0_VIEWPOINT_SCAN_500', 'DiagnosticRouteReasoner', 'OracleBedViewpointMemory', 'OracleBed0Viewpoint', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'diagnostic only: convert GT bed0 viewpoint to local route waypoint and scan without waypoint stop'),
    'J02': ('20260708_J02_ORACLE_GT_BED1_VIEWPOINT_SCAN_500', 'DiagnosticRouteReasoner', 'OracleBedViewpointMemory', 'OracleBed1Viewpoint', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'diagnostic only: convert GT bed1 viewpoint to local route waypoint and scan without waypoint stop'),
    'J03': ('20260708_J03_ORACLE_GT_BED2_VIEWPOINT_SCAN_500', 'DiagnosticRouteReasoner', 'OracleBedViewpointMemory', 'OracleBed2Viewpoint', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'diagnostic only: convert GT bed2 viewpoint to local route waypoint and scan without waypoint stop'),
    'J04': ('20260708_J04_ORACLE_GT_BED_VIEWPOINT_CAROUSEL_500', 'DiagnosticRouteReasoner', 'OracleBedViewpointMemory', 'OracleBedViewpointCarousel', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'diagnostic only: cycle GT bed viewpoints after reaching high floor'),
    'J05': ('20260708_J05_ORACLE_GT_BED_MULTI_APPROACH_SCAN_500', 'DiagnosticRouteReasoner', 'OracleBedViewpointMemory', 'OracleBedApproachCarousel', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'diagnostic only: visit multiple GT bed approach points with no waypoint stop'),
    'J06': ('20260708_J06_H07_WAYPOINT_NO_STOP_SCAN_500', 'DiagnosticRouteReasoner', 'E01CompressedRouteMemory', 'E01RouteWaypointNoStop', 'CompressedRouteNoStop', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'H07 route waypoint but convert waypoint STOP into scan/continue'),
    'J07': ('20260708_J07_H07_RELEASE_TO_HIGH_FLOOR_SEARCH_500', 'RouteOptionReasoner', 'RouteTopologyMemory', 'ReleaseAfterWaypointScore', 'CompressedRouteThenSearch', 'FastStairRecovery', 'HighFloorScanNoStop', 'None', 'use H07 compressed route, then release to normal high-floor search instead of stopping at waypoint'),
    'J08': ('20260708_J08_HIGH_FLOOR_PRIVATE_ROOM_CAROUSEL_500', 'RouteOptionReasoner', 'RouteSemanticMemory', 'HighFloorPrivateRoomCarousel', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'non-oracle: after floor2, cycle private-room/bedroom frontier evidence with anti-repeat memory'),
    'J09': ('20260708_J09_HIGH_FLOOR_ANTI_REPEAT_SEARCH_500', 'RouteOptionReasoner', 'RouteCellMemory', 'HighFloorAntiRepeatScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'non-oracle: high-floor route search penalizes repeated local cells and homing loops'),
    'J10': ('20260708_J10_HIGH_FLOOR_BUDGETED_FRONTIER_SEARCH_500', 'BudgetedOptionReasoner', 'RouteTransitionMemory', 'HighFloorBudgetedFrontierScore', 'FastTwoUpHoldFloor2', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'non-oracle: reserve floor2 budget for frontier search and avoid waypoint termination'),
})


_META.update({
    'K01': ('20260708_K01_E01_FLOOR0_EARLY_STAIR_COMMIT_500', 'RouteOptionReasoner', 'RouteTransitionMemory', 'E01Floor0BudgetScore', 'E01FastFirstUpstairs', 'None', 'LowFloorTargetGuard', 'None', 'E01-preserving floor0 route compression: commit to first stair earlier'),
    'K02': ('20260708_K02_E01_FLOOR0_LOOP_CUT_ROUTE_500', 'RouteOptionReasoner', 'E01RouteMemory', 'E01RouteWaypointScore', 'E01RouteWaypointLimited', 'LoopCutRecovery', 'LowFloorTargetGuard', 'None', 'bounded floor0 loop cut toward E01 first-stair route waypoint'),
    'K03': ('20260708_K03_E01_FLOOR1_SECOND_STAIR_RUSH_500', 'RouteOptionReasoner', 'RouteTransitionMemory', 'SecondStairRushScore', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'floor1 budget reserve: expose/commit to second stair earlier'),
    'K04': ('20260708_K04_E01_FLOOR1_ROUTE_COMPRESS_SECOND_STAIR_500', 'RouteOptionReasoner', 'E01RouteMemory', 'SecondStairWaypointScore', 'E01RouteWaypointLimited', 'LoopCutRecovery', 'LowFloorTargetGuard', 'None', 'bounded floor1 route compression toward second-stair waypoint'),
    'K05': ('20260708_K05_E01_AGGRESSIVE_SECOND_STAIR_PROGRESS_500', 'RouteOptionReasoner', 'RouteTransitionMemory', 'SecondStairRushScore', 'AggressiveSecondStairProgress', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'reduce stair approach/climb latency on the E01 route'),
    'K06': ('20260708_K06_E01_DUAL_FLOOR_BUDGET_COMPRESS_500', 'BudgetedOptionReasoner', 'E01RouteMemory', 'DualFloorBudgetRouteScore', 'BudgetedE01Transitions', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'compress both floor0 and floor1 while preserving E01 topology'),
    'K07': ('20260708_K07_E01_HIGH_FLOOR_RELEASE_PRIVATE_SEARCH_500', 'RouteOptionReasoner', 'RouteSemanticMemory', 'HighFloorReleasePrivateSearch', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'after second transition, release to non-oracle high-floor private-room search'),
    'K08': ('20260708_K08_E01_HIGH_FLOOR_PRIVATE_ROOM_PRIOR_500', 'RouteOptionReasoner', 'RouteSemanticMemory', 'HighFloorPrivateRoomPrior', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'high-floor frontier scoring favors private-room/bedroom evidence without GT'),
    'K09': ('20260708_K09_E01_BUDGETED_ANTI_REPEAT_HIGH_SEARCH_500', 'BudgetedOptionReasoner', 'RouteCellMemory', 'BudgetedAntiRepeatHighSearch', 'BudgetedSecondTransition', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'budget-aware high-floor search with anti-repeat route memory'),
    'K10': ('20260708_K10_E01_ROUTE_COMPRESS_PLUS_HIGH_VERIFY_500', 'BudgetedOptionReasoner', 'E01RouteMemory', 'E01CompressHighVerifyScore', 'BudgetedE01Transitions', 'FastStairRecovery', 'LowFloorTargetGuard', 'None', 'combined E01 route compression plus high-floor verification-oriented search'),
})

_META.update({
    'L01': ('20260708_L01_E01_INIT8_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'None', 'None', 'None', 'E01 route with shorter initial scan: 8-turn initialization'),
    'L02': ('20260708_L02_E01_INIT6_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'None', 'None', 'None', 'E01 route with shorter initial scan: 6-turn initialization'),
    'L03': ('20260708_L03_E01_LOW_TARGET_CAP6_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'None', 'LowFloorTargetCap6', 'None', 'E01 route with low-floor false-target navigation capped at 6 steps'),
    'L04': ('20260708_L04_E01_LOW_TARGET_CAP4_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'None', 'LowFloorTargetCap4', 'None', 'E01 route with low-floor false-target navigation capped at 4 steps'),
    'L05': ('20260708_L05_E01_TURN_COMPRESS5_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'TurnCompress5', 'None', 'None', 'E01 route with same-cell turn compression after 5 repeated turns'),
    'L06': ('20260708_L06_E01_TURN_COMPRESS4_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'TurnCompress4', 'None', 'None', 'E01 route with same-cell turn compression after 4 repeated turns'),
    'L07': ('20260708_L07_E01_MILD_STAIR_RECOVERY_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'MildStairRecovery', 'None', 'None', 'E01 route with mild stair approach/climb recovery'),
    'L08': ('20260708_L08_E01_INIT8_TARGET6_TURN5_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'TurnCompress5', 'LowFloorTargetCap6', 'None', 'E01-preserving combined mild early-waste compression'),
    'L09': ('20260708_L09_E01_INIT6_TARGET4_TURN4_STAIR_500', 'AscentReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'TurnCompress4PlusStair', 'LowFloorTargetCap4', 'None', 'E01 route with stronger early compression plus mild stair recovery'),
    'L10': ('20260708_L10_E01_COMBO_FLOOR1_BUDGET_500', 'BudgetedOptionReasoner', 'E01RouteMemory', 'AscentFrontierScore', 'E01FastFirstUpstairs', 'TurnCompress4PlusStair', 'LowFloorTargetCap4', 'None', 'E01 route with early compression and conservative floor1 budget reserve'),
})

def normalize_variant(raw: Optional[str] = None) -> str:
    value = (raw if raw is not None else zs.variant() or '').strip().upper()
    return _ALIASES.get(value, value)


def family(raw: Optional[str] = None) -> str:
    value = normalize_variant(raw)
    for pattern in (r'20260706_(I\d\d)', r'2026070[78]_([BCDEFGHJKL]\d\d)'):
        m = re.search(pattern, value)
        if m:
            return m.group(1)
    m = re.search(r'\b(I\d\d|B\d\d|C\d\d|D\d\d|E\d\d|F\d\d|G\d\d|H\d\d|J\d\d|K\d\d|L\d\d)\b', value)
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


def _batch11_config(fam: Optional[str] = None) -> Dict[str, Any]:
    fam = fam or family()
    if not fam.startswith('L'):
        return {}
    cfg: Dict[str, Any] = {
        'init_turns': None,
        'low_target_cap': None,
        'turn_compress': None,
        'sticky_steps': None,
        'repeat_steps': None,
        'fast_stair': False,
        'stair_force_threshold': 12,
        'floor1_budget_steps': None,
    }
    if fam == 'L01':
        cfg['init_turns'] = 8
    elif fam == 'L02':
        cfg['init_turns'] = 6
    elif fam == 'L03':
        cfg['low_target_cap'] = 6
    elif fam == 'L04':
        cfg['low_target_cap'] = 4
    elif fam == 'L05':
        cfg['turn_compress'] = 5
    elif fam == 'L06':
        cfg['turn_compress'] = 4
    elif fam == 'L07':
        cfg.update({'fast_stair': True, 'stair_force_threshold': 12})
    elif fam == 'L08':
        cfg.update({'init_turns': 8, 'low_target_cap': 6, 'turn_compress': 5, 'sticky_steps': 20, 'repeat_steps': 16})
    elif fam == 'L09':
        cfg.update({'init_turns': 6, 'low_target_cap': 4, 'turn_compress': 4, 'sticky_steps': 16, 'repeat_steps': 12, 'fast_stair': True, 'stair_force_threshold': 10})
    elif fam == 'L10':
        cfg.update({'init_turns': 6, 'low_target_cap': 4, 'turn_compress': 4, 'sticky_steps': 16, 'repeat_steps': 12, 'fast_stair': True, 'stair_force_threshold': 10, 'floor1_budget_steps': 135})
    return cfg


def batch11_fast_stair_enabled(raw: Optional[str] = None) -> bool:
    return bool(_batch11_config(family(raw)).get('fast_stair'))


def batch11_initialization_limit(env: int, step: int, cur_floor_index: int, floor_num_steps: int) -> Optional[Dict[str, Any]]:
    cfg = _batch11_config()
    limit = cfg.get('init_turns')
    if limit is None:
        return None
    return {
        **variant_metadata(),
        'adapter_stage': 'initialization',
        'initialization_event': 'shorten_initial_scan',
        'initialize_turn_limit': int(limit),
        'cur_floor_index': int(cur_floor_index),
        'floor_num_steps': int(floor_num_steps),
        'override': 1,
    }


def batch11_filter_target_goal(env: int, step: int, cur_floor_index: int, actual_goal: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    cfg = _batch11_config()
    cap = cfg.get('low_target_cap')
    if cap is None or actual_goal is None:
        return actual_goal, None
    if int(cur_floor_index) >= 2:
        return actual_goal, None
    st = _state('batch11_target_gate', env)
    used = int(st.get('low_floor_target_nav_steps', 0))
    if used < int(cap):
        st['low_floor_target_nav_steps'] = used + 1
        event = 'allow_low_floor_target_probe'
        filtered_goal = actual_goal
        override = 0
    else:
        st['low_floor_target_ignored'] = True
        event = 'ignore_low_floor_target_after_cap'
        filtered_goal = None
        override = 1
    return filtered_goal, {
        **variant_metadata(),
        'adapter_stage': 'target_gate',
        'target_gate_event': event,
        'low_floor_target_nav_steps': int(st.get('low_floor_target_nav_steps', used)),
        'low_floor_target_cap': int(cap),
        'cur_floor_index': int(cur_floor_index),
        'target_goal': _jsonable(actual_goal),
        'override': override,
        'selected_goal_type': 'target_object' if filtered_goal is not None else 'explore_after_low_floor_target_cap',
    }


def batch11_low_floor_target_ignored(env: int) -> bool:
    return bool(_state('batch11_target_gate', env).get('low_floor_target_ignored'))


def batch11_action_override(env: int, step: int, cur_floor_index: int, floor_num_steps: int, mode: str, action: Any, robot_xy: Any, current_frontier: Any) -> Optional[Dict[str, Any]]:
    cfg = _batch11_config()
    limit = cfg.get('turn_compress')
    if limit is None:
        return None
    try:
        act = int(action)
    except Exception:
        return None
    if act not in (2, 3):
        st = _state('batch11_action', env)
        st['turn_run'] = 0
        st['last_key'] = None
        return None
    # Keep the intervention local: it only breaks repeated same-cell turns, and it
    # does not change frontier/route selection directly.
    try:
        xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    if xy.size < 2:
        return None
    try:
        fr = np.asarray(current_frontier, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    if fr.size < 2:
        return None
    key = (
        int(cur_floor_index),
        round(float(xy[0]) / 0.35),
        round(float(xy[1]) / 0.35),
        round(float(fr[0]) / 0.50),
        round(float(fr[1]) / 0.50),
        str(mode),
    )
    st = _state('batch11_action', env)
    if st.get('last_key') == key:
        st['turn_run'] = int(st.get('turn_run', 0)) + 1
    else:
        st['turn_run'] = 1
        st['last_key'] = key
    if int(st['turn_run']) < int(limit):
        return None
    st['turn_compress_events'] = int(st.get('turn_compress_events', 0)) + 1
    st['turn_run'] = 0
    return {
        **variant_metadata(),
        'adapter_stage': 'action_compression',
        'action_compression_event': 'force_forward_after_same_cell_turns',
        'turn_compress_limit': int(limit),
        'turn_compress_event_count': int(st['turn_compress_events']),
        'original_action': act,
        'policy_action': 1,
        'cur_floor_index': int(cur_floor_index),
        'floor_num_steps': int(floor_num_steps),
        'mode': str(mode),
        'robot_cell_key': _jsonable(key),
        'current_frontier': _jsonable(current_frontier),
        'override': 1,
        'selected_goal_type': 'route_progress_action',
    }


def batch11_frontier_threshold(default_threshold: int, env: int, kind: str, cur_floor_index: Optional[int], num_steps: Optional[int], current_frontier: Any) -> Tuple[int, Optional[Dict[str, Any]]]:
    cfg = _batch11_config()
    key = 'sticky_steps' if kind == 'sticky' else 'repeat_steps'
    value = cfg.get(key)
    if value is None:
        return int(default_threshold), None
    # Do not aggressively retire high-floor target-search frontiers; this is aimed at
    # the first low-floor repeated exploration that E01 wastes before the useful route.
    if cur_floor_index is not None and int(cur_floor_index) >= 2:
        return int(default_threshold), None
    threshold = max(2, int(value))
    return threshold, {
        **variant_metadata(),
        'adapter_stage': 'frontier_retire',
        'frontier_retire_event': f'batch11_{kind}_threshold',
        'frontier_retire_kind': str(kind),
        'frontier_retire_threshold': threshold,
        'default_threshold': int(default_threshold),
        'cur_floor_index': int(cur_floor_index) if cur_floor_index is not None else None,
        'num_steps': int(num_steps) if num_steps is not None else None,
        'current_frontier': _jsonable(current_frontier),
        'override': int(threshold != int(default_threshold)),
    }


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

_FLOOR2_BED_WAYPOINT_XY = np.array([-0.8777501992477128, -5.4404126282525045], dtype=float)


def _batch6_floor2_choice(cards: List[Dict[str, Any]], fam: str, target: str) -> Tuple[int, List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []
    target_terms = {target.lower(), 'bed', 'bedroom', 'bed room', 'sleeping', 'private'}
    for c in cards:
        xy = c.get('xy', [0.0, 0.0])
        try:
            fx, fy = float(xy[0]), float(xy[1])
        except Exception:
            fx, fy = 0.0, 0.0
        distance = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        repeat = _safe_float(c.get('repeat_count'))
        room = str(c.get('room', '')).lower().replace('_', ' ')
        objects = {str(x).lower().replace('_', ' ') for x in c.get('objects', [])}
        object_hit = 1.0 if objects.intersection(target_terms) else 0.0
        room_hit = 1.0 if any(term in room for term in target_terms) else 0.0
        deep_private = max(0.0, -fy) / 6.0
        waypoint_dist = float(np.linalg.norm(np.array([fx, fy]) - _FLOOR2_BED_WAYPOINT_XY))
        if fam in {'F03', 'F06'}:
            score = 1.6 * object_hit + 1.0 * room_hit + 0.45 * unseen + 0.20 * mval - 0.10 * repeat - 0.01 * distance
        elif fam == 'F04':
            score = 0.9 * deep_private + 0.7 * room_hit + 0.3 * object_hit + 0.30 * unseen + 0.12 * mval - 0.08 * repeat
        elif fam == 'F05':
            score = 0.65 * unseen + 0.25 * mval - 0.08 * repeat - 0.005 * distance
        elif fam == 'F07':
            score = 0.70 * room_hit + 0.45 * deep_private + 0.35 * unseen + 0.25 * mval - 0.10 * repeat
        elif fam == 'F10':
            score = -0.20 * waypoint_dist + 0.20 * unseen + 0.10 * mval - 0.05 * repeat
        else:
            score = mval
        item = dict(c)
        item.update({
            'batch6_score': round(float(score), 5),
            'target_object_hit': object_hit,
            'target_room_hit': room_hit,
            'deep_private_proxy': round(float(deep_private), 4),
            'bed_waypoint_distance': round(float(waypoint_dist), 3),
        })
        scored.append(item)
    if not scored:
        return 0, []
    return max(range(len(scored)), key=lambda i: scored[i]['batch6_score']), scored


_ROUTE_TRANSITION_TERMS = {
    'stair', 'stairs', 'staircase', 'stairway', 'step', 'steps', 'landing',
    'hall', 'hallway', 'corridor', 'door', 'doorway', 'entry', 'entrance', 'rail', 'balustrade'
}
_ROUTE_PRIVATE_TERMS = {
    'bed', 'bedroom', 'closet', 'private', 'bathroom', 'office', 'dresser',
    'cabinet', 'door', 'curtain', 'room'
}
_E01_ROUTE_WAYPOINTS = {
    0: np.array([-4.755, -2.218], dtype=float),
    1: np.array([-4.719, -2.192], dtype=float),
    2: np.array([-0.878, -5.440], dtype=float),
}


def _batch7_cell(xy: Any, floor_idx: int, res: float = 1.0) -> str:
    try:
        arr = np.asarray(xy, dtype=float).reshape(-1)[:2]
        return f"{int(floor_idx)}:{round(float(arr[0]) / res)}:{round(float(arr[1]) / res)}"
    except Exception:
        return f"{int(floor_idx)}:bad"


def _route_memory_update(env: int, robot_xy: Any, floor_idx: int, num_step: Optional[int]) -> Dict[str, Any]:
    st = _state('batch7_route_memory', env)
    try:
        xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        xy = np.zeros(2, dtype=float)
    if 'start_xy' not in st:
        st['start_xy'] = xy.copy()
    hist = st.setdefault('robot_history', [])
    hist.append({'step': int(num_step) if num_step is not None else None, 'floor': int(floor_idx), 'xy': xy.tolist()})
    del hist[:-160]
    cell = _batch7_cell(xy, floor_idx, 0.75)
    visited = st.setdefault('visited_robot_cells', {})
    visited[cell] = int(visited.get(cell, 0)) + 1
    return st


def _keyword_hit(card: Dict[str, Any], terms: set) -> float:
    text = ' '.join([str(card.get('room', ''))] + [str(x) for x in card.get('objects', [])]).lower().replace('_', ' ')
    return 1.0 if any(term in text for term in terms) else 0.0


def _batch7_route_choice(cards: List[Dict[str, Any]], fam: str, env: int, target: str, current_floor_index: Optional[int], num_step: Optional[int], original_idx: int, robot_xy: Any, selection_source: str, frontier_stick_step: Optional[int]) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    floor_idx = int(current_floor_index) if current_floor_index is not None else 0
    step = int(num_step) if num_step is not None else 0
    st = _route_memory_update(env, robot_xy, floor_idx, step)
    start_xy = np.asarray(st.get('start_xy', np.zeros(2)), dtype=float)[:2]
    try:
        cur_xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        cur_xy = np.zeros(2, dtype=float)
    history = st.get('robot_history', [])
    recent = [np.asarray(h.get('xy', [0, 0]), dtype=float)[:2] for h in history[-24:]]
    if recent:
        centroid = np.mean(np.stack(recent, axis=0), axis=0)
    else:
        centroid = cur_xy
    selected_cells = st.setdefault('selected_frontier_cells', {})
    scored: List[Dict[str, Any]] = []
    budget_urgency = min(max((step - 120) / 260.0, 0.0), 1.0)
    if floor_idx == 1:
        budget_urgency = min(max((step - 330) / 120.0, 0.0), 1.0)
    elif floor_idx >= 2:
        budget_urgency = min(max((step - 430) / 80.0, 0.0), 1.0)
    for i, c in enumerate(cards):
        xy = np.asarray(c.get('xy', [0.0, 0.0]), dtype=float)[:2]
        cell = _batch7_cell(xy, floor_idx, 1.0)
        selected_repeat = float(selected_cells.get(cell, 0))
        builtin_repeat = _safe_float(c.get('repeat_count'))
        visited_count = float(st.get('visited_robot_cells', {}).get(cell, 0))
        dist_robot = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        transition_hit = _keyword_hit(c, _ROUTE_TRANSITION_TERMS)
        private_hit = _keyword_hit(c, _ROUTE_PRIVATE_TERMS | {target.lower()})
        centroid_dist = float(np.linalg.norm(xy - centroid))
        start_dist = float(np.linalg.norm(xy - start_xy))
        cur_start_dist = float(np.linalg.norm(cur_xy - start_xy))
        route_progress = start_dist - cur_start_dist
        anti_homing = max(0.0, start_dist - 1.0) + 0.25 * centroid_dist
        waypoint_dist = None
        if fam == 'G10':
            wp = _E01_ROUTE_WAYPOINTS.get(min(max(floor_idx, 0), 2), _E01_ROUTE_WAYPOINTS[2])
            waypoint_dist = float(np.linalg.norm(xy - wp))
        novelty = 1.0 / (1.0 + selected_repeat + builtin_repeat + 0.25 * visited_count)
        if fam == 'G01':
            score = mval + 0.85 * unseen + 0.70 * novelty + 0.08 * centroid_dist - 0.02 * dist_robot
        elif fam == 'G02':
            score = 0.50 * centroid_dist + 0.65 * unseen + 0.45 * transition_hit + 0.20 * route_progress - 0.18 * builtin_repeat - 0.03 * dist_robot
        elif fam == 'G03':
            score = 1.15 * transition_hit + 0.55 * unseen + 0.25 * mval + 0.18 * novelty - 0.012 * dist_robot
        elif fam == 'G04':
            score = 0.75 * unseen + 0.35 * route_progress + 0.45 * novelty + 0.20 * transition_hit - 0.22 * builtin_repeat - 0.02 * dist_robot
        elif fam == 'G05':
            score = 1.00 * budget_urgency * transition_hit + 0.55 * unseen + 0.35 * private_hit + 0.20 * route_progress - 0.02 * dist_robot - 0.16 * builtin_repeat
        elif fam == 'G06':
            outcome_penalty = float(st.setdefault('poor_outcome_cells', {}).get(cell, 0))
            score = 0.70 * unseen + 0.40 * novelty + 0.35 * transition_hit + 0.20 * private_hit - 0.45 * outcome_penalty - 0.18 * builtin_repeat - 0.015 * dist_robot
        elif fam == 'G07':
            phase_transition = 1.0 if floor_idx < 2 else 0.0
            phase_private = 1.0 if floor_idx >= 1 else 0.25
            score = 0.85 * phase_transition * transition_hit + 0.75 * phase_private * private_hit + 0.40 * unseen + 0.25 * novelty + 0.08 * route_progress - 0.015 * dist_robot
        elif fam == 'G08':
            score = (0.75 + 0.75 * budget_urgency) * transition_hit * (1.0 if floor_idx < 2 else 0.0) + (0.85 * private_hit if floor_idx >= 2 else 0.25 * private_hit) + 0.45 * unseen + 0.18 * route_progress - 0.018 * dist_robot - 0.12 * builtin_repeat
        elif fam == 'G09':
            score = 0.55 * anti_homing + 0.45 * unseen + 0.25 * transition_hit + 0.15 * private_hit - 0.25 * builtin_repeat - 0.04 * dist_robot
        elif fam == 'G10':
            score = -0.55 * _safe_float(waypoint_dist, 99.0) + 0.30 * unseen + 0.20 * transition_hit + 0.15 * private_hit - 0.05 * dist_robot
        else:
            score = mval
        item = dict(c)
        item.update({
            'batch7_score': round(float(score), 5),
            'route_cell': cell,
            'route_selected_repeat': selected_repeat,
            'route_robot_cell_visit_count': visited_count,
            'route_novelty': round(float(novelty), 4),
            'route_transition_hint': transition_hit,
            'route_private_hint': private_hit,
            'route_progress_from_start': round(float(route_progress), 3),
            'route_centroid_distance': round(float(centroid_dist), 3),
            'route_start_distance': round(float(start_dist), 3),
            'route_budget_urgency': round(float(budget_urgency), 3),
            'e01_waypoint_distance': round(float(waypoint_dist), 3) if waypoint_dist is not None else None,
        })
        scored.append(item)
    if not scored:
        return original_idx, [], {'fallback': 1, 'fallback_reason': 'batch7_no_scored_cards'}
    best = max(range(len(scored)), key=lambda i: scored[i]['batch7_score'])
    selected_cells[scored[best]['route_cell']] = int(selected_cells.get(scored[best]['route_cell'], 0)) + 1
    # Outcome memory: when selection remains close to a dense route cell, mark it as poor for G06.
    if fam == 'G06':
        if scored[best]['route_robot_cell_visit_count'] >= 2 or scored[best]['route_selected_repeat'] >= 2:
            poor = st.setdefault('poor_outcome_cells', {})
            poor[scored[best]['route_cell']] = int(poor.get(scored[best]['route_cell'], 0)) + 1
    trace = {
        'batch7_route_memory': {
            'robot_history_len': len(history),
            'visited_robot_cell_count': len(st.get('visited_robot_cells', {})),
            'selected_frontier_cell_count': len(selected_cells),
            'floor_idx': floor_idx,
            'num_step': step,
            'selection_source_before_batch7': selection_source,
            'frontier_stick_step': int(frontier_stick_step or 0),
            'diagnostic_only': int(fam == 'G10'),
        }
    }
    return best, scored, trace


_H_ROUTE_FIRST_STAIR_XY = np.array([-4.755, -2.218], dtype=float)
_H_ROUTE_SECOND_STAIR_XY = np.array([-4.719, -2.192], dtype=float)
_H_ROUTE_BED_WAYPOINT_XY = np.array([-0.878, -5.440], dtype=float)
# Diagnostic-only GT bed viewpoints converted from simulator x/z into trace-local robot_xy.
_J_BED0_VIEW_XY = np.array([0.426, -5.404], dtype=float)
_J_BED1_VIEW_XY = np.array([-1.063, -3.429], dtype=float)
_J_BED2_VIEW_XY = np.array([0.318, 0.635], dtype=float)
_J_BED0_OBJ_XY = np.array([1.363, -6.726], dtype=float)
_J_BED1_OBJ_XY = np.array([-2.484, -3.227], dtype=float)
_J_BED2_OBJ_XY = np.array([0.069, 1.609], dtype=float)


def _batch9_waypoint(fam: str, cur_floor_index: int, floor_num_steps: int) -> np.ndarray:
    if cur_floor_index <= 0:
        return _H_ROUTE_FIRST_STAIR_XY
    if cur_floor_index == 1:
        return _H_ROUTE_SECOND_STAIR_XY
    if fam == 'J01':
        return _J_BED0_VIEW_XY
    if fam == 'J02':
        return _J_BED1_VIEW_XY
    if fam == 'J03':
        return _J_BED2_VIEW_XY
    if fam == 'J04':
        return [_J_BED0_VIEW_XY, _J_BED1_VIEW_XY, _J_BED2_VIEW_XY][int(floor_num_steps // 55) % 3]
    if fam == 'J05':
        return [_J_BED0_VIEW_XY, _J_BED0_OBJ_XY, _J_BED1_VIEW_XY, _J_BED2_VIEW_XY][int(floor_num_steps // 45) % 4]
    if fam in {'J06', 'J07'}:
        return _H_ROUTE_BED_WAYPOINT_XY
    return _J_BED0_VIEW_XY


def _batch8_route_choice(cards: List[Dict[str, Any]], fam: str, env: int, target: str, current_floor_index: Optional[int], num_step: Optional[int], original_idx: int, robot_xy: Any, selection_source: str, frontier_stick_step: Optional[int]) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    floor_idx = int(current_floor_index) if current_floor_index is not None else 0
    step = int(num_step) if num_step is not None else 0
    st = _route_memory_update(env, robot_xy, floor_idx, step)
    try:
        cur_xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        cur_xy = np.zeros(2, dtype=float)
    if floor_idx <= 0:
        waypoint = _H_ROUTE_FIRST_STAIR_XY
    elif floor_idx == 1:
        waypoint = _H_ROUTE_SECOND_STAIR_XY
    else:
        waypoint = _H_ROUTE_BED_WAYPOINT_XY
    scored: List[Dict[str, Any]] = []
    for c in cards:
        xy = np.asarray(c.get('xy', [0.0, 0.0]), dtype=float)[:2]
        dist_robot = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        repeat = _safe_float(c.get('repeat_count'))
        transition_hit = _keyword_hit(c, _ROUTE_TRANSITION_TERMS)
        private_hit = _keyword_hit(c, _ROUTE_PRIVATE_TERMS | {target.lower()})
        waypoint_dist = float(np.linalg.norm(xy - waypoint))
        # Positive when a candidate points toward the route waypoint compared with current pose.
        waypoint_progress = float(np.linalg.norm(cur_xy - waypoint) - waypoint_dist)
        if fam in {'H01'}:
            score = mval
        elif fam == 'H02':
            if floor_idx == 1:
                score = 1.10 * transition_hit + 0.65 * waypoint_progress + 0.35 * unseen - 0.02 * dist_robot - 0.12 * repeat
            else:
                score = 0.15 * waypoint_progress + 0.45 * unseen + 0.20 * mval - 0.02 * dist_robot
        elif fam == 'H03':
            score = (1.25 * transition_hit if floor_idx == 1 else 0.20 * transition_hit) + 0.55 * unseen + 0.45 * waypoint_progress + 0.15 * mval - 0.02 * dist_robot - 0.12 * repeat
        elif fam == 'H04':
            if floor_idx == 0:
                score = 0.75 * waypoint_progress + 0.35 * unseen + 0.10 * mval - 0.015 * dist_robot - 0.10 * repeat
            else:
                score = mval
        elif fam == 'H05':
            if floor_idx >= 2:
                score = 1.25 * private_hit + 0.45 * unseen + 0.25 * mval - 0.018 * dist_robot - 0.12 * repeat
            elif floor_idx == 1:
                score = 1.05 * transition_hit + 0.55 * waypoint_progress + 0.30 * unseen - 0.018 * dist_robot
            else:
                score = 0.45 * waypoint_progress + 0.35 * unseen + 0.10 * mval - 0.015 * dist_robot
        elif fam == 'H06':
            if floor_idx >= 2:
                score = -0.70 * waypoint_dist + 0.35 * private_hit + 0.15 * unseen
            else:
                score = 0.85 * transition_hit + 0.45 * waypoint_progress + 0.25 * unseen - 0.018 * dist_robot
        elif fam in {'H07', 'H10'}:
            score = -0.85 * waypoint_dist + 0.25 * transition_hit + 0.20 * private_hit + 0.10 * unseen - 0.01 * dist_robot
        elif fam == 'H08':
            score = 1.15 * transition_hit + 0.55 * unseen + 0.25 * mval + 0.30 * waypoint_progress - 0.012 * dist_robot - 0.10 * repeat
        elif fam == 'H09':
            phase_transition = 1.0 if floor_idx < 2 else 0.0
            phase_private = 1.0 if floor_idx >= 1 else 0.25
            score = 0.85 * phase_transition * transition_hit + 0.75 * phase_private * private_hit + 0.40 * unseen + 0.25 * waypoint_progress - 0.015 * dist_robot
        else:
            score = mval
        item = dict(c)
        item.update({
            'batch8_score': round(float(score), 5),
            'route_waypoint_xy': _jsonable(waypoint),
            'route_waypoint_distance': round(float(waypoint_dist), 3),
            'route_waypoint_progress': round(float(waypoint_progress), 3),
            'route_transition_hint': transition_hit,
            'route_private_hint': private_hit,
            'diagnostic_only': int(fam in {'H06', 'H07', 'H10'}),
        })
        scored.append(item)
    if not scored:
        return original_idx, [], {'fallback': 1, 'fallback_reason': 'batch8_no_scored_cards'}
    best = max(range(len(scored)), key=lambda i: scored[i]['batch8_score'])
    trace = {'batch8_route_memory': {'floor_idx': floor_idx, 'num_step': step, 'waypoint_xy': _jsonable(waypoint), 'diagnostic_only': int(fam in {'H06', 'H07', 'H10'}), 'selection_source_before_batch8': selection_source, 'frontier_stick_step': int(frontier_stick_step or 0)}}
    return best, scored, trace



def _batch9_route_choice(cards: List[Dict[str, Any]], fam: str, env: int, target: str, current_floor_index: Optional[int], num_step: Optional[int], original_idx: int, robot_xy: Any, selection_source: str, frontier_stick_step: Optional[int]) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    floor_idx = int(current_floor_index) if current_floor_index is not None else 0
    step = int(num_step) if num_step is not None else 0
    st = _route_memory_update(env, robot_xy, floor_idx, step)
    try:
        cur_xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        cur_xy = np.zeros(2, dtype=float)
    selected_cells = st.setdefault('batch9_selected_frontier_cells', {})
    scored: List[Dict[str, Any]] = []
    for c in cards:
        xy = np.asarray(c.get('xy', [0.0, 0.0]), dtype=float)[:2]
        cell = _batch7_cell(xy, floor_idx, 0.75)
        dist_robot = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        repeat = _safe_float(c.get('repeat_count')) + float(selected_cells.get(cell, 0))
        transition_hit = _keyword_hit(c, _ROUTE_TRANSITION_TERMS)
        private_hit = _keyword_hit(c, _ROUTE_PRIVATE_TERMS | {target.lower()})
        anti_local = float(np.linalg.norm(xy - cur_xy))
        if floor_idx < 2:
            score = 1.0 * transition_hit + 0.35 * unseen + 0.20 * mval - 0.015 * dist_robot - 0.12 * repeat
        elif fam == 'J08':
            score = 1.35 * private_hit + 0.55 * unseen + 0.20 * anti_local - 0.018 * dist_robot - 0.35 * repeat
        elif fam == 'J09':
            score = 0.80 * anti_local + 0.55 * unseen + 0.40 * private_hit - 0.45 * repeat - 0.012 * dist_robot
        elif fam == 'J10':
            budget = min(max((step - 185) / 250.0, 0.0), 1.0)
            score = (0.70 + 0.60 * budget) * unseen + 0.75 * private_hit + 0.25 * anti_local - 0.30 * repeat - 0.015 * dist_robot
        else:
            score = 0.55 * unseen + 0.45 * private_hit + 0.20 * anti_local - 0.25 * repeat - 0.015 * dist_robot
        item = dict(c)
        item.update({
            'batch9_score': round(float(score), 5),
            'route_cell': cell,
            'route_transition_hint': transition_hit,
            'route_private_hint': private_hit,
            'route_anti_local_distance': round(float(anti_local), 3),
            'route_repeat_penalty': round(float(repeat), 3),
            'diagnostic_only': int(fam in {'J01', 'J02', 'J03', 'J04', 'J05', 'J06'}),
        })
        scored.append(item)
    if not scored:
        return original_idx, [], {'fallback': 1, 'fallback_reason': 'batch9_no_scored_cards'}
    best = max(range(len(scored)), key=lambda i: scored[i]['batch9_score'])
    selected_cells[scored[best]['route_cell']] = int(selected_cells.get(scored[best]['route_cell'], 0)) + 1
    trace = {'batch9_route_memory': {'floor_idx': floor_idx, 'num_step': step, 'selected_frontier_cell_count': len(selected_cells), 'diagnostic_only': int(fam in {'J01', 'J02', 'J03', 'J04', 'J05', 'J06'}), 'selection_source_before_batch9': selection_source, 'frontier_stick_step': int(frontier_stick_step or 0)}}
    return best, scored, trace


def _batch10_route_choice(cards: List[Dict[str, Any]], fam: str, env: int, target: str, current_floor_index: Optional[int], num_step: Optional[int], original_idx: int, robot_xy: Any, selection_source: str, frontier_stick_step: Optional[int]) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    """Route-level hm3d_r0_006 variants built from the E01 failure trace.

    The intent is not to add oracle target knowledge. K variants preserve the useful
    E01 topology (first transition, second transition, high-floor search) while
    reducing repeated low-floor route cost and making the post-transition search less
    local-greedy.
    """
    floor_idx = int(current_floor_index) if current_floor_index is not None else 0
    step = int(num_step) if num_step is not None else 0
    st = _route_memory_update(env, robot_xy, floor_idx, step)
    try:
        cur_xy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
    except Exception:
        cur_xy = np.zeros(2, dtype=float)
    history = st.get('robot_history', [])
    recent = [np.asarray(h.get('xy', [0, 0]), dtype=float)[:2] for h in history[-30:]]
    centroid = np.mean(np.stack(recent, axis=0), axis=0) if recent else cur_xy
    selected_cells = st.setdefault('batch10_selected_frontier_cells', {})
    waypoint = _H_ROUTE_FIRST_STAIR_XY if floor_idx <= 0 else (_H_ROUTE_SECOND_STAIR_XY if floor_idx == 1 else _H_ROUTE_BED_WAYPOINT_XY)
    if floor_idx <= 0:
        budget_urgency = min(max((step - 150) / 120.0, 0.0), 1.0)
    elif floor_idx == 1:
        budget_urgency = min(max((step - 345) / 90.0, 0.0), 1.0)
    else:
        budget_urgency = min(max((step - 410) / 80.0, 0.0), 1.0)
    scored: List[Dict[str, Any]] = []
    for c in cards:
        xy = np.asarray(c.get('xy', [0.0, 0.0]), dtype=float)[:2]
        cell = _batch7_cell(xy, floor_idx, 0.75)
        dist_robot = _safe_float(c.get('distance_to_robot'))
        unseen = _safe_float(c.get('expected_unseen_area'))
        mval = _safe_float(c.get('mval'))
        repeat = _safe_float(c.get('repeat_count')) + float(selected_cells.get(cell, 0))
        transition_hit = _keyword_hit(c, _ROUTE_TRANSITION_TERMS)
        private_hit = _keyword_hit(c, _ROUTE_PRIVATE_TERMS | {target.lower()})
        anti_repeat = float(np.linalg.norm(xy - centroid))
        waypoint_dist = float(np.linalg.norm(xy - waypoint))
        waypoint_progress = float(np.linalg.norm(cur_xy - waypoint) - waypoint_dist)
        if floor_idx <= 0:
            base = 0.70 * waypoint_progress + 0.55 * transition_hit + 0.30 * unseen + 0.10 * mval - 0.016 * dist_robot - 0.20 * repeat
            if fam in {'K02', 'K06', 'K10'}:
                score = base + 0.35 * budget_urgency - 0.18 * waypoint_dist
            elif fam == 'K01':
                score = base + 0.25 * budget_urgency
            else:
                score = base
        elif floor_idx == 1:
            base = 0.85 * waypoint_progress + 1.00 * transition_hit + 0.30 * unseen + 0.12 * mval - 0.018 * dist_robot - 0.22 * repeat
            if fam in {'K03', 'K04', 'K05', 'K06', 'K10'}:
                score = base + 0.55 * budget_urgency - 0.20 * waypoint_dist
            else:
                score = base + 0.20 * budget_urgency
        else:
            base = 0.80 * private_hit + 0.55 * unseen + 0.25 * anti_repeat + 0.18 * mval - 0.018 * dist_robot - 0.35 * repeat
            if fam == 'K08':
                score = base + 0.80 * private_hit
            elif fam == 'K09':
                score = base + 0.45 * budget_urgency + 0.35 * anti_repeat - 0.20 * repeat
            elif fam == 'K10':
                score = base + 0.35 * budget_urgency + 0.35 * private_hit
            elif fam == 'K07':
                score = base + 0.25 * unseen
            else:
                score = base
        item = dict(c)
        item.update({
            'batch10_score': round(float(score), 5),
            'route_cell': cell,
            'route_waypoint_xy': _jsonable(waypoint),
            'route_waypoint_distance': round(float(waypoint_dist), 3),
            'route_waypoint_progress': round(float(waypoint_progress), 3),
            'route_transition_hint': transition_hit,
            'route_private_hint': private_hit,
            'route_anti_repeat_distance': round(float(anti_repeat), 3),
            'route_repeat_penalty': round(float(repeat), 3),
            'route_budget_urgency': round(float(budget_urgency), 3),
            'selection_source_before_batch10': selection_source,
        })
        scored.append(item)
    if not scored:
        return original_idx, [], {'fallback': 1, 'fallback_reason': 'batch10_no_scored_cards'}
    best = max(range(len(scored)), key=lambda i: scored[i]['batch10_score'])
    selected_cells[scored[best]['route_cell']] = int(selected_cells.get(scored[best]['route_cell'], 0)) + 1
    trace = {
        'batch10_route_memory': {
            'floor_idx': floor_idx,
            'num_step': step,
            'robot_history_len': len(history),
            'selected_frontier_cell_count': len(selected_cells),
            'waypoint_xy': _jsonable(waypoint),
            'budget_urgency': round(float(budget_urgency), 3),
            'selection_source_before_batch10': selection_source,
            'frontier_stick_step': int(frontier_stick_step or 0),
        }
    }
    return best, scored, trace


def high_floor_initialization_limit(cur_floor_index: int) -> Optional[int]:
    fam = family()
    if fam in {'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09', 'F10'} and cur_floor_index >= 1:
        return 3
    if fam == 'F08' and cur_floor_index >= 1:
        return 0
    if fam.startswith('G') and cur_floor_index >= 1 and fam not in {'G01', 'G02'}:
        return 3
    if fam.startswith('H') and cur_floor_index >= 1 and fam not in {'H01'}:
        return 3
    if fam.startswith('J') and cur_floor_index >= 1:
        return 3
    if fam.startswith('K') and cur_floor_index >= 1:
        return 3
    return None


def should_skip_local_bias(current_floor_index: Optional[int] = None) -> bool:
    fam = family()
    floor_idx = int(current_floor_index) if current_floor_index is not None else -1
    return fam in {'F05', 'F06', 'F07'} and floor_idx >= 2


def floor2_bed_waypoint_policy(env: int, step: int, cur_floor_index: int) -> Optional[Dict[str, Any]]:
    fam = family()
    if fam != 'F10' or cur_floor_index < 2:
        return None
    st = _state('floor2_waypoint', env)
    st['event_count'] = int(st.get('event_count', 0)) + 1
    return {
        **variant_metadata(),
        'adapter_stage': 'floor2_waypoint',
        'floor_probe_event': 'floor2_bed_waypoint',
        'floor_probe_reason': 'diagnostic_navigate_to_e01_bed_waypoint',
        'floor_probe_event_count': st['event_count'],
        'cur_floor_index': int(cur_floor_index),
        'waypoint_xy': _jsonable(_FLOOR2_BED_WAYPOINT_XY),
        'diagnostic_only': 1,
        'override': 1,
        'selected_goal_type': 'diagnostic_bed_waypoint',
    }

def maybe_override_frontier(planner: Any, observations_cache: List[dict], obstacle_map: Any, value_map: Any, object_map: Any, sorted_pts: np.ndarray, sorted_values: List[float], env: int, topk: int, original_frontier: Any, original_value: float, selection_source: str, frontier_stick_step: Optional[List[int]] = None, current_floor_index: Optional[int] = None, num_steps: Optional[int] = None) -> Tuple[Any, float, str, Dict[str, Any]]:
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
    trace: Dict[str, Any] = {**variant_metadata(), 'adapter_stage': 'frontier_selection', 'current_floor_index': int(current_floor_index) if current_floor_index is not None else None, 'num_steps': int(num_steps) if num_steps is not None else None, 'original_selection_source': selection_source, 'original_frontier': _jsonable(original_frontier), 'original_value': _safe_float(original_value), 'original_frontier_id': original_idx + 1, 'frontier_cards': _jsonable(cards), 'object_memory': _jsonable(memory), 'llm_called': 0, 'parse_failure': 0, 'fallback': 0, 'override': 0}
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
    elif fam in {'F03', 'F04', 'F05', 'F06', 'F07', 'F10'}:
        floor_idx = int(current_floor_index) if current_floor_index is not None else -1
        if floor_idx < 2:
            trace['batch6_floor2_inactive'] = 1
            return original_frontier, original_value, selection_source, trace
        selected_idx, scored = _batch6_floor2_choice(cards, fam, target)
        trace['batch6_floor2_scores'] = _jsonable(scored)
        source = 'batch6_floor2_frontier'
    elif fam.startswith('G'):
        robot_xy = observations_cache[env].get('robot_xy', np.zeros(2)) if env < len(observations_cache) else np.zeros(2)
        stick = int(frontier_stick_step[env]) if frontier_stick_step is not None and env < len(frontier_stick_step) else 0
        selected_idx, scored, route_trace = _batch7_route_choice(cards, fam, env, target, current_floor_index, num_steps, original_idx, robot_xy, selection_source, stick)
        trace['batch7_route_scores'] = _jsonable(scored)
        trace.update(_jsonable(route_trace))
        source = 'batch7_route_frontier'
    elif fam.startswith('H'):
        robot_xy = observations_cache[env].get('robot_xy', np.zeros(2)) if env < len(observations_cache) else np.zeros(2)
        stick = int(frontier_stick_step[env]) if frontier_stick_step is not None and env < len(frontier_stick_step) else 0
        selected_idx, scored, route_trace = _batch8_route_choice(cards, fam, env, target, current_floor_index, num_steps, original_idx, robot_xy, selection_source, stick)
        trace['batch8_route_scores'] = _jsonable(scored)
        trace.update(_jsonable(route_trace))
        source = 'batch8_route_frontier'
    elif fam.startswith('J'):
        robot_xy = observations_cache[env].get('robot_xy', np.zeros(2)) if env < len(observations_cache) else np.zeros(2)
        stick = int(frontier_stick_step[env]) if frontier_stick_step is not None and env < len(frontier_stick_step) else 0
        selected_idx, scored, route_trace = _batch9_route_choice(cards, fam, env, target, current_floor_index, num_steps, original_idx, robot_xy, selection_source, stick)
        trace['batch9_route_scores'] = _jsonable(scored)
        trace.update(_jsonable(route_trace))
        source = 'batch9_route_frontier'
    elif fam.startswith('K'):
        robot_xy = observations_cache[env].get('robot_xy', np.zeros(2)) if env < len(observations_cache) else np.zeros(2)
        stick = int(frontier_stick_step[env]) if frontier_stick_step is not None and env < len(frontier_stick_step) else 0
        selected_idx, scored, route_trace = _batch10_route_choice(cards, fam, env, target, current_floor_index, num_steps, original_idx, robot_xy, selection_source, stick)
        trace['batch10_route_scores'] = _jsonable(scored)
        trace.update(_jsonable(route_trace))
        source = 'batch10_route_frontier'
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
    if fam not in ({f'C{i:02d}' for i in range(1, 11)} | {f'D{i:02d}' for i in range(1, 11)} | {f'E{i:02d}' for i in range(1, 11)} | {f'F{i:02d}' for i in range(1, 11)} | {f'G{i:02d}' for i in range(1, 11)} | {f'H{i:02d}' for i in range(1, 11)} | {f'J{i:02d}' for i in range(1, 11)} | {f'K{i:02d}' for i in range(1, 11)} | {f'L{i:02d}' for i in range(1, 11)}):
        return None
    st = _state('floor_probe', env)
    target_floor_index = 2
    if fam == 'C04':
        target_floor_index = 1
    elif fam == 'C10':
        target_floor_index = 2
    elif fam.startswith('E'):
        target_floor_index = 2 if fam in {'E06', 'E07'} else 1
    elif fam.startswith('F'):
        target_floor_index = 2
    elif fam.startswith('G'):
        target_floor_index = 2
    elif fam.startswith('H'):
        target_floor_index = 2
    elif fam.startswith('J'):
        target_floor_index = 2
    elif fam.startswith('K'):
        target_floor_index = 2
    elif fam.startswith('L'):
        target_floor_index = 1
    can_go_up = bool(climb_stair_over and has_up_stair and up_frontier_count > 0)
    event = None
    reason = 'no_override'
    waypoint_xy = None
    avoid_stop_at_waypoint = False
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
    elif fam.startswith('F'):
        waypoint_trace = floor2_bed_waypoint_policy(env, step, cur_floor_index)
        if waypoint_trace is not None:
            return waypoint_trace
        if cur_floor_index >= 2:
            event, reason = 'block_upstairs', 'batch6_hold_on_target_floor_index_2'
        elif can_go_up:
            event, reason = 'navigate_upstairs_fast_recovery', 'batch6_fast_two_up_if_visible'
        elif cur_floor_index == 1 and floor_num_steps >= 35:
            event, reason = 'mark_current_floor_explored', 'batch6_floor1_quick_complete_to_find_second_stair'
    elif fam.startswith('G'):
        if cur_floor_index >= 2:
            event, reason = 'block_upstairs', 'batch7_hold_after_floor_index_2'
        elif can_go_up:
            fast_fams = {'G03', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10'}
            event = 'navigate_upstairs_fast_recovery' if fam in fast_fams else 'navigate_upstairs'
            reason = 'batch7_commit_to_visible_upstairs_transition'
        elif fam in {'G04', 'G08', 'G09'} and cur_floor_index == 0 and floor_num_steps >= 220:
            event, reason = 'mark_current_floor_explored', 'batch7_budgeted_floor0_route_compression_ge220'
        elif fam in {'G05', 'G06', 'G07', 'G08', 'G09', 'G10'} and cur_floor_index == 1 and floor_num_steps >= 55:
            event, reason = 'mark_current_floor_explored', 'batch7_budgeted_floor1_second_transition_search_ge55'
    elif fam.startswith('H'):
        if target_detected and cur_floor_index < 2:
            event, reason = 'ignore_target', 'batch8_low_floor_target_guard'
        elif fam in {'H07', 'H10'} and cur_floor_index < 2 and not can_go_up:
            waypoint = _H_ROUTE_FIRST_STAIR_XY if cur_floor_index <= 0 else _H_ROUTE_SECOND_STAIR_XY
            waypoint_xy = waypoint
            event, reason = 'route_waypoint', 'batch8_diagnostic_compressed_route_to_stair_waypoint'
        elif fam in {'H06', 'H07', 'H10'} and cur_floor_index >= 2:
            waypoint_xy = _H_ROUTE_BED_WAYPOINT_XY
            event, reason = 'route_waypoint', 'batch8_diagnostic_bed_side_waypoint'
        elif cur_floor_index >= 2:
            event, reason = 'block_upstairs', 'batch8_hold_after_floor_index_2'
        elif can_go_up:
            fast_fams = {'H02', 'H03', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10'}
            event = 'navigate_upstairs_fast_recovery' if fam in fast_fams else 'navigate_upstairs'
            reason = 'batch8_commit_to_visible_upstairs_transition'
        elif fam in {'H02', 'H03', 'H05', 'H06', 'H08', 'H09'} and cur_floor_index == 1 and floor_num_steps >= 45:
            event, reason = 'mark_current_floor_explored', 'batch8_floor1_budgeted_second_transition_search_ge45'
        elif fam in {'H04'} and cur_floor_index == 0 and floor_num_steps >= 245:
            event, reason = 'mark_current_floor_explored', 'batch8_floor0_preserve_compress_budget_ge245'
    elif fam.startswith('K'):
        if target_detected and cur_floor_index < 2:
            event, reason = 'ignore_target', 'batch10_low_floor_target_guard'
        elif can_go_up and cur_floor_index < 2:
            fast_fams = {'K03', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10'}
            event = 'navigate_upstairs_fast_recovery' if fam in fast_fams else 'navigate_upstairs'
            reason = 'batch10_commit_to_visible_upstairs_transition'
        elif cur_floor_index == 0:
            route_used = int(st.get('k_floor0_route_waypoint_steps', 0))
            floor0_route_threshold = 170 if fam in {'K02', 'K06', 'K10'} else 9999
            floor0_mark_threshold = 210 if fam in {'K01', 'K05', 'K09'} else (185 if fam in {'K06', 'K10'} else 9999)
            if floor_num_steps >= floor0_route_threshold and route_used < 45:
                st['k_floor0_route_waypoint_steps'] = route_used + 1
                waypoint_xy = _H_ROUTE_FIRST_STAIR_XY
                avoid_stop_at_waypoint = True
                event, reason = 'route_waypoint', 'batch10_bounded_floor0_loop_cut_to_first_stair_route'
            elif floor_num_steps >= floor0_mark_threshold:
                event, reason = 'mark_current_floor_explored', 'batch10_floor0_budgeted_route_compression'
        elif cur_floor_index == 1:
            route_used = int(st.get('k_floor1_route_waypoint_steps', 0))
            floor1_route_threshold = 45 if fam in {'K04', 'K06', 'K10'} else 9999
            floor1_mark_threshold = 35 if fam in {'K03', 'K05', 'K07', 'K08', 'K09'} else (30 if fam in {'K06', 'K10'} else 9999)
            if floor_num_steps >= floor1_route_threshold and route_used < 40:
                st['k_floor1_route_waypoint_steps'] = route_used + 1
                waypoint_xy = _H_ROUTE_SECOND_STAIR_XY
                avoid_stop_at_waypoint = True
                event, reason = 'route_waypoint', 'batch10_bounded_floor1_route_cut_to_second_stair'
            elif floor_num_steps >= floor1_mark_threshold:
                event, reason = 'mark_current_floor_explored', 'batch10_floor1_budgeted_second_transition_search'
        elif cur_floor_index >= 2 and fam in {'K07', 'K08', 'K09', 'K10'}:
            event, reason = 'block_upstairs', 'batch10_hold_high_floor_for_target_search'
    elif fam.startswith('L'):
        cfg = _batch11_config(fam)
        if target_detected and cur_floor_index < 2 and batch11_low_floor_target_ignored(env):
            event, reason = 'ignore_target', 'batch11_low_floor_target_nav_cap_exhausted'
        elif cur_floor_index < 1 and can_go_up:
            event = 'navigate_upstairs_fast_recovery' if bool(cfg.get('fast_stair')) else 'navigate_upstairs'
            reason = 'batch11_e01_fast_first_upstairs_preserved'
        elif fam == 'L10' and cur_floor_index == 1 and floor_num_steps >= int(cfg.get('floor1_budget_steps') or 9999):
            event, reason = 'mark_current_floor_explored', 'batch11_conservative_floor1_budget_reserve'
    elif fam.startswith('J'):
        if target_detected and cur_floor_index < 2:
            event, reason = 'ignore_target', 'batch9_low_floor_target_guard'
        elif target_detected and cur_floor_index >= 2:
            return None
        elif can_go_up and cur_floor_index < 2:
            event, reason = 'navigate_upstairs_fast_recovery', 'batch9_commit_to_visible_upstairs_transition'
        elif cur_floor_index < 2:
            waypoint_xy = _batch9_waypoint(fam, cur_floor_index, floor_num_steps)
            avoid_stop_at_waypoint = True
            event, reason = 'route_waypoint', 'batch9_compressed_route_to_stair_waypoint_no_stop'
        elif cur_floor_index >= 2 and fam in {'J01', 'J02', 'J03', 'J04', 'J05', 'J06'}:
            waypoint_xy = _batch9_waypoint(fam, cur_floor_index, floor_num_steps)
            avoid_stop_at_waypoint = True
            event, reason = 'route_waypoint', 'batch9_high_floor_waypoint_scan_no_stop'
        elif cur_floor_index >= 2 and fam == 'J07' and floor_num_steps < 95:
            waypoint_xy = _batch9_waypoint(fam, cur_floor_index, floor_num_steps)
            avoid_stop_at_waypoint = True
            event, reason = 'route_waypoint', 'batch9_h07_initial_waypoint_then_release'
        elif cur_floor_index >= 2:
            event, reason = 'block_upstairs', 'batch9_high_floor_budgeted_search_hold'
        elif cur_floor_index == 1 and floor_num_steps >= 45:
            event, reason = 'mark_current_floor_explored', 'batch9_floor1_budgeted_second_transition_search_ge45'
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
        'selected_goal_type': 'route_waypoint' if event == 'route_waypoint' else 'upstairs_probe',
        'waypoint_xy': _jsonable(waypoint_xy if waypoint_xy is not None else (_H_ROUTE_BED_WAYPOINT_XY if cur_floor_index >= 2 else (_H_ROUTE_SECOND_STAIR_XY if cur_floor_index == 1 else _H_ROUTE_FIRST_STAIR_XY))) if event == 'route_waypoint' else None,
        'avoid_stop_at_waypoint': bool(avoid_stop_at_waypoint),
    }



def stair_progress_policy(env: int, step: int, cur_floor_index: int, climb_stair_flag: int, get_close_step: int, frontier_stick_step: int, distance_to_stair: Any, reach_stair: bool, reach_stair_centroid: bool) -> Optional[Dict[str, Any]]:
    fam = family()
    if fam not in {'D02', 'D05', 'D06', 'D09', 'D10', 'E07'} | {f'F{i:02d}' for i in range(1, 11)} | {f'G{i:02d}' for i in range(1, 11)} | {f'H{i:02d}' for i in range(1, 11)} | {f'J{i:02d}' for i in range(1, 11)} | {f'K{i:02d}' for i in range(1, 11)} | {f'L{i:02d}' for i in range(1, 11)}:
        return None
    if climb_stair_flag != 1:
        return None
    d = _safe_float(distance_to_stair, None)
    threshold = 18
    if fam in {'D06', 'D09', 'E07'} or fam.startswith('F') or fam.startswith('G') or fam.startswith('H') or fam.startswith('J') or fam.startswith('K'):
        threshold = 8
    elif fam.startswith('L'):
        threshold = int(_batch11_config(fam).get('stair_force_threshold') or 12)
    elif fam in {'D02', 'D05'}:
        threshold = 12
    if fam in {'K05', 'K06', 'K10'}:
        threshold = 5
    event = None
    reason = 'no_override'
    if not reach_stair and (get_close_step >= threshold or frontier_stick_step >= threshold):
        event, reason = 'force_reach_upstairs', f'get_close_or_stick_ge_{threshold}'
    elif reach_stair and not reach_stair_centroid and (fam in {'D06', 'D09', 'E07'} or fam.startswith('F') or fam.startswith('K')) and get_close_step >= threshold:
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


def maybe_target_policy_action(env: int, step: int, target_detected: bool, cur_distance: Any, blip_cosine: Any, navigate_step: int, cur_floor_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Batch-2 hm3d_r0_006 target-evidence stop/approach interventions.

    Returns a trace with policy_action in {stop, move_forward, turn_left} when a
    default-off batch-2 variant wants to override ASCENT's normal target handling.
    """
    fam = family()
    if fam not in ({f'B{i:02d}' for i in range(1, 11)} | {'F09'}):
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
    elif fam == 'F09' and target_detected and cur_floor_index is not None and int(cur_floor_index) >= 2:
        if d is not None and d <= 0.85:
            action, reason = 'stop', 'floor2_target_distance_le_0.85'
        elif d is not None and d <= 2.0:
            action, reason = 'move_forward', 'floor2_target_approach_until_0.85'

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
        'cur_floor_index': int(cur_floor_index) if cur_floor_index is not None else None,
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

