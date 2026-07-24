# Codebase Learning Journal

> This journal tracks your understanding of the codebase. Claude updates it
> during learning sessions to maintain continuity across conversations.

## Focus & Goals

- **Primary goal**: 熟悉掌握 ASCENT 论文与代码，以它作为 object-goal navigation 实验基线，完成创新性改进并发表科研论文
- **Interested in**: 联通论文方法、代码执行路径、实验复现、局限分析与科研改造
- **Background**: Python 与深度学习较熟悉；ROS 有一定经验；Habitat 初学；已成功运行 ASCENT，HM3D/MP3D val 指标与论文接近，环境主要由 AI 协助配置
- **Learning style**: 以真实执行链路和实验为主，结合论文图表与主动回忆；偏好先获得精确文件/行号，再自行阅读推导

setup_done: true

## Concept Mastery Map

### 🟢 Confident
<!-- Concepts you can explain to others and apply in new situations -->
<!-- Format: - Concept name (date achieved) -->

### 🟡 Learning
<!-- Partial understanding, making connections, have active questions -->
<!-- Format: - Concept name - brief note on what's unclear -->
- ASCENT 顶层模块职责 - 已能推断 policy、地图、LLM 与 PointNav 的高低层分工；需修正入口与 trainer 的职责边界
- 单个 ObjectNav episode 执行链 - 已正确定位闭环，并理解 VectorEnv auto-reset 后 `observation=B`、`done/info=A`；待追踪 mask 如何重置 policy（hint count: 3）
- Habitat ObjectNav 成功判定 - 能正确应用 `is_stop_called AND distance < success_distance` 判定；等待间隔复习后再升为 Confident
- ASCENT episode reset 与地图所有权 - 已预测 value/obstacle 等场景状态必须清空；正在追踪 `Policy._reset → Map_Controller.reset`
- Inter-floor reasoning 触发条件 - 已区分 `τperiod` 只约束跨楼层 LLM、`τfloor` 是当前楼层地图累计步数；发现公开代码存在楼层数未接线疑点，待作者确认
- Python 参数传递与 planner 状态所有权 - 正在区分局部形参 `floor_num`、planner 实例属性 `self.floor_num`、Map Controller 属性三者的作用域与引用关系（hint count: 1）
- 目标检测到 STOP 的证据链 - 正在区分视频检测轮廓、通过深度/SAM/DBSCAN 后的 object cloud、PointNav goal 与 BLIP double-check
- ASCENT 视频可视化图层 - 正在区分 Habitat top-down map、ASCENT obstacle/value/object maps 及其 frontier/goal 标记

### 🔴 Need to Explore
<!-- Cannot explain or apply yet, need guided exploration -->
<!-- Format: - Concept name -->
- ASCENT 的研究问题、核心贡献与关键假设
- 环境、数据集、模型与 LLM 依赖的复现流程
- 指标、消融实验与论文结果的可复核性
- 可发表改进的研究缺口、实验设计与证伪标准

## Open Questions

<!-- Questions that surfaced during learning, to be resolved -->
<!-- Format: - [ ] Question text -->
<!--         - [x] Resolved question - **Answer**: brief resolution -->
- [x] 学习者当前的 Python、ROS、Habitat/ObjectNav、深度学习与机器人导航基础如何？ **Answer**: Python/深度学习较熟悉，ROS 尚可，Habitat 初学；已跑通 ASCENT
- [x] 当前最紧迫任务是先理解论文、跑通基线，还是定位创新点？ **Answer**: 本轮先理解论文模块与 episode 执行链，复现故障暂不展开
- [ ] ASCENT 论文方法的主数据流怎样映射到仓库文件和运行时对象？
- [ ] 官方/当前仓库能否复现论文主结果，复现差距在哪里？
- [ ] 为什么评测首次运行偶发 environment abort，而 recovery 后可继续？这是否造成 episode 选择或指标偏差？
- [ ] `Ascent_Policy.act` 如何由 YAML 名称经 registry 动态绑定到 `self._agent.actor_critic.act`？
- [x] Environment 的 episode 步数在哪里递增并触发终止？ **Answer**: `habitat/core/env.py::_update_step_stats()` 递增 `_elapsed_steps`；`_past_limit()` 比较 `max_episode_steps`，达到上限后设置 `_episode_over=True`
- [x] VectorEnv 在 `done=True` 后返回的是终止帧，还是自动 reset 后下一 episode 的初始观测？ **Answer**: 默认 auto-reset 时，worker 先得到 A 的 `done/info`，随后用 `env.reset()` 将 observation 替换为 B 的初始观测，再一起返回
- [ ] `Ascent_LLM_Planner.reset(env)` 已定义，但当前 `Ascent_Policy._reset()` 中未见调用；跨 episode 状态是否泄漏，后果是什么？
- [x] `_get_best_frontier_with_llm()` 的 `floor_num` 默认值为 `[1]`，唯一调用点未传入 `Map_Controller.floor_num`；当前多楼层 LLM 分支是否实际不可达？ **Answer**: 是。当前唯一调用中未传 `floor_num`，因此 env 0 绑定到默认列表 `[1]`，`floor_num[0] > 1` 恒假；只有单楼层 LLM 路径和“无 frontier 时走未探索楼梯”的非 LLM 路径仍可执行
- [ ] `Map_Controller.floor_num`、`_get_best_frontier_with_llm(..., floor_num)` 与 `Ascent_LLM_Planner.self.floor_num` 三处楼层数未同步，是公开 refactor 的遗漏还是论文实验代码另有实现？
- [ ] 已向官方仓库提交 floor-count wiring GitHub issue；待作者回复。之后在独立 branch 中恢复论文预期逻辑，并与原始公开基线做受控对照实验
- [ ] 为什么调用时传入 `floor_num=Map_Controller.floor_num` 只会替代局部形参，而不会自动改写 `Ascent_LLM_Planner.self.floor_num`？
- [x] 60 步与 100 步阈值是否冲突或覆盖？ **Answer**: 不冲突；条件用 AND。60 是距上次楼层决策的全局冷却时间，100 是当前楼层至少探索的步数，两个时钟必须同时满足
- [x] 目标第一次出现在当前帧时，会等下一步才转入 navigate 吗？ **Answer**: 不会；当前 `act()` 先更新 object map，再查询 goal，因此只要点云被接受，同一次 `act()` 就能进入 `_navigate()`
- [ ] 为什么视频中看似正确的目标检测有时没有导致 STOP？需分别验证候选轮廓是否形成 object cloud、BLIP double-check 是否成功、以及接近后是否触发清空与区域 blacklist
- [x] 视频右上 Habitat top-down map 中白/蓝/红空心圆各表示什么？ **Answer**: 白色是通往 BaseExplorer 所选 frontier 的下一短程 waypoint（不是 frontier）；蓝色是其余 frontier waypoint；红色是 BaseExplorer 按 A* 代价选出的 closest frontier。红色不一定等于 ASCENT 经 value/LLM 选择的 `_last_goal`
- [x] 右上 BaseExplorer frontiers 与 value map frontiers 是否是同一集合？ **Answer**: 不是。BaseExplorer 从 Habitat pathfinder 真值 navmesh 与自己的 fog-of-war 检测一套 frontier，只在右上测量图中画圈；value map 的圆来自 ASCENT 用 RGB-D 构建的 `ObstacleMap.frontiers`。`Map_Controller` 将后一套写入 policy cache，LLM 与 PointNav 使用后一套。两者有时空间上接近，但没有身份或数量上的一一对应关系
- [x] BaseExplorer 是否参与 ASCENT 的机器人行为？ **Answer**: 不直接参与当前 ASCENT 的动作控制。它作为 Habitat lab sensor 每步仍会维护 GT-map/fog/frontiers 并计算一个建议动作，但 `Ascent_Policy` 不读取该动作；trainer 执行的是 ASCENT 返回的 PointNav/楼梯/STOP 动作。BaseExplorer 状态用于右上可视化及部分事后可行性/失败分析
- [x] Value map 黄色圈是否参与 ASCENT 决策？ **Answer**: 黄色圈只是 `_last_goal` 与当前 frontier 精确匹配时绘制的 selected-frontier marker；ASCENT/PointNav 使用的是二维 goal 坐标，所有彩色空心圆都是输出侧可视化，不会反馈给策略
- [x] ASCENT 视频 2×4 custom layout 的各子图分别是什么？ **Answer**: 上排依次是带检测轮廓的 RGB、RedNet MPCAT40 语义分割、ASCENT obstacle map、Habitat multi-floor top-down map；下排依次是带同轮廓与指标的 depth、ASCENT object map、ASCENT value map、frontier 历史 RGB 与 Qwen 决策摘要
- [x] 右下角 `vlm_input` 是否就是送给 Qwen 的视觉输入？ **Answer**: 不是。它是 frontier 第一次出现时缓存的历史原始 RGB，主要用于视频解释；当前单楼层 Qwen prompt 使用对应时刻由 RAM 和 Places365 得到的 objects/room 文字描述及统计先验

## Spaced Review Queue

<!-- Concepts scheduled for review based on spaced repetition -->
<!-- Format: - [ ] Concept (review by: YYYY-MM-DD) - Nth review -->
<!--         - [x] Concept (completed N reviews) - moved to Confident -->
- [ ] ASCENT 顶层模块职责（review by: 2026-07-25）- 第 2 次复习；2026-07-22 完成第 1 次复习
- [ ] Habitat ObjectNav success 条件（review by: 2026-07-23）- 第 1 次复习
- [ ] VectorEnv auto-reset 与 episode 归属（review by: 2026-07-23）- 第 1 次复习
- [ ] BaseExplorer frontier 与 ASCENT frontier 所有权（review by: 2026-07-24）- 第 1 次复习；2026-07-23 首次正确主动回忆“普通探索看 value map 黄圈”

## Aha Moments

<!-- Insights captured in your own words—these cement understanding -->
<!-- Format: ### YYYY-MM-DD: Brief title -->
<!--         Insight description in your own words -->

### 2026-07-20: 高层规划与底层执行
“上层给 PointNav 一个 frontier 后，PointNav 比较偏底层，直接负责走过去”；Map Controller 管理多楼层地图，LLM Planner 在特定时刻参与 frontier 选择。

### 2026-07-22: 动作—观测闭环
“`self.envs.step(step_data)` 把 `policy.act()` 的动作传给 simulator，再从 outputs 得到 observations；RGB-D 由这次环境 step 返回。”

### 2026-07-22: 到达不等于成功
“即使进入成功距离，没有主动 STOP，success 仍应为 0。”

### 2026-07-22: Auto-reset 的混合返回值
“`observation` 属于下一个 Episode B，但 `done` 和 `info` 属于刚结束的 Episode A。”

### 2026-07-22: 场景状态必须隔离
“新 episode 开始时基本所有内部状态都要重置，例如 value map 和 obstacle map。”

### 2026-07-22: 计时状态泄漏压缩规划预算（修复楼层数接线后的反事实）
若先修复多楼层分支可达性，而仍不调用 planner reset：A 遗留 `multi_floor_ask_step=400`，B 从 0 重新计数，则间隔条件到第 460 步才可能满足，已接近 500 步上限。当前公开代码里该分支先被默认 `floor_num=[1]` 阻断。

### 2026-07-22: Inter-floor reasoning 需要楼梯依据
从论文推断：“只有 stair-related frontiers 存在时，才会考虑 LLM 的 inter-floor reasoning。”代码中的 `floor_num > 1` 似乎试图作为已发现楼梯/额外楼层的间接门控。

### 2026-07-22: 60 步不是所有 LLM 的冷却
`multi_floor_ask_step` 只在进入 multi-floor prompt 前更新，因此记录的是上一次跨楼层 LLM 尝试的全局 policy step；单楼层 frontier LLM 调用不会更新它，实际完成爬楼也不会更新它。

### 2026-07-22: 三套楼层数状态没有接通
地图模块的 `Map_Controller.floor_num[0]` 可随新楼层地图变为 2/3；函数形参因调用点漏传而固定看到默认值 1；planner 自身的 `self.floor_num[0]` 也只初始化/重置为 1，却被 prompt 构造与响应校验使用。

### 2026-07-22: 当前帧可以立即切换到目标导航
学习者正确预测：`act()` 在动作选择前先用当前 RGB-D 更新 object map，因此当前检测若成功形成点云，当前 timestep 就从 explore 转入 navigate。

### 2026-07-23: 同一视频中混合了“策略信念”和“仿真真值”
Obstacle/value/object map 是 ASCENT 根据 RGB-D 构建的在线内部状态；右上 Habitat top-down map 来自 pathfinder 与 episode goals，含完整 navmesh、真值目标和 oracle shortest path，只能用于评测/诊断，不能当成 ASCENT 已知信息。

### 2026-07-23: 两套 frontier 只是同名，不是共享对象
右上图的 BaseExplorer frontier 来自 Habitat navmesh + BaseExplorer fog；obstacle/value map 的 ASCENT frontier 来自在线 RGB-D obstacle/explored maps。当前策略选择、LLM 排序和 PointNav goal 均沿后一条链路。

## Session Log

<!-- Brief log of each learning session -->
<!-- Format:
### YYYY-MM-DD
- **Explored**: Topics/files covered
- **Learned**: Key takeaways
- **Struggled with**: Areas that needed multiple hints
- **Next**: Suggested follow-up areas
-->

### 2026-07-20
- **Explored**: 初始化学习计划；盘点论文、README、入口 `ascent/run.py`、trainer、policy、地图与 LLM planner 文件
- **Learned**: 正确推断了 policy、Map Controller、LLM Planner、PointNav 的总体分工；确认学习背景与已跑通基线
- **Struggled with**: `run.py` 与 trainer 的 Habitat 框架职责边界仍需澄清
- **Next**: 追踪 `execute_exp → AscentTrainer._eval_checkpoint → Ascent_Policy.act`，理解闭环由谁驱动

### 2026-07-22
- **Explored**: `AscentTrainer` 的 `policy.act → env.step → observations` timestep 闭环
- **Learned**: 正确判断下一帧 RGB-D 来自环境输出，找到 policy 主动 STOP 逻辑，并正确预测“仅接近目标但未 STOP”不算成功
- **Struggled with**: 尚未完全区分 policy 的 `_num_steps` 镜像计数与 Habitat environment 的权威 `_elapsed_steps`；配置/注册表动态绑定仍待巩固
- **Current difficulty signal**: 初次预测 auto-reset 后仍返回终止帧，并猜测 `current_episodes_info` 用于 while；需要沿 worker 中 `step → reset → write` 的顺序重试
- **Resolved with hint**: 沿 `_worker_env` 的执行顺序，正确修正为 observation 属于 B、done/info 属于 A；`current_episodes_info` 用于给 A 的最终指标保留正确身份
- **New code observation**: 地图由 `Map_Controller` 的 per-env/per-floor object、obstacle、value map 列表维护；其 `reset()` 删除额外楼层并清空首层。LLM planner 虽定义了 `reset()`，policy reset 中却未见调用，待验证影响
- **Reference audit**: 全仓库（排除 third_party/.git）未找到 `llm_planner.reset(...)` 调用；当前静态证据支持“方法已设计但未接入正常 episode reset 路径”，下一步由学习者阅读计时条件并预测状态泄漏影响
- **New reachability concern**: learner 正确算出 stale timer 需到 B 的第 460 步；进一步审计发现 `_get_best_frontier_with_llm` 唯一调用点未传 `floor_num`，其默认 `[1]` 可能令 `floor_num[env] > 1` 恒假，需先验证分支可达性
- **Paper-code connection**: learner 从论文识别出 inter-floor reasoning 应以 stair-related frontiers 为前提；代码中 60/100 分别对应周期冷却与当前楼层探索门槛，而 `floor_num>1` 是楼梯/额外楼层存在性的间接代理
- **Reachability audit resolved**: 当前相关文件与官方 `origin/main` 一致；唯一调用点确实省略 `floor_num`，故 env 0 的分支形参实际为默认值 1，周期性 multi-floor LLM 分支静态不可达。另发现 prompt/响应解析使用的 planner `self.floor_num` 也未与 Map Controller 同步
- **Still reachable**: 单楼层 LLM frontier 选择仍经 `_decide_frontier_with_llm → llm_analyze_single_floor` 执行；当前楼层 frontier 耗尽后仍可通过非 LLM 楼梯兜底跨楼层，因此“能爬楼/指标接近”不能证明 multi-floor LLM 分支被调用
- **Current clarification**: learner 已正确理解显式传参会替代 `_get_best_frontier_with_llm` 的默认形参；当前卡点是局部形参不会自动赋值给 planner 的同名实例属性，下一轮用最小 Python 例子检验作用域与数据流
- **Research workflow decision**: learner 已提交 floor-count wiring GitHub issue；当前保持基线源码不变，待后续新建 branch 修复论文逻辑并比较性能
- **Next code trace**: 从 `Ascent_Policy.act()` 单步内部顺序入手，连接 observation 缓存、目标/楼梯检测、三类地图更新、goal 查询与 explore/navigate 动作选择
- **Successful prediction**: learner 正确判断当前帧检测可在同一次 `act()` 中形成 goal 并立即进入 navigate
- **New empirical observation**: 复现视频中有时出现视觉上正确的目标候选，但 agent 很快经过并继续探索；当前追踪 `target detection → SAM/depth/DBSCAN cloud → BLIP double-check → distance gate → STOP/blacklist`
- **Next**: 追踪 STOP 如何经 task 令 environment 产生 `done=True`，并预测 VectorEnv auto-reset 后返回的 observation；之后完成 YAML → registry → `Ascent_Policy.act` 绑定追踪

### 2026-07-23
- **Explored**: 复现视频 custom layout 与右上 Habitat multi-floor top-down map 的 waypoint/frontier 画法
- **Learned**: 右上图的白圈是到 BaseExplorer closest frontier 的 next waypoint；蓝圈是其他 frontier；红圈是 BaseExplorer closest frontier，并非必然是 ASCENT 的 value/LLM 最终选择
- **Clarified**: Value map 中黄色空心圈是 ASCENT 当前 `_last_goal` 恰好仍属于当前 frontier 集合时的可视化；圆圈颜色/图像本身不参与规划，策略使用 frontier/goal 坐标
- **Full video legend**: 上排 RGB / RedNet seg / obstacle map / Habitat GT top-down；下排 annotated depth+metrics / object map / value map / cached frontier RGB+Qwen response。前三张在线地图共享绿色轨迹与橙色朝向 agent；obstacle map 蓝圈是 ASCENT 当前普通 frontiers，value map 红圈是同一 frontier 集，黄色圈是当前选中 frontier，绿色圈是非-frontier `_last_goal`
- **Important implementation caveats**: Habitat 大地图的红色既可能来自真值 goal，也可能来自可视化函数叠加的 ASCENT target point cloud；右下 frontier 图是历史缓存而非当前帧，且 Qwen 实际接收文字区域描述。`Ascent_LLM_Planner.reset(env)` 仍未接入 policy reset，因此该缓存理论上可跨 episode 残留
- **Frontier ownership clarified**: BaseExplorer frontier 与 ASCENT `ObstacleMap.frontiers` 是两套独立集合；前者只供右上 Habitat measurement 可视化，后者被写入 policy cache，并用于 value-map marker、LLM 选择和 PointNav
- **Action chain confirmed**: 普通探索时执行链为 `ObstacleMap.frontiers → observations_cache["frontier_sensor"] → LLM best_frontier → _pointnav(best_frontier) → trainer env.step(action)`；BaseExplorer 的 action/frontier 不进入这条 ASCENT 动作链
- **Successful recall**: learner 正确回答普通 frontier 探索时应看 value map 黄圈，而不是右上 BaseExplorer 红圈
- **BaseExplorer role bounded**: 它在后台运行并提供 measurement/diagnostic state，但其返回的动作 observation 未进入 `Ascent_Policy` 的 cache 或最终 `PolicyActionData.actions`，所以不改变当前 ASCENT 的机器人运动
- **Paused**: “正确目标检测却错过”的 `detection → cloud → BLIP double-check → STOP/blacklist` 追踪等待学习者稍后继续

---

## Quick Reference

### Mastery Levels

| Level | Meaning | Indicator |
|-------|---------|-----------|
| 🔴 Confused | Cannot explain or apply | Need exploration |
| 🟡 Learning | Partial understanding | Making connections |
| 🟢 Confident | Can explain & apply | Ready to teach others |

### Review Schedule (Spaced Repetition)

| Review # | Wait Time | After Success |
|----------|-----------|---------------|
| 1st | 1 day | Schedule 2nd |
| 2nd | 3 days | Schedule 3rd |
| 3rd | 1 week | Schedule 4th |
| 4th | 2 weeks | Schedule 5th |
| 5th+ | Consider 🟢 Confident | Long-term memory |

### How to Use This Journal

1. **Start sessions** by reviewing Focus & Goals and Open Questions
2. **During learning** let Claude update mastery levels and add questions
3. **Capture insights** in Aha Moments using your own words
4. **Check review queue** at session start for spaced repetition
5. **Keep it honest** — 🟡 is fine! Moving to 🟢 too fast defeats the purpose
