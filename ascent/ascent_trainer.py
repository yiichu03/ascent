import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    # generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from ascent.utils import generate_video
from omegaconf import OmegaConf
from habitat_baselines.rl.ppo.evaluator import pause_envs ## For Habitat 3.0 
from gym import spaces
import time
def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="ascent")
class AscentTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}
# liuyi: checkpoint_path:'dummy_policy.pth' why? what is the dummy_policy.pth used for?
# ASCENT 的高层探索策略不是通过 PPO 训练得到的，但 Habitat-Baselines 的 evaluation 框架围绕 checkpoint 设计，所以仍保留 checkpoint 接口。
        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
# when liuyi debug, the video_option=[],所以这整段跳过。第一次理解主流程时，不需要设置断点。
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            # 只有保存视频时才添加额外相机、打开 debug_render。
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True # do not know why

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # 创建 Habitat environments
        # 执行完成后，self.envs 对外提供统一接口：self.envs.reset() self.envs.step(actions) self.envs.current_episodes() self.envs.close()
        self._init_envs(config, is_eval=True) 

        # 创建 Agent 和 Ascent Policy
        # self._agent is Habitat agent/access-manager 外壳;
        # self._agent.actor_critic is 真正的 Ascent_Policy(虽然变量名叫 actor_critic，但在 ASCENT 中它不代表当前正在训练 actor-critic，只是 Habitat-Baselines 的统一接口名称。)
        self._agent = self._create_agent(None) 
        # 确认动作空间是离散还是连续，以及 action tensor 形状。
        action_shape, discrete_actions = get_action_space_info(self._agent.actor_critic.policy_action_space) # actor_critic. for Habitat 3.0

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)
        # 环境 reset：取得第一帧 observation 这是 episode 真正开始的位置。 1. Habitat 原始 observation
        observations = self.envs.reset()
        # 把“每个 environment 一个字典”整理为“每个 sensor 一个 batched tensor”：  2. → batched tensor
        batch = batch_obs(observations, device=self.device)
        # 应用 resize 等 observation transform。   3. → policy 可以直接使用的输入
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        if hasattr(self._agent, "_ppo_cfg"):
            test_recurrent_hidden_states = torch.zeros(
                (
                    self.config.habitat_baselines.num_environments,
                    0,
                    self._agent._ppo_cfg.hidden_size, # For Habitat 3.0
                    # *self._agent.actor_critic.hidden_state_shape, 
                ),
                device=self.device,
            )
        else: 
            # 循环状态初始化
            # 这是 Habitat recurrent policy 接口需要的隐藏状态。
            # ASCENT 高层策略本身主要依靠地图和显式状态，不依赖这里的 RNN hidden state，但仍要满足统一的 act() 签名。
            test_recurrent_hidden_states = torch.zeros(
                (
                    self.config.habitat_baselines.num_environments,
                    0,
                    self._agent._agents[0]._ppo_cfg.hidden_size, # For Habitat 3.0
                    # *self._agent.actor_critic.hidden_state_shape, 
                ),
                device=self.device,
            )
        # prev_actions记录上一步动作。第一步没有上一步动作，所以初始化为零。
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        # not down masks 第一步是 False，告诉 policy：这是新 episode 需要 reset 内部状态;
        # 后续 episode 未结束时变成 True。
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            # *self._agent.masks_shape,
            device=self.device,
            dtype=torch.bool,
        )
        sensor_objectgoal = [
            "chair",
            "bed",
            "plant",
            "toilet",
            "tv",
            "sofa",
        ]
        stats_episodes: Dict[Any, Any] = {}  # 保存每个 episode 的最终指标； dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0) # 记录每条 episode 已评测多少次。

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from ascent.habitat_visualizer import HabitatVis

        num_successes = 0
        num_spl = 0
        num_dtg = 0
        num_total = 0
        scene_stats = {}  # 用于记录每个场景的统计信息

        hab_vis = HabitatVis(self.envs.num_envs)
        goal_name = ["" for _ in range(self.envs.num_envs)]
        # 最核心的 timestep 循环
        # 只要：目标 episode 数还没完成 并且仍有活跃 environment 就继续执行。
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            # 读取当前 episode 信息. 为什么要在 env.step() 前保存？
            current_episodes_info = self.envs.current_episodes()
            # 调用 Ascent_Policy.act()
            with inference_mode(): # inference_mode() 告诉 PyTorch： 不需要梯度不进行反向传播只做推理
                action_data = self._agent.actor_critic.act(
                    batch, # 当前 RGB-D、GPS、目标类别等观测
                    test_recurrent_hidden_states, # Habitat 接口状态
                    prev_actions, # 上一步动作
                    not_done_masks, # 是否为 episode 第一帧
                    deterministic=False,
                    current_episodes_info =current_episodes_info,
                )
                # 输出 action_data，主要包含：
                # actions: policy 动作 tensor; env_actions:  转换后要交给 environment 的动作; policy_info: ASCENT 产生的目标、地图、frontier、LLM response 等辅助信息
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                # 把当前动作保存成下一步的 previous action。
                prev_actions.copy_(action_data.actions)  # only need to copy
            # batch_t→ Ascent_Policy.act()→ action_t
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.            
            if hasattr(self._agent, '_agents') and isinstance(self._agent._agents[0]._actor_critic.policy_action_space, spaces.Discrete):
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                # 把动作转换成 CPU 数据:
                # policy 在 GPU 上产生动作→ 转为 CPU 标量→ 交给 Habitat worker
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]
            # 执行 Habitat 动作
            # 它把动作真正交给 simulator，例如：MOVE_FORWARD, TURN_LEFT, ...
            outputs = self.envs.step(step_data)

            # 四个结果： 1. observations 动作执行后的下一帧 RGB-D 和传感器信息。
            # 2. rewards_l 评测时虽然不训练，仍然记录 reward。
            # 3. dones 当前 episode 是否结束。
            # 4. infos Habitat measurements，例如：success spl distance_to_goal...
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            
            # 合并 policy 信息
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                # 把ASCENT policy 产生的辅助字段合并到 Habitat info。 这使视频和日志能够同时看到： Habitat 真值指标,ASCENT 内部决策信息
                infos[i].update(policy_infos[i])

            # 准备下一次循环
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore
            
            # done=False → mask=True; done=True  → mask=False
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            # 至此一个 timestep 完成。
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    if "vlm_response" in action_data.policy_info[i]:       
                        hab_vis.collect_data_with_third_view_and_seg_map_vlm_input(batch, infos, action_data.policy_info, i)
                    else:
                        hab_vis.collect_data_with_third_view_and_seg_map(batch, infos, action_data.policy_info, i)
                    if "num_steps" in infos[i] and infos[i]["num_steps"] == 1 and action_data.policy_info is not None:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        goal_name[i] = action_data.policy_info[i]['target_object']
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}. The goal is {goal_name[i]} for this episode.") # for debug
                # episode ended 
                # Episode 结束处理
                if not not_done_masks[i].item(): # 等价于： if dones[i]:
                    pbar.update()
                    # 先提取指标：
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use k 标识刚刚结束的 episode。
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_spl += episode_stats["spl"]
                    num_dtg += episode_stats["distance_to_goal"]
                    num_total += 1
                    # 更新场景统计信息
                    scene_id = current_episodes_info[i].scene_id
                    if scene_id not in scene_stats:
                        scene_stats[scene_id] = {
                            "num_successes": 0,
                            "num_spl": 0,
                            "num_dtg": 0,
                            "num_total": 0,
                        }
                    scene_stats[scene_id]["num_successes"] += episode_stats["success"]
                    scene_stats[scene_id]["num_spl"] += episode_stats["spl"]
                    scene_stats[scene_id]["num_dtg"] += episode_stats["distance_to_goal"]
                    scene_stats[scene_id]["num_total"] += 1
                    # 打印当前场景的平均值
                    scene_num_total = scene_stats[scene_id]["num_total"]
                    scene_success_rate = scene_stats[scene_id]["num_successes"] / scene_num_total * 100
                    scene_avg_spl = scene_stats[scene_id]["num_spl"] / scene_num_total
                    scene_avg_dtg = scene_stats[scene_id]["num_dtg"] / scene_num_total
                    print(f"Success rate of Scene {scene_id}: {scene_success_rate:.5f}% ({scene_stats[scene_id]['num_successes']} out of {scene_num_total})")
                    print(f"Average Spl of Scene {scene_id}: {scene_avg_spl:.5f}")
                    print(f"Average Dtg of Scene {scene_id}: {scene_avg_dtg:.5f}")
                    print(f"Till Now Average Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")
                    print(f"Till Now Average Spl: {num_spl / num_total * 100:.2f}%")
                    print(f"Till Now Average Dtg: {num_dtg / num_total}")
                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )
                    import ascent.failure_logger as failure_logger

                    try:
                        # Failure 分类
                        # 根据最终 metric 将失败归类，例如：
                        # false_positive, false_negative, timeout, collision
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception as e:
                        print(f"Error information:{e}")
                        failure_cause = "Unknown"
                        failure_logger.failure_stats[failure_cause] += 1
                        failure_logger.failure_records.append({
                            "failure_cause": failure_cause,
                            "scene_id": current_episodes_info[i].scene_id,
                            "episode_id": current_episodes_info[i].episode_id
                        })
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        if "vlm_response" in action_data.policy_info[i]:
                            rgb_frames[i] = hab_vis.flush_frames_with_rednet_vlm_input(failure_cause, i)
                        else:
                            rgb_frames[i] = hab_vis.flush_frames_with_rednet(failure_cause, i)

                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=current_episodes_info[i].episode_id,
                            goal_name=goal_name[i],# goal_name, sensor_objectgoal[observations[i]['objectgoal'].item()]
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                            failure_cause=failure_cause,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )
                # else:
                #     goal_name = sensor_objectgoal[observations[i]['objectgoal'].item()]

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs( # self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
