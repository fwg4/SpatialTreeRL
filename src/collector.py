# src/collector.py
import torch
from state_encoder import StateEncoder
from runner import smdp_generator


class SMDPCollector:
    def __init__(self, gamma=0.99, gae_lambda=0.95, time_scale=1000.0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.time_scale = time_scale

    def collect_episode(self, env, agent, device, seed=None):
        state_buffer = []
        action_buffer = []
        last_time_ms = 0
        pending_decision = None
        gen = smdp_generator(env, seed=seed)
        event = next(gen)

        while True:
            if event["type"] == "request_decision":
                current_time_ms = event["obs"]["structured"]["time_ms"]

                if pending_decision is not None:
                    pending_decision["reward"] = event["smdp_reward"] / 10.0
                    pending_decision["delta_t"] = (current_time_ms - last_time_ms) / self.time_scale
                    state_buffer.append(pending_decision)

                result = agent.act(event["obs"], event["active_lines"])

                pending_decision = {
                    "state": result["state"],
                    "value": result["value"],
                    "decisions": result["atomic_decisions"]
                }

                decision_idx = len(state_buffer)
                for d in result["atomic_decisions"]:
                    action_buffer.append({
                        "decision_idx": decision_idx,
                        **d  # type, id, inputs, sample, log_prob
                    })

                last_time_ms = current_time_ms
                event = gen.send(result["env_actions"])

            elif event["type"] in ("game_over", "game_crash"):
                if pending_decision is not None:
                    pending_decision["reward"] = event["smdp_reward"] / 10.0
                    obs = event.get("obs")
                    current_time = obs["structured"]["time_ms"] if (
                        obs and "structured" in obs) else last_time_ms

                    pending_decision["delta_t"] = (
                        current_time - last_time_ms) / self.time_scale
                    state_buffer.append(pending_decision)
                break

            else:
                event = next(gen)

        return self._compute_smdp_gae(state_buffer, action_buffer, device)

    def _compute_smdp_gae(self, state_buffer, action_buffer, device):
        gae = 0.0

        # 反向計算 Advantage
        for i in reversed(range(len(state_buffer))):
            step = state_buffer[i]
            v_s = step["value"]
            v_next = state_buffer[i + 1]["value"] if i + \
                1 < len(state_buffer) else 0.0

            discount = self.gamma ** step["delta_t"]
            td = step["reward"] + (discount * v_next) - v_s
            gae = td + (discount * self.gae_lambda) * gae

            # 直接寫入 buffer，不需要展開
            step["advantage"] = float(gae)
            step["return"] = float(gae + v_s)

        return state_buffer, action_buffer
