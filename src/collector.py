# src/collector.py
import torch
from state_encoder import StateEncoder
from state import TensorizedState
from runner import smdp_generator

class SMDPCollector:
    def __init__(self, gamma=0.99, gae_lambda=0.95, time_scale=1000.0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.time_scale = time_scale

    def collect_episode(
        self,
        env,
        agent,
        device,
        seed=None
    ):
        decision_buffer = []
        atomic_buffer = []
        last_time_ms = 0
        pending_decision = None
        gen = smdp_generator(
            env,
            seed=seed
        )
        event = next(gen)

        while True:
            if event["type"] == "request_decision":
                current_time_ms = event["obs"]["structured"]["time_ms"]

                if pending_decision is not None:
                    pending_decision["reward"] = event["smdp_reward"] / 10.0
                    pending_decision["delta_t"] = (
                        current_time_ms - last_time_ms) / self.time_scale
                    decision_buffer.append(pending_decision)

                result = agent.act(event["obs"], event["active_lines"])

                pending_decision = {
                    "state": result["state"],
                    "value": result["value"],
                    "decisions": result["atomic_decisions"]
                }

                decision_idx = len(decision_buffer)
                for d in result["atomic_decisions"]:
                    atomic_buffer.append({
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
                    decision_buffer.append(pending_decision)
                break

            else:
                event = next(gen)

        return self._compute_smdp_gae(decision_buffer, atomic_buffer, device)

    def _compute_smdp_gae(self, decision_buffer, atomic_buffer, device):
        gae = 0.0
        
        # 反向計算 Advantage
        for i in reversed(range(len(decision_buffer))):
            step = decision_buffer[i]
            v_s = step["value"]
            v_next = decision_buffer[i + 1]["value"] if i + 1 < len(decision_buffer) else 0.0
            
            discount = self.gamma ** step["delta_t"]
            td = step["reward"] + (discount * v_next) - v_s
            gae = td + (discount * self.gae_lambda) * gae
            
            # 直接寫入 buffer，不需要展開
            step["advantage"] = float(gae)
            step["target_value"] = float(gae + v_s)

        return decision_buffer, atomic_buffer
    