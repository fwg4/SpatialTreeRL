# src/agent.py
import torch.nn as nn

class MetroTreeAgent(nn.Module):
    def __init__(self, encoder, policy, critic, decision_maker):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.critic = critic
        self.decision_maker = decision_maker

    def act(self, obs, active_lines):
        # 1. 狀態更新與編碼
        state = self.encoder.encode(obs)
        
        # 2. 評估價值 (Critic)
        value = self.critic(state)
        
        # 3. 樹狀決策 (DecisionMaker)
        env_actions, atomic_decisions = self.decision_maker.make_tree_decision(
            state, self.policy, active_lines
        )
        
        return {
            "state": state,
            "value": value.item(),
            "env_actions": env_actions,
            "atomic_decisions": atomic_decisions
        }