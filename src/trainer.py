import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOTrainer:
    def __init__(
            self,
            agent,
            policy_optimizer,
            critic_optimizer,
            clip_ratio=0.2,
            c_entropy=0.01,
            max_grad_norm=0.5):

        self.agent = agent
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_ratio = clip_ratio
        self.c_entropy = c_entropy
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, ppo_epochs, batch_size):
        
        adv_list = [d["advantage"] for d in buffer.decision_buffer]
        if len(adv_list) > 1:
            # 轉為 Tensor 計算平均與標準差
            advs_tensor = torch.tensor(adv_list, dtype=torch.float32)
            mean = advs_tensor.mean().item()
            std = (advs_tensor.std() + 1e-8).item()
            
            # 將算好的結果寫回 buffer
            for d in buffer.decision_buffer:
                d["advantage"] = (d["advantage"] - mean) / std
                
        total_p_loss = 0.0
        total_p_entropy = 0.0
        total_p_kl = 0.0
        total_p_clip_frac = 0.0
        total_v_loss = 0.0
        total_v_explained_var = 0.0
        steps = 0

        for epoch in range(ppo_epochs):
            p_loss, p_entropy, p_kl, p_clip = self._update_policy(buffer, batch_size)
            v_loss, v_explained_var = self._update_critic(buffer, batch_size)

            total_p_loss += p_loss
            total_p_entropy += p_entropy
            total_p_kl += p_kl
            total_p_clip_frac += p_clip
            total_v_loss += v_loss
            total_v_explained_var += v_explained_var
            steps += 1

        return {
            "policy_loss": total_p_loss / steps if steps > 0 else 0.0,
            "entropy": total_p_entropy / steps if steps > 0 else 0.0,
            "value_loss": total_v_loss / steps if steps > 0 else 0.0,
            "approx_kl": total_p_kl / steps if steps > 0 else 0.0,
            "clip_frac": total_p_clip_frac / steps if steps > 0 else 0.0,
            "explained_var": total_v_explained_var / steps if steps > 0 else 0.0,
        }

    def _update_policy(self, buffer, batch_size):
        epoch_p_loss = 0.0
        epoch_entropy = 0.0
        epoch_approx_kl = 0.0
        epoch_clip_frac = 0.0
        batches = 0
        
        for batch in buffer.get_policy_batches(batch_size):
            
            self.policy_optimizer.zero_grad()

            new_logp, entropy = self.agent.policy.evaluate_batch(
                batch["type"],
                batch["ids"],
                batch["samples"],
                batch["inputs"]
            )

            ratio = torch.exp(new_logp - batch["log_probs"])

            with torch.no_grad():
                log_ratio = new_logp - batch["log_probs"]
                # Approx KL 經典算法: (exp(log_ratio) - 1) - log_ratio
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) >
                             self.clip_ratio).float().mean().item()

            adv = batch["advantages"]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                1.0 + self.clip_ratio) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.c_entropy * entropy.mean()

            loss = policy_loss + entropy_loss
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.agent.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            epoch_p_loss += policy_loss.item()
            epoch_entropy += entropy.mean().item()
            epoch_approx_kl += approx_kl
            epoch_clip_frac += clip_frac
            batches += 1

        return (epoch_p_loss / batches, epoch_entropy / batches,
                epoch_approx_kl / batches, epoch_clip_frac / batches) if batches > 0 else (0.0, 0.0, 0.0, 0.0)

    def _update_critic(self, buffer, batch_size):
        epoch_v_loss = 0.0
        epoch_explained_var = 0.0
        batches = 0

        for batch in buffer.get_value_batches(batch_size):
            self.critic_optimizer.zero_grad()

            preds = self.agent.critic(
                batch["features"], batch["masks"]).squeeze(-1)
            loss = F.mse_loss(preds, batch["target_values"])

            with torch.no_grad():
                y_true = batch["target_values"]
                y_pred = preds
                var_y = torch.var(y_true)
                explained_var = torch.nan if var_y == 0 else 1.0 - \
                    torch.var(y_true - y_pred) / (var_y + 1e-8)

            # Critic 獨立優化，無需乘上 c_value
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.agent.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            epoch_v_loss += loss.item()
            if not torch.isnan(explained_var):
                epoch_explained_var += explained_var.item()
            batches += 1

        return (epoch_v_loss / batches, epoch_explained_var / batches) if batches > 0 else (0.0, 0.0)
