# src/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli


class MetroTreePolicy_GATopo(nn.Module):
    def __init__(self, num_nodes: int, num_leaves: int = 20, max_lines: int = 4, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.max_lines = max_lines

        self.params = nn.ParameterDict({
            "node_filter_mu": nn.Parameter(torch.zeros(num_nodes)),
            "node_filter_logstd": nn.Parameter(torch.zeros(num_nodes)),
            "node_route_raw_alpha": nn.Parameter(torch.zeros(num_nodes)),
            "node_route_phi": nn.Parameter(torch.zeros(num_nodes)),
        })

        for i in range(max_lines):
            for key in ["filter", "x", "y"]:
                # 產出 "leaf_x_0_mu", "leaf_filter_2_logstd" 等變數名
                self.params[f"leaf_{key}_{i}_mu"] = nn.Parameter(
                    torch.zeros(num_leaves))
                self.params[f"leaf_{key}_{i}_logstd"] = nn.Parameter(
                    torch.zeros(num_leaves))

    def _make_normal(self, mu, logstd):
        return Normal(torch.tanh(mu), F.softplus(torch.clamp(logstd, -20, 2)) + self.eps)

    def _make_bernoulli(self, raw_alpha, phi, scalar_x):
        return Bernoulli(logits=(raw_alpha - phi) * torch.sigmoid(scalar_x))

    def get_atomic_dist(self, batch_type, batch_ids, batch_inputs=None):
        if batch_inputs is None:
            batch_inputs = {}

        if batch_type == "node_filter":
            mu = self.params["node_filter_mu"][batch_ids]
            logstd = self.params["node_filter_logstd"][batch_ids]
            return self._make_normal(mu, logstd)

        elif batch_type == "node_route":
            scalar_x = batch_inputs.get(
                "scalar_x",
                torch.zeros_like(batch_ids, dtype=torch.float32)
            )

            logits = (
                self.params["node_route_raw_alpha"][batch_ids]
                - self.params["node_route_phi"][batch_ids]
            ) * torch.sigmoid(scalar_x)

            return Bernoulli(logits=logits)

        elif batch_type.startswith("leaf_"):
            mu = self.params[f"{batch_type}_mu"][batch_ids]
            logstd = self.params[f"{batch_type}_logstd"][batch_ids]
            return self._make_normal(mu, logstd)

        else:
            raise ValueError(f"未知的決策類型: {batch_type}")

    def evaluate_batch(self, batch_type, batch_ids, batch_samples, batch_inputs):
        dist = self.get_atomic_dist(batch_type, batch_ids, batch_inputs)

        log_prob = dist.log_prob(batch_samples)
        entropy = dist.entropy()

        return log_prob, entropy
