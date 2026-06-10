# src/buffer.py
import torch
import random
from collections import defaultdict
from typing import List, Dict, Generator
from state import StateBatch


class PPOBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.state_buffer: List[Dict] = []
        self.action_buffer: List[Dict] = []

    def add_episode(self, decision_buf: List[Dict], atomic_buf: List[Dict]):
        offset = len(self.state_buffer)
        self.state_buffer.extend(decision_buf)

        for a in atomic_buf:
            a_copy = a.copy()
            a_copy["decision_idx"] += offset
            # 取消 advantage 的複製，只保留 index 以確保資料一致性
            self.action_buffer.append(a_copy)

    def get_policy_batches(self, batch_size: int):
        # 1. 分組
        grouped = defaultdict(list)
        for a in self.action_buffer:
            grouped[a["type"]].append(a)

        # 2. 產出 Batch
        for decision_type, items in grouped.items():
            indices = list(range(len(items)))
            random.shuffle(indices)

            for start_idx in range(0, len(items), batch_size):
                batch_indices = indices[start_idx: start_idx + batch_size]
                batch = [items[i] for i in batch_indices]

                advantages = [self.state_buffer[b["decision_idx"]]
                              ["advantage"] for b in batch]

                # ==========================================
                # 🌟 [最佳化] 穩健的 Collate 邏輯
                # ==========================================
                batched_inputs = {}
                # 確保 inputs 存在且為 dict (防呆 KeyError)
                if len(batch) > 0 and "inputs" in batch[0] and isinstance(batch[0]["inputs"], dict):
                    for k in batch[0]["inputs"]:
                        # 直接用純 Python List 建立 Tensor，效能最佳
                        batched_inputs[k] = torch.tensor(
                            [b["inputs"][k] for b in batch],
                            dtype=torch.float32,
                            device=self.device
                        )
                # ==========================================

                yield {
                    "type": decision_type,
                    "inputs": batched_inputs,
                    "ids": torch.tensor([b["id"] for b in batch], dtype=torch.long, device=self.device),
                    "samples": torch.tensor([b["sample"] for b in batch], dtype=torch.float32, device=self.device),
                    "log_probs": torch.tensor([b["log_prob"] for b in batch], dtype=torch.float32, device=self.device),
                    "advantages": torch.tensor(advantages, dtype=torch.float32, device=self.device)
                }

    def get_value_batches(self, batch_size: int) -> Generator[Dict, None, None]:
        indices = list(range(len(self.state_buffer)))
        random.shuffle(indices)

        for start_idx in range(0, len(self.state_buffer), batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            batch_decisions = [self.state_buffer[i] for i in batch_indices]

            features = torch.stack(
                [d["state"].features for d in batch_decisions])
            masks = torch.stack(
                [d["state"].station_mask for d in batch_decisions])

            yield {
                "states": StateBatch(
                    features,
                    masks
                ),
                "returns": torch.tensor([d["return"] for d in batch_decisions], dtype=torch.float32, device=self.device)
            }

    def clear(self):
        self.state_buffer.clear()
        self.action_buffer.clear()
