import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from geometry.type import ShapeType
from my_utils import format_time_ms
import torch
import torch.nn.functional as F


class MetroVisualizer:
    _fig = None
    _ax = None
    _id_cache = {}
    _last_decisions = None

    _MARKERS = {
        ShapeType.CIRCLE: "o", ShapeType.TRIANGLE: "^", ShapeType.RECT: "s",
        ShapeType.CROSS: "P", ShapeType.STAR: "*", ShapeType.DIAMOND: "D",
        ShapeType.PENTAGON: "p"
    }

    @classmethod
    def render_frame(cls, obs: dict, decisions: list = None) -> np.ndarray:
        if cls._fig is None:
            cls._fig, cls._ax = plt.subplots(figsize=(16, 9), dpi=120)
            cls._fig.tight_layout(pad=0)

        ax = cls._ax
        ax.clear()

        arrays = obs.get("arrays", {})
        structured = obs.get("structured", {})
        positions = arrays.get("station_positions", [])

        # ==========================================
        # 1. 乘客佇列映射 (極簡化)
        # ==========================================
        pax_dests = arrays.get("passenger_destination_types", [])
        pax_st_idx = arrays.get("passenger_station_indices", [])
        station_pax = defaultdict(list)

        for dest, st_idx in zip(pax_dests, pax_st_idx):
            try:
                # 把 2.0 轉成 '2' 再轉成 ShapeType.CIRCLE
                shape = ShapeType(str(int(dest)))
            except ValueError:
                shape = ShapeType.CIRCLE
            station_pax[int(st_idx)].append(shape)

        # ==========================================
        # 2. 實體車站 & 等待乘客
        # ==========================================
        for i, st in enumerate(structured.get("stations", [])):
            cls._id_cache[st.get("id", "")] = i
            if i >= len(positions):
                continue

            x, y = positions[i, 0], positions[i, 1]

            # [核心簡化] 直接抓取 Enum 實體，防呆預設給圓形
            shape = st.get("shape_type", ShapeType.CIRCLE)

            ax.scatter(x, y, facecolors='white', edgecolors='black',
                       linewidths=2.5, marker=cls._MARKERS.get(shape, "o"), s=250, zorder=4)

            # 繪製乘客
            if pax := station_pax.get(i, []):
                for p_idx, p_shape in enumerate(pax):
                    ax.scatter(x + 35 + (p_idx * 20), y, facecolors='black', edgecolors='none',
                               marker=cls._MARKERS.get(p_shape, "o"), s=50, zorder=5)
            elif (count := st.get("passenger_count", 0)) > 0:
                ax.text(x + 35, y, f"+{count}",
                        fontweight='bold', va='center', zorder=5)

        # ==========================================
        # 3. 路線拓撲
        # ==========================================
        # line_colors = [tuple(c / 255.0 for c in p.get("color", (100, 100, 100))) for p in structured.get("paths", [])]
        line_colors = [
            (0.13, 0.59, 0.95),  # 1. 天藍
            (0.96, 0.26, 0.21),  # 2. 櫻紅
            (0.98, 0.75, 0.18),  # 3. 亮黃
            (0.30, 0.69, 0.31),  # 4. 草綠
            (0.61, 0.15, 0.69),  # 5. 皇紫 (備用)
            (1.00, 0.60, 0.00),  # 6. 鮮橘 (備用)
        ]
        for path_idx, path in enumerate(structured.get("paths", [])):
            valid_indices = []
            for sid in path.get("station_ids", []):
                # 簡化模糊搜尋，提高可讀性
                idx = cls._id_cache.get(sid)
                if idx is None:
                    idx = next(
                        (v for k, v in cls._id_cache.items() if sid in k), None)

                if idx is not None and idx < len(positions):
                    valid_indices.append(idx)

            if len(valid_indices) >= 2:
                if path.get("is_looped", False):
                    valid_indices.append(valid_indices[0])
                pts = positions[valid_indices]
                ax.plot(pts[:, 0], pts[:, 1], linewidth=6,
                        alpha=0.8, color=line_colors[path_idx], zorder=1)

        # ==========================================
        # 4. 動態列車
        # ==========================================
        m_path_indices = arrays.get("metro_path_indices", [])
        for m_idx, metro in enumerate(structured.get("metros", [])):
            if not (m_pos := metro.get("position")):
                continue
            mx, my = m_pos[0], m_pos[1]

            path_idx = int(m_path_indices[m_idx]) if m_idx < len(
                m_path_indices) else 0
            m_color = line_colors[path_idx] if path_idx < len(
                line_colors) else (0.4, 0.4, 0.4)

            ax.scatter(mx, my, facecolors=m_color, edgecolors='white',
                       marker='s', s=250, linewidths=2.5, zorder=7)

            m_pax = len(metro.get("passenger_ids", [])
                        ) or metro.get("passenger_count", 0)
            if m_pax > 0:
                ax.text(mx, my, str(m_pax), color='white', fontsize=9,
                        fontweight='bold', ha='center', va='center', zorder=8)
        # ==========================================
        # 5. 抽象模型意圖 (統一解析)
        # ==========================================
        if decisions is not None:
            cls._last_decisions = decisions

        if cls._last_decisions:
            intents = defaultdict(dict)

            for d in cls._last_decisions:
                if d["type"].startswith("leaf_"):
                    axis, l_idx = d["type"].split("_")[1:]
                    intents[int(l_idx)][axis] = d["sample"]

            for l_idx, coords in intents.items():
                if 'x' in coords and 'y' in coords:
                    px = np.clip(((coords['x'] + 1.0) / 2.0) * 1920, 0, 1920)
                    py = np.clip(((coords['y'] + 1.0) / 2.0) * 1080, 0, 1080)
                    color = line_colors[l_idx] if l_idx < len(
                        line_colors) else 'red'
                    ax.scatter(px, py, facecolors=color, edgecolors='black',
                               marker='X', s=150, linewidths=2, zorder=6)

        # ==========================================
        # 6. HUD 資訊與畫面輸出
        # ==========================================
        time_ms = structured.get("time_ms", 0)
        time_str = format_time_ms(time_ms)
        ax.text(1880, 40, f"Time: {time_str}\nScore: {structured.get('score', 0)}",
                color='white', fontsize=16, fontweight='bold', ha='right', va='top', zorder=10,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))

        ax.set(xlim=(0, 1920), ylim=(1080, 0))
        ax.axis('off')

        cls._fig.canvas.draw()
        return np.array(cls._fig.canvas.buffer_rgba())[:, :, :3].copy()
