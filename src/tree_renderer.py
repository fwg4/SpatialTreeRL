import matplotlib.pyplot as plt
import numpy as np

class TreeRenderer:
    _fig = None
    _ax = None

    @classmethod
    def _init_canvas(cls):
        if cls._fig is None:
            cls._fig, cls._ax = plt.subplots(figsize=(16, 9), dpi=120)
            cls._fig.tight_layout(pad=1.0)
            cls._fig.patch.set_facecolor('#1e1e1e')
        cls._ax.clear()
        cls._ax.set_facecolor('#1e1e1e')
        cls._ax.axis('off')

    @classmethod
    def _draw(cls, tree_data: dict):
        """根據解析好的字典資料進行渲染"""
        cls._init_canvas()
        ax = cls._ax

        # 1. 繪製邊緣 (Edges)
        node_pos = {n["id"]: (n["x"], n["y"]) for n in tree_data["nodes"]}
        for edge in tree_data["edges"]:
            px, py = node_pos[edge["parent"]]
            cx, cy = node_pos[edge["child"]]
            
            color = "#ff9800" if edge["is_active"] else "#424242"
            lw = 4 if edge["is_active"] else 1.5
            alpha = 1.0 if edge["is_active"] else 0.4
            ax.plot([px, cx], [py, cy], color=color, linewidth=lw, alpha=alpha, zorder=1)

        # 2. 繪製節點與文字 (Nodes)
        x_vals, y_vals, colors, sizes = [], [], [], []
        for node in tree_data["nodes"]:
            x_vals.append(node["x"])
            y_vals.append(node["y"])
            
            colors.append(("#4caf50" if node["is_leaf"] else "#2196f3") if node["is_active"] else "#616161")
            sizes.append(250 if node["is_active"] else 80)

            ax.text(node["x"], node["y"] - 0.25, node["text"], color='white', fontsize=8,
                    ha='center', va='top', zorder=3,
                    bbox=dict(facecolor='#000000', alpha=0.8, edgecolor='#ffffff' if node["is_active"] else 'none', pad=3.0))

        ax.scatter(x_vals, y_vals, c=colors, s=sizes, edgecolors='white', linewidths=1.5, zorder=2)
        ax.text(0.02, 0.95, "Decision Tree Architecture (MFA / MFT View)", transform=ax.transAxes, color='white', fontsize=16, fontweight='bold')

    @classmethod
    def get_frame(cls, tree_data: dict) -> np.ndarray:
        """[錄影專用] 回傳 numpy array"""
        cls._draw(tree_data)
        cls._fig.canvas.draw()
        return np.array(cls._fig.canvas.buffer_rgba())[:, :, :3].copy()

    @classmethod
    def export_image(cls, tree_data: dict, filepath: str):
        """[靜態匯出專用] 儲存為圖片檔"""
        cls._draw(tree_data)
        cls._fig.savefig(filepath, facecolor=cls._fig.get_facecolor(), bbox_inches='tight')
        print(f"✅ 靜態決策樹架構圖已儲存至: {filepath}")