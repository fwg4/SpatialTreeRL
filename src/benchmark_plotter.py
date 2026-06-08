# src/benchmark_plotter.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("\n" + "="*60)
    print("📊 [階段 2] 產出極簡高層次簡報圖表 (Presentation Ready)")
    print("="*60)

    csv_path = "outputs/benchmark_raw_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到資料檔 {csv_path}！請先執行 benchmark_collector.py")
        return

    # 讀取完整的原始資料
    df = pd.read_csv(csv_path)

    # ---------------------------------------------------------
    # 1. 名稱精簡化 (Engineering -> Presentation)
    # ---------------------------------------------------------
    name_mapping = {
        "Baseline": "Ring",
        "Ep-1000": "1k",
        "Ep-2000": "2k",
        "Ep-3000": "3k",
        "Ep-4000": "4k",
        "Ep-5000": "5k"
    }
    df["Model"] = df["Model"].map(name_mapping)
    
    # 確保順序正確
    model_order = ["Ring", "1k", "2k", "3k", "4k", "5k"]

    # ---------------------------------------------------------
    # 2. 計算關鍵指標 (Baseline vs 5k)
    # ---------------------------------------------------------
    # 計算每個模型的平均存活時間
    summary = df.groupby("Model")["Survival Time (s)"].mean()
    
    baseline_time = summary.get("Ring", 0)
    best_time = summary.get("5k", 0)
    
    improvement_ratio = (best_time / baseline_time) if baseline_time > 0 else 0
    improvement_pct = ((best_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0

    print("\n📝 【簡報核心數據 (Core Metrics)】")
    print("-" * 50)
    print(f"🔹 Baseline (Ring): {baseline_time:.1f} 秒")
    print(f"🔹 最終模型 (5k)  : {best_time:.1f} 秒")
    print(f"🚀 效能提升倍率  : {improvement_ratio:.1f}x (+{improvement_pct:.0f}%)")
    print("-" * 50)
    print(f"💡 簡報金句建議 (請直接複製貼到投影片上)：")
    print(f"\"SpatialTreeRL achieved {improvement_ratio:.1f}x longer survival than the Dumb Ring baseline under identical seeds.\"")
    print("-" * 50)

    # ==========================================
    # 3. 繪製極簡簡報圖表 (Bar Plot + 95% CI)
    # ==========================================
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    # 簡報通常使用無襯線字體(sans-serif)會比論文的襯線字體(serif)在投影幕上更清晰
    plt.rcParams['font.family'] = 'sans-serif' 

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # 視覺引導設計：Baseline 用灰色低調處理，SpatialTreeRL 用專業藍色強調
    palette = ["#bdc3c7"] + ["#2980b9"] * 5
    
    sns.barplot(
        data=df, 
        x="Model", 
        y="Survival Time (s)", 
        order=model_order,
        errorbar=("ci", 95), # 使用 95% 信心區間
        capsize=0.15,
        palette=palette,
        hue="Model",
        legend=False,
        alpha=0.9,
        ax=ax
    )
    
    ax.set_title("SpatialTreeRL Performance", fontweight="bold", pad=20)
    ax.set_xlabel("", fontweight="bold")
    ax.set_ylabel("Average Survival Time (s)", fontweight="bold", labelpad=15)
    
    # 移除上方與右方邊框，讓視覺更透氣
    sns.despine(left=True, bottom=False)
    
    # 調整網格線，讓背景不要干擾數據柱體
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    
    output_img = "outputs/benchmark_presentation.png"
    fig.savefig(output_img, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ 簡報圖表已成功輸出：{output_img}")

if __name__ == "__main__":
    main()