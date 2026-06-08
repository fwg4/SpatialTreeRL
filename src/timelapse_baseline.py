# src/timelapse_baseline.py
import os
import imageio
import logging
import argparse
from tqdm import tqdm

from env import MiniMetroEnv
from runner import process_generator
from my_utils import set_global_seed
from visualizer import MetroVisualizer

# 引入你的 baseline 決策邏輯
from naive_baseline import build_ring_actions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def export_baseline_timelapse(env, speedup=10.0, seed=42):
    logger.info(f"啟動 Baseline (Dumb Ring) 視覺化管線 (倍速: {speedup}x)...")
    
    video_fps = 30.0
    capture_interval_ms = (1.0 / video_fps) * speedup * 1000.0
    last_capture_time_ms = -capture_interval_ms
    
    frames_saved = 0
    step_count = 0
    
    os.makedirs("outputs", exist_ok=True)
    # 檔案名稱標示為 baseline
    main_video_path = f"outputs/baseline_dumb_ring_{int(speedup)}x_main.mp4"
    
    writer_main = imageio.get_writer(main_video_path, fps=video_fps, macro_block_size=None)
    last_main_frame = None

    pbar = tqdm(desc="模擬影像擷取中", unit=" 幀", dynamic_ncols=True)
    gen = process_generator(env, seed=seed)
    
    try:
        event = next(gen)
        while True:
            msg_type = event["type"]
            obs = event.get("obs")
            
            # --- 1. 畫面渲染邏輯 ---
            if obs:
                current_time_ms = obs.get("structured", {}).get("time_ms", 0)
                if (current_time_ms - last_capture_time_ms) >= capture_interval_ms:
                    # Baseline 沒有意圖座標 (decisions)，直接傳 None
                    last_main_frame = MetroVisualizer.render_frame(
                        obs=obs, decisions=None
                    )
                    writer_main.append_data(last_main_frame)
                    last_capture_time_ms = current_time_ms
                    frames_saved += 1
                    
                    pbar.update(1)
                    pbar.set_postfix(game_time=f"{current_time_ms / 1000:.1f}s")

            # --- 2. 事件狀態機邏輯 ---
            if msg_type == "request_decision":
                # 🌟 直接呼叫 Dumb Ring 的寫死邏輯
                actions = build_ring_actions(obs, event["active_lines"])
                event = gen.send(actions)
                step_count += 1
                
            elif msg_type in ("step", "decision_result"):
                event = next(gen)
                step_count += 1
                
            elif msg_type in ("game_over", "game_crash"):
                crash_reason = event.get("info", {}).get("crash_reason", "未知原因")
                pbar.write(f"\n💀 遊戲結束 (狀態: {msg_type} | 原因: {crash_reason})")
                
                if obs:
                    stations = obs.get("structured", {}).get("stations", [])
                    if stations:
                        sorted_stations = sorted(stations, key=lambda s: s.get("passenger_count", 0), reverse=True)
                        pbar.write("\n💀 [死亡瞬間診斷] 嫌疑最大的車站：")
                        for i, st in enumerate(sorted_stations[:3]):
                            raw_shape = str(st.get("shape_type", "未知"))
                            clean_shape = raw_shape.split('.')[-1].split(':')[0] if '.' in raw_shape else raw_shape
                            pos = st.get("position", (0.0, 0.0))
                            p_count = st.get("passenger_count", 0)
                            pbar.write(f"  Top {i+1}: {clean_shape} 車站 (座標: {pos[0]:.1f}, {pos[1]:.1f}) - 堆積了 {p_count} 名乘客")

                if last_main_frame is not None:
                    pbar.write("延遲停留 3 秒以供觀察最終狀態...")
                    for _ in range(int(video_fps * 3)):
                        writer_main.append_data(last_main_frame)
                break
                
            else:
                event = next(gen)

    except StopIteration:
        pass
    except Exception as e:
        logger.error(f"錄影過程發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        writer_main.close()

    if frames_saved == 0:
        logger.error("未收集到任何影像幀，錄影失敗。")
        return

    logger.info(f"總共收集了 {frames_saved} 幀。 (總步數: {step_count})")
    logger.info(f"✅ Baseline 縮時影片已成功匯出至: {main_video_path}")


def main():
    parser = argparse.ArgumentParser(description="產生 Baseline (Dumb Ring) 的遊戲縮時影片")
    parser.add_argument("--speedup", type=float, default=15.0, help="影片倍速 (預設: 15.0)")
    parser.add_argument("--seed", type=int, default=42, help="環境隨機參數種子 (預設: 42)")
    args = parser.parse_args()

    set_global_seed(args.seed)
    
    # 保持與訓練環境相同的 dt_ms
    env = MiniMetroEnv(dt_ms=16)

    export_baseline_timelapse(
        env=env,
        speedup=args.speedup,
        seed=args.seed
    )
    
if __name__ == "__main__":
    main()