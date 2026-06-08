# src/timelapse.py
import os
import torch
import imageio
import logging
import argparse
from tqdm import tqdm

from env import MiniMetroEnv
from ga import DummyGARegistry
from runner import process_generator
from my_utils import set_global_seed
from visualizer import MetroVisualizer

from state_encoder import StateEncoder
from policy import MetroTreePolicy_GATopo
from critic import MetroTreeCritic
from decision_maker import GATopoDecisionMaker
from agent import MetroTreeAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def export_timelapse(agent, env, speedup=10.0, output_prefix="demo", seed=42):
    logger.info(f"啟動視覺化雙流管線 (倍速: {speedup}x)...")
    
    video_fps = 30.0
    capture_interval_ms = (1.0 / video_fps) * speedup * 1000.0
    last_capture_time_ms = -capture_interval_ms
    
    frames_saved = 0
    step_count = 0
    output_prefix += f"_{int(speedup)}x"
    
    # 確保 outputs 目錄存在
    os.makedirs("outputs", exist_ok=True)
    main_video_path = f"outputs/{output_prefix}_main.mp4"
    
    writer_main = imageio.get_writer(main_video_path, fps=video_fps, macro_block_size=None)
    last_main_frame = None
    last_decisions = None

    pbar = tqdm(desc="模擬影像擷取中", unit=" 幀", dynamic_ncols=True)
    gen = process_generator(env, seed=seed)
    
    try:
        event = next(gen)
        while True:
            msg_type = event["type"]
            obs = event.get("obs")
            
            if obs:
                current_time_ms = obs.get("structured", {}).get("time_ms", 0)
                if (current_time_ms - last_capture_time_ms) >= capture_interval_ms:
                    last_main_frame = MetroVisualizer.render_frame(
                        obs=obs, decisions=last_decisions
                    )
                    writer_main.append_data(last_main_frame)
                    last_capture_time_ms = current_time_ms
                    frames_saved += 1
                    
                    pbar.update(1)
                    pbar.set_postfix(game_time=f"{current_time_ms / 1000:.1f}s")
                    last_decisions = None 

            if msg_type == "request_decision":
                with torch.no_grad():
                    result = agent.act(obs, event["active_lines"])
                last_decisions = result["atomic_decisions"]
                event = gen.send(result["env_actions"])
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

    logger.info(f"總共收集了 {frames_saved} 幀。 (決策與移動總步數: {step_count})")
    logger.info(f"✅ 縮時影片已成功匯出至: {main_video_path}")


def main():
    # --- 加入命令列參數解析 ---
    parser = argparse.ArgumentParser(description="產生經過訓練的模型之遊戲縮時影片")
    parser.add_argument("--model", type=str, default="checkpoints/policy_best.pth",
                        help="模型的權重路徑 (預設: checkpoints/policy_best.pth)")
    parser.add_argument("--speedup", type=float, default=15.0,
                        help="影片倍速 (預設: 15.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="環境隨機參數種子 (預設: 42)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"執行環境: {device}")
    
    set_global_seed(args.seed)

    # 動態萃取輸出前綴名稱 (例如: policy_ep_3400.pth -> policy_ep_3400)
    model_basename = os.path.splitext(os.path.basename(args.model))[0]

    SAVE_DIR = "checkpoints"
    ga_path = os.path.join(SAVE_DIR, "ga.json")

    if os.path.exists(ga_path):
        ga_registry = DummyGARegistry.load_from_file(ga_path)
    else:
        ga_registry = DummyGARegistry()

    env = MiniMetroEnv(dt_ms=16)
    
    max_node_id = max(ga_registry.node_configs.keys()) if ga_registry.node_configs else 0
    max_leaf_id = max(ga_registry.leaf_configs.keys()) if ga_registry.leaf_configs else 0
    tensor_capacity = max(max_node_id, max_leaf_id) + 1

    encoder = StateEncoder(device=device)
    policy = MetroTreePolicy_GATopo(num_nodes=tensor_capacity, num_leaves=tensor_capacity).to(device)
    critic = MetroTreeCritic(feature_dim=StateEncoder.FEATURE_DIM).to(device)
    decision_maker = GATopoDecisionMaker(ga_registry, device=device)
    
    agent = MetroTreeAgent(encoder, policy, critic, decision_maker).to(device)

    # 根據輸入的路徑載入模型
    if os.path.exists(args.model):
        logger.info(f"🧠 讀取指定神經網路權重: {args.model}")
        agent.policy.load_state_dict(torch.load(args.model, map_location=device))
    else:
        # 如果使用者只輸入 policy_ep_3400.pth，自動去 checkpoints/ 下找找看
        fallback_path = os.path.join(SAVE_DIR, args.model)
        if os.path.exists(fallback_path):
            logger.info(f"🧠 讀取指定神經網路權重: {fallback_path}")
            agent.policy.load_state_dict(torch.load(fallback_path, map_location=device))
        else:
            logger.warning(f"⚠️ 找不到 {args.model}，模型將使用隨機初始權重！")

    agent.eval()

    export_timelapse(
        agent=agent,
        env=env,
        speedup=args.speedup,
        output_prefix=model_basename,
        seed=args.seed
    )
    
if __name__ == "__main__":
    main()