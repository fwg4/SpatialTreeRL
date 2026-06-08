# src/naive_baseline.py
import numpy as np
import logging

from env import MiniMetroEnv
from my_utils import format_time_ms
from runner import smdp_generator


logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_ring_actions(obs, active_lines):

    stations = obs["structured"]["stations"]
    positions = np.array([st["position"] for st in stations])
    cx, cy = positions.mean(axis=0)
    angles = np.arctan2(positions[:, 1] - cy, positions[:, 0] - cx)
    sorted_indices = np.argsort(angles).tolist()

    actions = []

    for path_index in range(active_lines):

        actions.append({
            "type": "create_path",
            "path_index": path_index,
            "stations": sorted_indices,
            "loop": True
        })

    return actions


def run_dumb_ring_baseline(dt_ms: int, seed: int = 42):

    env = MiniMetroEnv(dt_ms=dt_ms)

    gen = smdp_generator(env, seed=seed)

    try:
        event = next(gen)
        while True:
            msg_type = event["type"]

            # Decision Point
            if msg_type == "request_decision":
                actions = build_ring_actions(
                    event["obs"], event["active_lines"])
                event = gen.send(actions)

            # Terminal
            elif msg_type == "game_over":
                obs = event["obs"]
                final_time_ms = (obs["structured"]["time_ms"])
                final_score = (obs["structured"]["score"])

                return (final_time_ms, final_score, "OK")

            # Crash
            elif msg_type == "game_crash":
                reason = (event["info"].get("crash_reason"))
                return (0.0, 0, f"Crash: {reason}")

            # Defensive
            else:
                event = next(gen)

    except Exception as e:

        import traceback

        print(f"\n🚨 [Baseline Crash 攔截]\n{traceback.format_exc()}")

        return (0.0, 0, f"Crash: {type(e).__name__}")


def main():
    dt_test_cases = [2**i for i in range(8)]
    test_seeds = [11, 42, 99, 110, 255]

    print("\n" + "="*60)
    print("🚇 Mini Metro 'Dumb Ring' Baseline Test (Averaged)")
    print("="*60)
    print(f"{'dt_ms':>7} | {'Avg Time':>9} | {'Avg Score':>10} | {'Status'}")
    print("-" * 60)

    for dt in dt_test_cases:
        times = []
        scores = []
        crashes = 0

        for seed in test_seeds:
            time_s, score, status = run_dumb_ring_baseline(dt, seed)

            if status == "OK":
                times.append(time_s)
                scores.append(score)
            else:
                crashes += 1
                print(f"⚠️ dt={dt}, seed={seed} 發生崩潰: {status}")

        if times:
            avg_time = sum(times) / len(times)
            avg_score = sum(scores) / len(scores)
            status_msg = f"OK (Crashes: {crashes})" if crashes > 0 else "OK"

            print(
                f"{dt:>7d} | {format_time_ms(avg_time):>9} | {avg_score:>10.1f} | {status_msg}")
        else:
            print(f"{dt:>7d} | {'---':>9} | {'---':>10} | All Crashed")


if __name__ == "__main__":
    main()
