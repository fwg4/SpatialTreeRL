# src/runner.py

def game_generator(env, seed=None):
    action = yield {"type": "reset", "obs": env.reset(seed=seed)}

    while True:
        try:
            obs, reward, done, info = env.step(action)

            if done:
                yield {
                    "type": "game_over",
                    "obs": obs,
                    "reward": reward,
                    "info": info
                }
                return

            action = yield {
                "type": "step",
                "obs": obs,
                "reward": reward,
                "info": info
            }

        except Exception as e:

            import traceback
            error_trace = traceback.format_exc()

            yield {
                "type": "game_crash",
                "info": {"crash_reason": str(e), "traceback": error_trace}
            }
            return


def process_generator(env, seed=None):
    game_gen = game_generator(env, seed)

    event = next(game_gen)

    if event["type"] != "reset":
        raise RuntimeError("game_generator 必須先產生 reset")

    obs = event["obs"]

    last_station_count = 0
    current_lines_count = 1
    last_lines_count = 0

    while True:

        current_station_count = len(obs["structured"]["stations"])

        if (current_station_count > last_station_count or current_lines_count > last_lines_count):

            last_station_count = current_station_count

            # 清空舊路線
            for i in range(last_lines_count):
                event = game_gen.send(
                    {"type": "remove_path", "path_index": i})

                if event["type"] != "step":
                    yield event
                    return

                obs = event["obs"]

            # 向 Agent 要決策
            env_actions = yield {"type": "request_decision", "obs": obs, "active_lines": current_lines_count}
            last_lines_count = current_lines_count

            # 執行 Agent 決策
            for i in range(current_lines_count):

                event = game_gen.send(env_actions[i])

                if event["type"] != "step":
                    yield event
                    return

                obs = event["obs"]

                yield {
                    "type": "decision_result",
                    "obs": obs,
                    "reward": event["reward"],
                    "info": event["info"]
                }

        else:

            # 嘗試買線
            event = game_gen.send({
                "type": "buy_line"
            })

            if event["type"] != "step":
                yield event
                return

            obs = event["obs"]

            if event["info"].get("action_ok", False):

                current_lines_count += 1

                yield {
                    "type": "step",
                    "obs": obs,
                    "reward": event["reward"],
                    "info": event["info"]
                }

            else:

                # noop
                event = game_gen.send({
                    "type": "noop"
                })

                if event["type"] != "step":
                    yield event
                    return

                obs = event["obs"]

                yield {
                    "type": "step",
                    "obs": obs,
                    "reward": event["reward"],
                    "info": event["info"]
                }


def smdp_generator(env, seed=None):

    base_gen = process_generator(env, seed)

    accumulated_reward = 0.0

    try:

        event = next(base_gen)

        while True:

            msg_type = event["type"]

            # Agent Decision Point
            if msg_type == "request_decision":

                event["smdp_reward"] = accumulated_reward
                action_bundle = yield event
                accumulated_reward = 0.0
                event = base_gen.send(action_bundle)

            # Micro Steps
            elif msg_type in ("step", "decision_result"):

                accumulated_reward += event.get("reward", 0.0)

                event = next(base_gen)
            # Terminal
            elif msg_type in ("game_over", "game_crash"):

                event["smdp_reward"] = (
                    accumulated_reward + event.get("reward", 0.0))
                yield event
                return
            # Unknown Event
            else:

                event = next(base_gen)

    except StopIteration:
        return
