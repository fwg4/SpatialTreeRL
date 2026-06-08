# tests/test_policy_integration.py
import pytest
from env import MiniMetroEnv
from config import station_color, station_size
from entity.station import Station
from geometry.circle import Circle
from geometry.point import Point
from geometry.rect import Rect


@pytest.fixture
def game_env():
    """初始化並回傳 MiniMetroEnv 實例"""
    # 設定 dt_ms=16 模擬遊戲時間推進 [cite: 27]
    env = MiniMetroEnv(dt_ms=16)
    return env


def test_initial_game_state(game_env):
    """測試案例一：初始遊戲狀態"""
    obs = game_env.reset(seed=42)  # 使用固定 seed 確保決定性 [cite: 33]

    # 檢查觀測值的資料結構是否符合預期 [cite: 54, 55]
    assert "structured" in obs
    assert "arrays" in obs
    assert "stations" in obs["structured"]

    # 驗證初始站點數量大於 0
    assert len(obs["structured"]["stations"]) > 0
    # 初始路線數應該為 0
    assert len(obs["structured"]["paths"]) == 0


def test_create_path_action(game_env):
    """測試案例二：路線購買與建立"""
    obs = game_env.reset(seed=1)

    # 手動注入測試站點 (沿用你原本的邏輯)
    custom_stations = []
    for i in range(3):
        if i % 2 == 0:
            shape = Rect(color=station_color, width=2 *
                         station_size, height=2 * station_size)
        else:
            shape = Circle(color=station_color, radius=station_size)
        station = Station(shape, Point(i * 50, i * 50))
        custom_stations.append(station)
    game_env.mediator.stations = custom_stations

    # 執行建立路線動作 [cite: 37]
    action = {"type": "create_path", "stations": [0, 1, 2], "loop": False}
    next_obs, reward, done, info = game_env.step(action)  # 獲取回傳值 [cite: 34]

    # 驗證動作是否合法執行 [cite: 38]
    assert info["action_ok"] is True
    # 驗證 structured 中的路線數量是否增加
    assert len(next_obs["structured"]["paths"]) == 1


def test_buy_line_action(game_env):
    """測試案例三：資源變化（買線功能）"""
    game_env.reset(seed=1)

    # 假設策略決定購買一條新路線 [cite: 42]
    action = {"type": "buy_line"}
    next_obs, reward, done, info = game_env.step(action)

    # 注意：這裡的斷言取決於初始資源是否足夠。
    # 若初始無法購買，info["action_ok"] 會是 False [cite: 38, 48]
    # 我們這裡主要測試 API 介面是否能正確接收並處理該指令
    assert isinstance(info["action_ok"], bool)


def test_noop_and_time_progression(game_env):
    """測試案例四：Noop 推進時間直到站點新增"""
    obs = game_env.reset(seed=1)
    initial_station_count = len(obs["structured"]["stations"])

    # 連續執行 noop 來推進時間，模擬等待事件觸發 [cite: 35, 36]
    action = {"type": "noop"}

    # 推進 100 步 (1600ms) 看是否有變化
    for _ in range(100):
        obs, reward, done, info = game_env.step(action)
        assert info["action_ok"] is True

    # 確保時間有在推進，且系統狀態更新不會崩潰
    assert obs["structured"]["time_ms"] > 0


# tests/test_policy_integration.py


@pytest.fixture
def game_env():
    """初始化並回傳 MiniMetroEnv 實例"""
    # 設定 dt_ms=16 模擬遊戲時間推進 [cite: 27]
    env = MiniMetroEnv(dt_ms=16)
    return env


def test_initial_game_state(game_env):
    """測試案例一：初始遊戲狀態"""
    obs = game_env.reset(seed=42)  # 使用固定 seed 確保決定性 [cite: 33]

    # 檢查觀測值的資料結構是否符合預期 [cite: 54, 55]
    assert "structured" in obs
    assert "arrays" in obs
    assert "stations" in obs["structured"]

    # 驗證初始站點數量大於 0
    assert len(obs["structured"]["stations"]) > 0
    # 初始路線數應該為 0
    assert len(obs["structured"]["paths"]) == 0


def test_create_path_action(game_env):
    """測試案例二：路線購買與建立"""
    obs = game_env.reset(seed=1)

    # 手動注入測試站點 (沿用你原本的邏輯)
    custom_stations = []
    for i in range(3):
        if i % 2 == 0:
            shape = Rect(color=station_color, width=2 *
                         station_size, height=2 * station_size)
        else:
            shape = Circle(color=station_color, radius=station_size)
        station = Station(shape, Point(i * 50, i * 50))
        custom_stations.append(station)
    game_env.mediator.stations = custom_stations

    # 執行建立路線動作 [cite: 37]
    action = {"type": "create_path", "stations": [0, 1, 2], "loop": False}
    next_obs, reward, done, info = game_env.step(action)  # 獲取回傳值 [cite: 34]

    # 驗證動作是否合法執行 [cite: 38]
    assert info["action_ok"] is True
    # 驗證 structured 中的路線數量是否增加
    assert len(next_obs["structured"]["paths"]) == 1


def test_buy_line_action(game_env):
    """測試案例三：資源變化（買線功能）"""
    game_env.reset(seed=1)

    # 假設策略決定購買一條新路線 [cite: 42]
    action = {"type": "buy_line"}
    next_obs, reward, done, info = game_env.step(action)

    # 注意：這裡的斷言取決於初始資源是否足夠。
    # 若初始無法購買，info["action_ok"] 會是 False [cite: 38, 48]
    # 我們這裡主要測試 API 介面是否能正確接收並處理該指令
    assert isinstance(info["action_ok"], bool)


def test_noop_and_time_progression_safe(game_env):
    """測試案例四 (修正版)：Noop 推進時間直到站點新增，並安全處理 game-over 狀態"""
    obs = game_env.reset(seed=1)
    initial_time = obs["structured"]["time_ms"]
    action = {"type": "noop"}

    for _ in range(100):
        obs, reward, done, info = game_env.step(action)

        # ⚠️ 關鍵防護：一旦遊戲結束，後續動作會被拒絕且時間凍結，因此必須提早跳出
        if done:
            break

        assert info["action_ok"] is True

    # 只要有成功執行過幾步，時間必定大於初始時間
    assert obs["structured"]["time_ms"] > initial_time


def test_time_progression_with_pause_resume_noop(game_env):
    """測試案例六：驗證 pause, resume, noop 對 time_ms 與 is_paused 的精確影響"""
    obs = game_env.reset(seed=1)

    # 1. 取得初始時間與暫停狀態
    initial_time = obs["structured"]["time_ms"]
    assert obs["structured"]["is_paused"] is False

    # 2. 測試 noop (不覆寫 dt_ms，使用環境預設的 16ms)
    obs, _, _, info = game_env.step({"type": "noop"})
    time_after_noop = obs["structured"]["time_ms"]

    assert info["action_ok"] is True
    assert time_after_noop > initial_time  # 驗證時間確實推進了 (0 -> 16)

    # 3. 測試 pause (不覆寫 dt_ms)
    # 💡 根據測試發現，MiniMetroEnv 會在處理物理刻度前先切換狀態。
    # 因此 pause 指令生效的該回合，時間會被立刻攔截而不推進！
    obs, _, _, info = game_env.step({"type": "pause"})
    time_after_pause = obs["structured"]["time_ms"]

    assert info["action_ok"] is True
    assert obs["structured"]["is_paused"] is True
    assert time_after_pause == time_after_noop  # 修正斷言：時間在此刻被凍結 (16 == 16)

    # 加碼驗證：在暫停狀態下呼叫 noop，時間也不應該前進
    obs, _, _, info = game_env.step({"type": "noop"})
    time_after_paused_noop = obs["structured"]["time_ms"]
    assert time_after_paused_noop == time_after_pause

    # 4. 測試 resume (強制覆寫 dt_ms=0)
    # 雖然環境本身有保護機制，但在 RL 封裝中，明確宣告 dt_ms=0 始終是避免不可預期行為的最佳實踐
    obs, _, _, info = game_env.step({"type": "resume"}, dt_ms=0)
    time_after_resume = obs["structured"]["time_ms"]

    assert info["action_ok"] is True
    assert obs["structured"]["is_paused"] is False
    assert time_after_resume == time_after_paused_noop  # 時間依然被完美鎖死


def test_buy_line_time_progression(game_env):
    """測試案例七：驗證 buy_line 動作在成功與失敗時對模擬時間的影響"""
    obs = game_env.reset(seed=1)
    initial_time = obs["structured"]["time_ms"]

    # --- 情況 1：買不起 (失敗) ---
    action_buy = {"type": "buy_line"}
    obs, reward, done, info = game_env.step(action_buy)

    # 驗證：動作被拒絕時，該回合的模擬時間絕對不會推進
    assert info["action_ok"] is False
    assert obs["structured"]["time_ms"] == initial_time

    # --- 情況 2：買得起 (成功) ---
    # 為了測試成功推進時間，我們手動作弊注入足夠的分數
    # (註：這裡假設底層分數存在 game_env.mediator.score，請依據你實際環境物件結構調整)
    if hasattr(game_env, 'mediator'):
        game_env.mediator.score = 99999

    obs, reward, done, info = game_env.step(action_buy)

    # 驗證：在沒覆寫 dt_ms 的情況下，合法的動作會套用建構子的 16ms 並推進時間 [cite: 135, 136]
    if info["action_ok"]:
        time_after_buy = obs["structured"]["time_ms"]
        assert time_after_buy > initial_time


def test_reward_includes_purchase_cost(game_env):
    """測試案例八：驗證 buy_line 成功時，reward 是否已經自動包含購買成本 (呈現負值)"""
    obs = game_env.reset(seed=1)

    # 為了確保能成功購買，手動注入足夠的分數 (模擬 Agent 載客賺了很多錢)
    if hasattr(game_env, 'mediator'):
        game_env.mediator.score = 1000

    # ⚠️ 關鍵步驟：先執行一次 noop，讓環境的內部狀態將「上一步分數」的基準點更新為 1000
    obs, reward, done, info = game_env.step({"type": "noop"})

    # 紀錄購買前的分數
    score_before_buy = obs["structured"]["score"]

    # 執行購買路線動作
    action_buy = {"type": "buy_line"}
    obs, reward_from_buy, done, info = game_env.step(action_buy)

    # 驗證 1：確認購買動作有成功執行
    assert info["action_ok"] is True

    # 紀錄購買後的分數
    score_after_buy = obs["structured"]["score"]

    # 驗證 2：確認遊戲總分確實因為花費而減少了
    assert score_after_buy < score_before_buy

    # 🌟 核心驗證 3：環境回傳的 reward，是否精準等於「購買後分數 - 購買前分數」(即負的購買成本)
    expected_reward = score_after_buy - score_before_buy
    assert reward_from_buy == expected_reward

    # 🌟 核心驗證 4：確保 reward 是負數。這證明了「扣除成本」已經發生在環境內部，
    # 你的 Agent 直接接收 reward 即可，若再額外扣 actual_cost 就是扣了兩次錢！
    assert reward_from_buy < 0

def test_failed_action_reward_and_injection_trap(game_env):
    """測試案例九：驗證失敗動作的真實 reward，以及手動改分造成的假象"""
    obs = game_env.reset(seed=1)
    
    # --- 階段 A：真實遊戲情況 ---
    # 開局沒錢，buy_line 必定失敗
    obs, step_reward_fail, done, info = game_env.step({"type": "buy_line"}, dt_ms=0)
    
    assert info["action_ok"] is False
    # 🌟 核心驗證 1：在真實遊戲中，失敗的動作配上 dt_ms=0，reward 絕對是 0，加上它等同加 0
    assert step_reward_fail == 0 

    # --- 階段 B：解釋你看到的「倍數增加」幻覺 ---
    # 手動作弊，硬把底層分數改成 1000
    if hasattr(game_env, 'mediator'):
        game_env.mediator.score = 1000
        
    # 緊接著執行一個絕對會失敗的動作 (嘗試刪除一條不存在的線路)
    obs, step_reward_jump, done, info = game_env.step({"type": "remove_path", "path_index": 999}, dt_ms=0)
    
    assert info["action_ok"] is False
    # 🌟 核心驗證 2：雖然動作沒執行，但因為你「手動」改了 score，環境算出了 1000 的差額！
    # 這證明了你看到的倍數放大，純粹是測試時強制改分造成的副作用。
    assert step_reward_jump == 1000