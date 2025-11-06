import json
from pathlib import Path

DATA_DIR = Path("data/referee_conspiracy")
BEAST_PATH = DATA_DIR / "hrm_predictions.json"
SNIPER_PATH = DATA_DIR / "sniper_predictions.json"
CONSPIRACY_PATH = DATA_DIR / "conspiracy_predictions.json"
TASK_PATH = DATA_DIR / "hrm_tasks.json"
OUTPUT_PATH = DATA_DIR / "fused_hr_predictions.json"


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def heavy_weather(task):
    weather = task.get("weather", {})
    rain = float(weather.get("rain", 0.0))
    blackout = bool(task.get("blackout", False))
    return rain >= 0.45 or blackout


def extract_vote(name, payload):
    if not payload:
        return {
            "h_module": "NORMAL_FLOW",
            "l_module": {"under_prob": 0.5, "flag_spike": 1.0},
            "confidence": 0.5,
            "asshole_ref_detected": False,
            "flag_spike_series": {},
            "ref_bias_therapy_needed": 0.0,
        }
    entry = payload if isinstance(payload, dict) else {}
    return {
        "h_module": entry.get("h_module", "NORMAL_FLOW"),
        "l_module": entry.get("l_module", {"under_prob": 0.5, "flag_spike": 1.0}),
        "confidence": float(entry.get("confidence", 0.5)),
        "asshole_ref_detected": bool(entry.get("asshole_ref_detected", False)),
        "flag_spike_series": entry.get("flag_spike_series", {}),
        "ref_bias_therapy_needed": float(entry.get("ref_bias_therapy_needed", 0.0)),
        "posterior_dominant": entry.get("posterior_dominant"),
        "posterior_slice": entry.get("posterior_slice"),
        "signals": entry.get("signals"),
        "source": name,
    }


def fight_for_script(game_id, task, beast_vote, sniper_vote, conspiracy_vote):
    arena = [("BEAST", beast_vote), ("SNIPER", sniper_vote), ("CONSPIRACY", conspiracy_vote)]
    battle_log = []
    for name, vote in arena:
        battle_score = vote["confidence"] * (1.0 + 0.3 * float(vote["l_module"].get("flag_spike", 1.0)))
        battle_log.append({"source": name, "label": vote["h_module"], "score": battle_score})
    battle_log.sort(key=lambda item: item["score"], reverse=True)
    beast_conf = beast_vote["confidence"]
    sniper_conf = sniper_vote["confidence"]
    winner = battle_log[0]["source"]
    final_label = battle_log[0]["label"]
    final_under = float(arena[[s for s, _ in arena].index(winner)][1]["l_module"].get("under_prob", 0.5))
    flag_spike = float(arena[[s for s, _ in arena].index(winner)][1]["l_module"].get("flag_spike", 1.0))
    asshole = any(v["asshole_ref_detected"] for _, v in arena if v["asshole_ref_detected"])
    therapy = max(v["ref_bias_therapy_needed"] for _, v in arena)
    override_applied = False
    if beast_conf > 0.8 and sniper_conf > 0.8 and heavy_weather(task):
        final_label = "SCRIPTED_BLACKOUT"
        weather_weight = min(1.0, float(task.get("script_pressure", 0.5)) + 0.4)
        final_under = min(0.999, beast_vote["l_module"]["under_prob"] * 0.5 + sniper_vote["l_module"]["under_prob"] * 0.5 + 0.2 * weather_weight)
        flag_spike = max(flag_spike, 1.7)
        winner = "WEATHER_OVERLORD"
        override_applied = True
    if conspiracy_vote["confidence"] > 0.78 and conspiracy_vote["l_module"]["under_prob"] > final_under:
        final_under = conspiracy_vote["l_module"]["under_prob"]
        flag_spike = max(flag_spike, conspiracy_vote["l_module"]["flag_spike"])
    return {
        "game_id": game_id,
        "h_module": final_label,
        "under_prob": final_under,
        "flag_spike": flag_spike,
        "winner": winner,
        "battle_log": battle_log,
        "override_applied": override_applied,
        "asshole_ref_detected": asshole,
        "ref_bias_therapy_needed": therapy,
        "beast_meta": {
            "posterior_dominant": beast_vote.get("posterior_dominant"),
            "posterior_slice": beast_vote.get("posterior_slice"),
        },
        "signals": conspiracy_vote.get("signals", {}),
        "flag_spike_series": {
            "beast": beast_vote.get("flag_spike_series", {}),
            "sniper": sniper_vote.get("flag_spike_series", {}),
            "conspiracy": conspiracy_vote.get("flag_spike_series", {}),
        },
    }


def fuse_predictions():
    beast = read_json(BEAST_PATH)
    sniper = read_json(SNIPER_PATH)
    conspiracy = read_json(CONSPIRACY_PATH)
    tasks = read_json(TASK_PATH)
    final = {}
    all_keys = set(beast.keys()) | set(sniper.keys()) | set(conspiracy.keys())
    for game_id in all_keys:
        task = tasks.get(game_id, {})
        beast_vote = extract_vote("BEAST", beast.get(game_id))
        sniper_vote = extract_vote("SNIPER", sniper.get(game_id))
        conspiracy_vote = extract_vote("CONSPIRACY", conspiracy.get(game_id))
        fused = fight_for_script(game_id, task, beast_vote, sniper_vote, conspiracy_vote)
        final[game_id] = fused
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    fuse_predictions()
