import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path(os.getenv("REF_CONSPIRACY_DATA_DIR", "data/referee_conspiracy"))
GAME_LOG_PATH = DATA_DIR / "crew_game_log.parquet"
SCHEDULE_PATH = DATA_DIR / "schedules_2018_2024.parquet"
PENALTY_PATH = DATA_DIR / "penalties_2018_2024.parquet"
OUTPUT_PATH = DATA_DIR / "hrm_predictions.json"
TASK_PATH = DATA_DIR / "hrm_tasks.json"
CHECKPOINT_PATH = DATA_DIR / "hrm_checkpoint.pt"
BATCH_LIMIT = 500
HRM_LABELS = ["NORMAL_FLOW", "SCRIPTED_BLUEBALL", "SCRIPTED_BLACKOUT", "FLAG_FARM"]


class HModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class LModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HRMComposite(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.h_module = HModule(feature_dim, 64)
        self.l_module = LModule(feature_dim + 4, 64)

    def forward_high(self, x):
        return self.h_module(x)

    def forward_low(self, x, probs):
        return self.l_module(torch.cat([x, probs], dim=-1))


def load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def estimate_rain(temp, wind, roof):
    base = 0.1
    roof_str = str(roof).lower() if roof else ""
    if roof_str in {"outdoors", "open"}:
        if not math.isnan(temp):
            if temp < 35:
                base += 0.35
            elif temp < 55:
                base += 0.2
        if not math.isnan(wind):
            base += min(0.3, max(0.0, wind / 40.0))
    else:
        base = 0.05
    return float(max(0.0, min(1.0, base)))


def infer_scoring(points):
    total = int(round(points if points is not None else 0))
    best = None
    for td7 in range(0, total // 7 + 2):
        for td8 in range(0, total // 8 + 2):
            for td6 in range(0, total // 6 + 2):
                td_points = td7 * 7 + td8 * 8 + td6 * 6
                if td_points > total:
                    continue
                td_count = td7 + td8 + td6
                remaining = total - td_points
                for fg in range(0, remaining // 3 + 1):
                    rem2 = remaining - fg * 3
                    for safety in range(0, rem2 // 2 + 1):
                        rem3 = rem2 - safety * 2
                        if rem3 != 0:
                            continue
                        candidate = (td_count, fg)
                        if best is None:
                            best = candidate
                            continue
                        if candidate[1] > best[1]:
                            best = candidate
                            continue
                        if candidate[1] == best[1] and candidate[0] < best[0]:
                            best = candidate
    if best is None:
        td_guess = total // 7
        fg_guess = (total - td_guess * 7) // 3
        return int(td_guess), int(max(fg_guess, 0))
    return int(best[0]), int(best[1])


def compute_crowd_noise(is_primetime, is_overseas, roof):
    value = 0.65
    if is_primetime:
        value += 0.15
    if is_overseas:
        value -= 0.2
    if roof:
        if str(roof).lower() in {"dome", "closed", "fixed"}:
            value += 0.05
    return float(max(0.0, min(1.0, value)))


def derive_script_tag(red_zone, blackout, flag_density, penalties):
    if red_zone and flag_density >= 0.11:
        return "SCRIPTED_BLUEBALL"
    if blackout and flag_density >= 0.1:
        return "SCRIPTED_BLACKOUT"
    if penalties >= 18 or flag_density >= 0.125:
        return "FLAG_FARM"
    return "NORMAL_FLOW"


def build_penalty_metrics():
    penalties = load_parquet(PENALTY_PATH)
    if penalties.empty:
        return pd.DataFrame(
            {
                "game_id": [],
                "defensive_rate": [],
                "offensive_rate": [],
                "late_rate": [],
                "positive_swing_rate": [],
                "swing_magnitude": [],
            }
        )
    penalties["is_defensive"] = penalties["penalty_team"] == penalties["defteam"]
    penalties["is_offensive"] = penalties["penalty_team"] == penalties["posteam"]
    penalties["is_late"] = penalties["qtr"].fillna(0.0) >= 4.0
    penalties["positive_swing"] = penalties["score_swing"].fillna(0.0) > 0.0
    penalties["abs_swing"] = penalties["score_swing"].fillna(0.0).abs()
    summary = (
        penalties.groupby("game_id")
        .agg(
            defensive_rate=("is_defensive", "mean"),
            offensive_rate=("is_offensive", "mean"),
            late_rate=("is_late", "mean"),
            positive_swing_rate=("positive_swing", "mean"),
            swing_magnitude=("abs_swing", "mean"),
        )
        .reset_index()
    )
    return summary


def build_dataframe():
    crew = load_parquet(GAME_LOG_PATH)
    schedule = load_parquet(SCHEDULE_PATH)
    schedule_subset = schedule[
        [
            "game_id",
            "temp",
            "wind",
            "roof",
            "surface",
            "under_odds",
            "over_odds",
        ]
    ]
    penalty_metrics = build_penalty_metrics()
    data = crew.merge(schedule_subset, on="game_id", how="left")
    data = data.merge(penalty_metrics, on="game_id", how="left")
    
    # Ensure columns exist (for future games not in schedules/penalties)
    for col in ["defensive_rate", "offensive_rate", "late_rate", "positive_swing_rate", "swing_magnitude"]:
        if col not in data.columns:
            data[col] = 0.5 if "rate" in col else 0.25 if col == "late_rate" else 0.5
    
    for col in ["temp", "wind", "under_odds", "over_odds", "roof", "surface"]:
        if col not in data.columns:
            data[col] = 65.0 if col == "temp" else 5.0 if col == "wind" else -110.0 if "odds" in col else ""
    
    data["home_score"] = data["home_score"].fillna(0.0)
    data["away_score"] = data["away_score"].fillna(0.0)
    data["points_total"] = data["home_score"] + data["away_score"]
    data["total_line"] = data["total_line"].fillna(data["points_total"])
    data["spread_line"] = data["spread_line"].fillna(0.0)
    data["flag_density"] = data["flag_density"].fillna(0.0)
    data["penalties"] = data["penalties"].fillna(0)
    data["penalty_yards"] = data["penalty_yards"].fillna(0.0)
    data["total_plays"] = data["total_plays"].fillna(0.0)
    data["overtime_plays"] = data["overtime_plays"].fillna(0.0)
    data["score_swing_mean_abs"] = data["score_swing_mean_abs"].fillna(0.0)
    data["score_swing_positive_rate"] = data["score_swing_positive_rate"].fillna(0.0)
    data["is_primetime"] = data["is_primetime"].fillna(False)
    data["is_low_visibility"] = data["is_low_visibility"].fillna(False)
    data["is_overseas"] = data["is_overseas"].fillna(False)
    data["defensive_rate"] = data["defensive_rate"].fillna(0.5)
    data["offensive_rate"] = data["offensive_rate"].fillna(0.5)
    data["late_rate"] = data["late_rate"].fillna(0.25)
    data["positive_swing_rate"] = data["positive_swing_rate"].fillna(0.5)
    data["swing_magnitude"] = data["swing_magnitude"].fillna(0.5)
    data["week"] = data["week"].fillna(0)
    data["season"] = data["season"].fillna(0)
    data["temp"] = data["temp"].fillna(65.0)
    data["wind"] = data["wind"].fillna(5.0)
    data["under_odds"] = data["under_odds"].fillna(-110.0)
    data["over_odds"] = data["over_odds"].fillna(-110.0)
    return data


def build_records():
    df = build_dataframe()
    if len(df) == 0:
        return [], np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    baseline_flag = df["flag_density"].replace(0.0, np.nan).mean()
    if math.isnan(baseline_flag) or baseline_flag <= 0.0:
        baseline_flag = 0.1
    records = []
    features = []
    high_targets = []
    low_targets = []
    for row in df.itertuples():
        temp = float(row.temp) if not math.isnan(float(row.temp)) else 65.0
        wind = float(row.wind) if not math.isnan(float(row.wind)) else 5.0
        rain = estimate_rain(temp, wind, row.roof)
        crowd = compute_crowd_noise(bool(row.is_primetime), bool(row.is_overseas), row.roof)
        blackout = bool(row.is_low_visibility) or (str(row.roof).lower() in {"dome", "closed", "fixed"})
        home_tds, home_fgs = infer_scoring(row.home_score)
        red_zone = bool(home_tds <= 1 and home_fgs >= 3)
        script_tag = derive_script_tag(red_zone, blackout, float(row.flag_density), int(row.penalties))
        label_index = HRM_LABELS.index(script_tag)
        total_line = float(row.total_line) if not math.isnan(float(row.total_line)) else float(row.points_total)
        under_target = 0.5
        if total_line > 0:
            under_target = 1.0 if float(row.points_total) <= total_line else 0.0
        flag_target = float(row.flag_density) / float(baseline_flag)
        flag_target = float(max(0.25, min(3.0, flag_target)))
        feature_vector = [
            float(row.points_total),
            float(row.home_score) - float(row.away_score),
            float(row.flag_density),
            float(row.penalties),
            float(row.penalty_yards),
            float(row.total_plays),
            float(row.overtime_plays),
            1.0 if bool(row.is_primetime) else 0.0,
            1.0 if bool(row.is_low_visibility) else 0.0,
            1.0 if bool(row.is_overseas) else 0.0,
            float(row.spread_line),
            total_line,
            temp / 100.0,
            wind / 50.0,
            rain,
            crowd,
            1.0 if blackout else 0.0,
            1.0 if red_zone else 0.0,
            float(home_tds),
            float(home_fgs),
            float(row.score_swing_mean_abs),
            float(row.score_swing_positive_rate),
            float(row.week) / 20.0,
            float(row.season) / 2030.0,
            float(row.under_odds),
            float(row.over_odds),
            float(row.defensive_rate),
            float(row.offensive_rate),
            float(row.late_rate),
            float(row.positive_swing_rate),
            float(row.swing_magnitude),
        ]
        task = {
            "game_id": row.game_id,
            "home_team": row.home_team,
            "home_tds": home_tds,
            "home_fgs": home_fgs,
            "red_zone": red_zone,
            "weather": {"rain": round(rain, 3), "temperature": temp, "wind": wind},
            "crowd_noise": round(crowd, 3),
            "blackout": blackout,
            "script_tag": script_tag,
        }
        records.append({"game_id": row.game_id, "task": task})
        features.append(feature_vector)
        high_targets.append(label_index)
        low_targets.append([under_target, flag_target])
    return records, np.array(features, dtype=np.float32), np.array(high_targets, dtype=np.int64), np.array(low_targets, dtype=np.float32)


def normalize_features(features, mean=None, std=None):
    if features.size == 0:
        return features, np.zeros(features.shape[1] if features.ndim > 1 else 0), np.ones(features.shape[1] if features.ndim > 1 else 0)
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    normalized = (features - mean) / std
    return normalized, mean, std


def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def train_model(model, features, high_targets, low_targets, device, epochs=10):
    if len(features) == 0:
        return
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y_high = torch.tensor(high_targets, dtype=torch.long, device=device)
    y_low = torch.tensor(low_targets, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_high(x)
        h_loss = F.cross_entropy(logits, y_high)
        probs = torch.softmax(logits.detach(), dim=-1)
        low_logits = model.forward_low(x, probs)
        under_loss = F.binary_cross_entropy_with_logits(low_logits[:, 0], y_low[:, 0])
        flag_pred = torch.exp(torch.clamp(low_logits[:, 1], min=-4.0, max=4.0))
        flag_loss = F.mse_loss(flag_pred, y_low[:, 1])
        loss = h_loss + under_loss + flag_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
    model.eval()


def run_inference(model, features, records, device):
    if len(features) == 0:
        return {}
    x = torch.tensor(features, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model.forward_high(x)
        probs = torch.softmax(logits, dim=-1)
        low_logits = model.forward_low(x, probs)
        under_probs = torch.sigmoid(low_logits[:, 0])
        flag_spikes = torch.exp(torch.clamp(low_logits[:, 1], min=-4.0, max=4.0))
    outputs = {}
    prob_values = probs.cpu().numpy()
    under_values = under_probs.cpu().numpy()
    flag_values = flag_spikes.cpu().numpy()
    for idx, rec in enumerate(records):
        class_index = int(np.argmax(prob_values[idx]))
        label = HRM_LABELS[class_index]
        max_prob = float(prob_values[idx][class_index])
        under_prob = float(under_values[idx])
        flag_spike = float(flag_values[idx])
        confidence = max(0.0, min(1.0, max_prob * 0.6 + under_prob * 0.4))
        outputs[rec["game_id"]] = {
            "h_module": label,
            "l_module": {"under_prob": round(under_prob, 4), "flag_spike": round(flag_spike, 4)},
            "confidence": round(confidence, 4),
        }
    return outputs


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_checkpoint(model, path, device):
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state"])
    return np.array(payload["mean"], dtype=np.float32), np.array(payload["std"], dtype=np.float32)


def save_checkpoint(model, mean, std, path):
    payload = {
        "state": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def select_training_slice(features, high_targets, low_targets, limit=BATCH_LIMIT):
    total = len(features)
    if total <= limit:
        return features, high_targets, low_targets
    indices = list(range(total))
    random.shuffle(indices)
    chosen = indices[:limit]
    return features[chosen], high_targets[chosen], low_targets[chosen]


def run_pipeline(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    records, features, high_targets, low_targets = build_records()
    if not records:
        raise RuntimeError("No games available")
    feature_dim = features.shape[1]
    model = HRMComposite(feature_dim)
    mean, std = load_checkpoint(model, CHECKPOINT_PATH, device)
    if args.train or mean is None or std is None:
        train_features, train_high, train_low = select_training_slice(features, high_targets, low_targets, limit=min(args.limit, BATCH_LIMIT))
        normalized_train, mean, std = normalize_features(train_features)
        model.to(device)
        train_model(model, normalized_train, train_high, train_low, device, epochs=args.epochs)
        save_checkpoint(model, mean, std, CHECKPOINT_PATH)
    else:
        model.to(device)
    normalized_all, mean, std = normalize_features(features, mean=mean, std=std)
    if args.int8:
        model.cpu()
        model = quantize_model(model)
    predictions = run_inference(model, normalized_all, records, device if not args.int8 else torch.device("cpu"))
    task_payload = {rec["game_id"]: rec["task"] for rec in records}
    save_json(task_payload, TASK_PATH)
    save_json(predictions, OUTPUT_PATH)
    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrm", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--int8", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.hrm:
        exit(0)
    run_pipeline(args)
