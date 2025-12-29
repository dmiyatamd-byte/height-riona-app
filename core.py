import sqlite3, json, uuid, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

DB_PATH = "riona_predict.db"
MODEL_PATH = "model.json"

HORIZONS = [12, 24]          # weeks
AUTO_CAL_MIN_N = 11          # 「10件超えたら」=> 11以上

# ---------------------------
# Utils
# ---------------------------
def now_ts() -> int:
    return int(time.time())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def calc_tsat(fe: float, tibc: float) -> Optional[float]:
    if tibc is None or tibc <= 0:
        return None
    return 100.0 * fe / tibc

def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_int(x, default=0):
    try:
        if x is None:
            return int(default)
        if isinstance(x, str) and x.strip() == "":
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)

# ---------------------------
# Data models
# ---------------------------
@dataclass
class Labs:
    hb: float
    fe: float
    ferritin: float
    tibc: float
    tsat: Optional[float] = None

@dataclass
class Ctx:
    dose_mg_day: int = 500
    adherence: float = 1.0
    bleed: float = 0.0
    inflam: float = 0.0

# ---------------------------
# Rule-based prediction (adult-equivalent)
# ---------------------------
def rule_predict(labs: Labs, ctx: Ctx, horizon_weeks: int) -> Dict[str, Any]:
    tsat0 = labs.tsat if labs.tsat is not None else calc_tsat(labs.fe, labs.tibc)
    if tsat0 is None:
        raise ValueError("TSATまたはFe/TIBCが必要です")

    scale = horizon_weeks / 12.0

    dose_scale = clamp(ctx.dose_mg_day / 500.0, 0.5, 2.0)
    eff = clamp(ctx.adherence * dose_scale, 0.2, 2.0)

    # TSAT (cap 40)
    tsat1 = clamp(tsat0 + (10.0 * eff * scale), 5.0, 40.0)

    # Hb (cap 17.5)
    low_hb_boost = clamp(1.0 + (10.5 - labs.hb) * 0.15, 0.8, 1.4)
    iron_boost = clamp((tsat1 - tsat0) / 10.0, 0.0, 1.5)
    loss_penalty = 1.0 - clamp(ctx.bleed * 0.7, 0.0, 0.7)
    delta_hb = (0.5 * eff * scale) * low_hb_boost * (0.6 + 0.4 * iron_boost) * loss_penalty
    hb1 = clamp(labs.hb + delta_hb, 5.0, 17.5)

    # Ferritin (cap 250)
    ferritin_gain = (30.0 * eff * scale) * clamp((tsat1 - tsat0) / 10.0, 0.0, 2.0)
    ferritin_inflation = ctx.inflam * (30.0 * scale)
    ferritin1 = clamp(labs.ferritin + ferritin_gain + ferritin_inflation, 1.0, 250.0)

    # Fe reconstructed for consistency
    fe1 = clamp(tsat1 * labs.tibc / 100.0, 1.0, 400.0)

    alerts = []
    if tsat1 >= 40.0:
        alerts.append("TSAT上限到達：鉄過剰注意")
    if ferritin1 >= 200.0:
        alerts.append("フェリチン高値域：経過観察/減量検討")
    if ctx.inflam > 0.3:
        alerts.append("炎症あり：フェリチン解釈注意")

    return {
        "Hb": round(hb1, 2),
        "Fe": round(fe1, 1),
        "Ferritin": round(ferritin1, 1),
        "TSAT": round(tsat1, 1),
        "alerts": alerts,
    }

# ---------------------------
# Model persistence
# ---------------------------
def load_models() -> Dict[str, Any]:
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"models": {}}

def save_models(models: Dict[str, Any]) -> None:
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

def get_model_for_horizon(horizon_weeks: int) -> Optional[Dict[str, Any]]:
    return load_models().get("models", {}).get(str(horizon_weeks))

def calibrated_predict(labs: Labs, ctx: Ctx, horizon_weeks: int, model: Dict[str, Any]) -> Dict[str, Any]:
    base = rule_predict(labs, ctx, horizon_weeks=horizon_weeks)
    tsat0 = labs.tsat if labs.tsat is not None else calc_tsat(labs.fe, labs.tibc)
    if tsat0 is None:
        raise ValueError("TSATまたはFe/TIBCが必要です")

    x = {
        "hb0": labs.hb,
        "tsat0": tsat0,
        "ferritin0": labs.ferritin,
        "tibc0": labs.tibc,
        "dose": float(ctx.dose_mg_day),
        "adherence": ctx.adherence,
        "bleed": ctx.bleed,
        "inflam": ctx.inflam,
    }

    def lin(adj):
        b = float(adj.get("bias", 0.0))
        w = adj.get("weights", {})
        s = b
        for k, v in w.items():
            s += float(v) * float(x.get(k, 0.0))
        return s

    adj_hb = lin(model["adjustments"]["hb"])
    adj_tsat = lin(model["adjustments"]["tsat"])
    adj_ferr = lin(model["adjustments"]["ferritin"])

    hb1 = clamp(base["Hb"] + adj_hb, 5.0, 17.5)
    tsat1 = clamp(base["TSAT"] + adj_tsat, 5.0, 40.0)
    ferr1 = clamp(base["Ferritin"] + adj_ferr, 1.0, 250.0)
    fe1 = clamp(tsat1 * labs.tibc / 100.0, 1.0, 400.0)

    out = dict(base)
    out.update(
        Hb=round(hb1, 2),
        TSAT=round(tsat1, 1),
        Ferritin=round(ferr1, 1),
        Fe=round(fe1, 1),
    )
    out["alerts"] = list(set(out.get("alerts", []) + ["校正モデル適用"]))
    return out

# ---------------------------
# DB schema
# ---------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Base tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases(
      case_id TEXT PRIMARY KEY,
      created_at INTEGER,
      note TEXT,
      external_id TEXT,              -- ★人が扱いやすいID（JAMS連動用）
      hb0 REAL, fe0 REAL, ferritin0 REAL, tibc0 REAL, tsat0 REAL,
      dose_mg_day INTEGER, adherence REAL, bleed REAL, inflam REAL,
      pred_hb_w12 REAL, pred_fe_w12 REAL, pred_ferr_w12 REAL, pred_tsat_w12 REAL, model_w12 TEXT,
      pred_hb_w24 REAL, pred_fe_w24 REAL, pred_ferr_w24 REAL, pred_tsat_w24 REAL, model_w24 TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS followups(
      case_id TEXT,
      horizon_weeks INTEGER,
      followup_at INTEGER,
      hb REAL, fe REAL, ferritin REAL, tibc REAL, tsat REAL,
      PRIMARY KEY(case_id, horizon_weeks),
      FOREIGN KEY(case_id) REFERENCES cases(case_id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_versions(
      horizon_weeks INTEGER,
      version TEXT,
      trained_at INTEGER,
      n_train INTEGER,
      metrics_json TEXT,
      model_json TEXT,
      PRIMARY KEY(horizon_weeks, version)
    )""")

    # ---- schema migration for older DBs (add external_id if missing) ----
    cur.execute("PRAGMA table_info(cases)")
    cols = [r[1] for r in cur.fetchall()]
    if "external_id" not in cols:
        cur.execute("ALTER TABLE cases ADD COLUMN external_id TEXT")

    # Index for faster lookup
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cases_external_id ON cases(external_id)")

    con.commit()
    con.close()
def get_counts() -> Dict[str, int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM cases")
    n_cases = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM followups WHERE horizon_weeks=12")
    n_f12 = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM followups WHERE horizon_weeks=24")
    n_f24 = int(cur.fetchone()[0])
    con.close()
    return {"cases": n_cases, "followups12": n_f12, "followups24": n_f24}

def list_cases(limit: int = 200) -> List[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    r = cur.fetchone()
    con.close()
    return dict(r) if r else None

def update_case_context_and_predictions(case_id: str, ctx: Ctx, note: Optional[str] = None, external_id: Optional[str] = None) -> Dict[str, Any]:
    """Update dose/adherence/bleed/inflam (and optional note/external_id), then recompute and store predictions for 12/24w.
    This is a *persistent* update (saved)."""
    row = get_case(case_id)
    if not row:
        raise ValueError("case not found")

    labs0 = Labs(
        hb=safe_float(row["hb0"]),
        fe=safe_float(row["fe0"]),
        ferritin=safe_float(row["ferritin0"]),
        tibc=safe_float(row["tibc0"]),
        tsat=safe_float(row["tsat0"]) if row.get("tsat0") is not None else None
    )

    p12, tag12 = predict_for_horizon(labs0, ctx, 12)
    p24, tag24 = predict_for_horizon(labs0, ctx, 24)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    fields = {
        "dose_mg_day": int(ctx.dose_mg_day),
        "adherence": float(ctx.adherence),
        "bleed": float(ctx.bleed),
        "inflam": float(ctx.inflam),
        "pred_hb_w12": p12["Hb"], "pred_fe_w12": p12["Fe"], "pred_ferr_w12": p12["Ferritin"], "pred_tsat_w12": p12["TSAT"], "model_w12": tag12,
        "pred_hb_w24": p24["Hb"], "pred_fe_w24": p24["Fe"], "pred_ferr_w24": p24["Ferritin"], "pred_tsat_w24": p24["TSAT"], "model_w24": tag24,
    }
    if note is not None:
        fields["note"] = note
    if external_id is not None:
        fields["external_id"] = external_id

    sets = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [case_id]
    cur.execute(f"UPDATE cases SET {sets} WHERE case_id=?", vals)
    con.commit()
    con.close()

    return {"case_id": case_id, "12w": p12, "24w": p24, "model_w12": tag12, "model_w24": tag24}

def simulate_predictions_for_case(case_id: str, ctx: Ctx) -> Dict[str, Any]:
    """Compute predictions with a modified ctx WITHOUT saving."""
    row = get_case(case_id)
    if not row:
        raise ValueError("case not found")
    labs0 = Labs(
        hb=safe_float(row["hb0"]),
        fe=safe_float(row["fe0"]),
        ferritin=safe_float(row["ferritin0"]),
        tibc=safe_float(row["tibc0"]),
        tsat=safe_float(row["tsat0"]) if row.get("tsat0") is not None else None
    )
    p12, tag12 = predict_for_horizon(labs0, ctx, 12)
    p24, tag24 = predict_for_horizon(labs0, ctx, 24)
    return {"case_id": case_id, "12w": p12, "24w": p24, "model_w12": tag12, "model_w24": tag24}

def get_followup(case_id: str, horizon_weeks: int) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM followups WHERE case_id=? AND horizon_weeks=?", (case_id, horizon_weeks))
    r = cur.fetchone()
    con.close()
    return dict(r) if r else None

# ---------------------------
# Case lookup / deletion utilities
# ---------------------------
def resolve_case_id(identifier: str) -> Optional[str]:
    """Accepts either case_id (UUID) or external_id (human/JAMS ID)."""
    if not identifier:
        return None
    ident = identifier.strip()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # First try exact match on case_id
    cur.execute("SELECT case_id FROM cases WHERE case_id = ?", (ident,))
    row = cur.fetchone()
    if row:
        con.close()
        return row[0]
    # Then try external_id
    cur.execute("SELECT case_id FROM cases WHERE external_id = ?", (ident,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None

def set_external_id(case_id: str, external_id: str) -> None:
    external_id = (external_id or "").strip()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE cases SET external_id=? WHERE case_id=?", (external_id, case_id))
    con.commit()
    con.close()

def delete_case(case_id: str) -> Dict[str, Any]:
    """Delete a case and all followups (hard delete)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM followups WHERE case_id = ?", (case_id,))
    cur.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
    con.commit()
    con.close()
    return {"deleted": True, "case_id": case_id}


# ---------------------------
# Register baseline
# ---------------------------
def predict_for_horizon(labs: Labs, ctx: Ctx, horizon_weeks: int) -> Tuple[Dict[str, Any], str]:
    model = get_model_for_horizon(horizon_weeks)
    if model:
        p = calibrated_predict(labs, ctx, horizon_weeks, model)
        tag = f"calibrated:{model.get('version','unknown')}"
    else:
        p = rule_predict(labs, ctx, horizon_weeks)
        tag = "rule_v1"
    return p, tag

def register_case(labs: Labs, ctx: Ctx, note: str = "", external_id: str = "") -> Tuple[str, Dict[str, Any]]:
    tsat0 = labs.tsat if labs.tsat is not None else calc_tsat(labs.fe, labs.tibc)
    if tsat0 is None:
        raise ValueError("TSATまたはFe/TIBCが必要です")

    p12, tag12 = predict_for_horizon(labs, ctx, 12)
    p24, tag24 = predict_for_horizon(labs, ctx, 24)

    case_id = str(uuid.uuid4())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """INSERT INTO cases(
            case_id, created_at, note, external_id,
            hb0, fe0, ferritin0, tibc0, tsat0,
            dose_mg_day, adherence, bleed, inflam,
            pred_hb_w12, pred_fe_w12, pred_ferr_w12, pred_tsat_w12, model_w12,
            pred_hb_w24, pred_fe_w24, pred_ferr_w24, pred_tsat_w24, model_w24
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            case_id, now_ts(), note, external_id,
            labs.hb, labs.fe, labs.ferritin, labs.tibc, tsat0,
            ctx.dose_mg_day, ctx.adherence, ctx.bleed, ctx.inflam,
            p12["Hb"], p12["Fe"], p12["Ferritin"], p12["TSAT"], tag12,
            p24["Hb"], p24["Fe"], p24["Ferritin"], p24["TSAT"], tag24
        )
    )
    con.commit()
    con.close()
    return case_id, {"12w": p12, "24w": p24, "model_w12": tag12, "model_w24": tag24}

# ---------------------------
# Calibration (Ridge on residuals)
# ---------------------------
FEATURE_KEYS = ["hb0", "tsat0", "ferritin0", "tibc0", "dose", "adherence", "bleed", "inflam"]

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.size == 0:
        raise ValueError("X is empty (no training data)")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean

    n_feat = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(n_feat)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y_mean - X_mean @ w)
    return w, b

def _mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def _fetch_training_rows(horizon_weeks: int) -> List[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """SELECT
            c.case_id,
            c.hb0, c.fe0, c.ferritin0, c.tibc0, c.tsat0,
            c.dose_mg_day, c.adherence, c.bleed, c.inflam,
            f.hb as hb_w, f.fe as fe_w, f.ferritin as ferr_w, f.tibc as tibc_w, f.tsat as tsat_w
        FROM cases c
        JOIN followups f ON c.case_id = f.case_id
        WHERE f.horizon_weeks = ?""", (horizon_weeks,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def train_calibration(horizon_weeks: int, force: bool = False, alpha: float = 10.0) -> Dict[str, Any]:
    rows = _fetch_training_rows(horizon_weeks)
    n = len(rows)

    if n == 0:
        return {"status": "skipped", "reason": "no followup data for training", "n_train": 0}

    if (not force) and n < AUTO_CAL_MIN_N:
        return {"status": "skipped", "reason": f"training rows < {AUTO_CAL_MIN_N}", "n_train": n}

    models_all = load_models()
    current = models_all.get("models", {}).get(str(horizon_weeks))
    if (not force) and current and int(current.get("n_train", 0)) == n:
        return {"status": "skipped", "reason": "no new data since last training", "n_train": n}

    X_list = []
    y_hb = []
    y_tsat = []
    y_ferr = []

    for r in rows:
        labs0 = Labs(
            hb=safe_float(r["hb0"]),
            fe=safe_float(r["fe0"]),
            ferritin=safe_float(r["ferritin0"]),
            tibc=safe_float(r["tibc0"]),
            tsat=safe_float(r["tsat0"]) if r["tsat0"] is not None else None,
        )
        ctx = Ctx(
            dose_mg_day=safe_int(r["dose_mg_day"], 500),
            adherence=safe_float(r["adherence"], 1.0),
            bleed=safe_float(r["bleed"], 0.0),
            inflam=safe_float(r["inflam"], 0.0),
        )

        rule = rule_predict(labs0, ctx, horizon_weeks=horizon_weeks)

        y_hb.append(safe_float(r["hb_w"]) - safe_float(rule["Hb"]))
        y_tsat.append(safe_float(r["tsat_w"]) - safe_float(rule["TSAT"]))
        y_ferr.append(safe_float(r["ferr_w"]) - safe_float(rule["Ferritin"]))

        X_list.append([
            safe_float(r["hb0"]),
            safe_float(r["tsat0"]),
            safe_float(r["ferritin0"]),
            safe_float(r["tibc0"]),
            safe_float(r["dose_mg_day"]),
            safe_float(r["adherence"]),
            safe_float(r["bleed"]),
            safe_float(r["inflam"]),
        ])

    X = np.asarray(X_list, dtype=float)
    y_hb = np.asarray(y_hb, dtype=float)
    y_tsat = np.asarray(y_tsat, dtype=float)
    y_ferr = np.asarray(y_ferr, dtype=float)

    w_hb, b_hb = _ridge_fit(X, y_hb, alpha=alpha)
    w_tsat, b_tsat = _ridge_fit(X, y_tsat, alpha=alpha)
    w_ferr, b_ferr = _ridge_fit(X, y_ferr, alpha=alpha)

    pred_hb = X @ w_hb + b_hb
    pred_tsat = X @ w_tsat + b_tsat
    pred_ferr = X @ w_ferr + b_ferr

    metrics = {
        "mae_hb_residual": _mae(y_hb, pred_hb),
        "mae_tsat_residual": _mae(y_tsat, pred_tsat),
        "mae_ferritin_residual": _mae(y_ferr, pred_ferr),
        "alpha": alpha,
    }

    def pack(w, b):
        return {"bias": float(b), "weights": {k: float(v) for k, v in zip(FEATURE_KEYS, w.tolist())}}

    version = time.strftime("%Y%m%d-%H%M%S")
    model = {
        "version": version,
        "trained_at": now_ts(),
        "horizon_weeks": horizon_weeks,
        "n_train": n,
        "features": FEATURE_KEYS,
        "adjustments": {
            "hb": pack(w_hb, b_hb),
            "tsat": pack(w_tsat, b_tsat),
            "ferritin": pack(w_ferr, b_ferr),
        },
        "metrics": metrics,
    }

    models_all.setdefault("models", {})
    models_all["models"][str(horizon_weeks)] = model
    save_models(models_all)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO model_versions(horizon_weeks, version, trained_at, n_train, metrics_json, model_json) VALUES(?,?,?,?,?,?)",
        (horizon_weeks, version, model["trained_at"], n, json.dumps(metrics, ensure_ascii=False), json.dumps(model, ensure_ascii=False)),
    )
    con.commit()
    con.close()

    return {"status": "trained", "version": version, "n_train": n, "metrics": metrics}

def add_followup(case_id: str, horizon_weeks: int, hb: float, fe: float, ferritin: float, tibc: float, tsat: Optional[float] = None) -> Dict[str, Any]:
    if horizon_weeks not in HORIZONS:
        raise ValueError("horizon_weeks must be 12 or 24")

    if tsat is None:
        tsat = calc_tsat(fe, tibc)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO followups(case_id, horizon_weeks, followup_at, hb, fe, ferritin, tibc, tsat) VALUES(?,?,?,?,?,?,?,?)",
        (case_id, horizon_weeks, now_ts(), hb, fe, ferritin, tibc, tsat),
    )
    con.commit()
    con.close()

    result = train_calibration(horizon_weeks, force=False)
    return {"saved": True, "auto_calibration": result}

def get_model_status() -> Dict[str, Any]:
    models_all = load_models()
    return {str(h): models_all.get("models", {}).get(str(h)) for h in HORIZONS}

# ---------------------------
# CSV templates
# ---------------------------
def baseline_template_csv() -> str:
    cols = ["external_id", "note", "hb0", "fe0", "ferritin0", "tibc0", "tsat0", "dose_mg_day", "adherence", "bleed", "inflam"]
    return ",".join(cols) + "\n"

def followup_template_csv() -> str:
    cols = ["identifier", "horizon_weeks", "hb", "fe", "ferritin", "tibc", "tsat"]
    return ",".join(cols) + "\n"
