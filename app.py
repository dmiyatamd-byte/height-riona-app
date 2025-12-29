import os
import sqlite3
import hashlib
import secrets
import json
import re
import base64
from datetime import datetime, timedelta, timezone, date, time

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import calendar
import altair as alt

from core import init_db, Labs, Ctx, register_case, add_followup, resolve_case_id, simulate_predictions_for_case

# =========================
# テスト用（後でSecretsへ移行）
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# =========================
# Config
# =========================
TZ = timezone(timedelta(hours=9))
SPORTS = ["サッカー", "ラグビー", "野球", "テニス", "水泳"]
RESERVE_URL = "https://qr.digikar-smart.jp/6bcfb249-1c73-4789-af01-2cb02fec9f42/reserve"

USERS_DB_PATH = "users.db"
DATA_DB_PATH = "patient_data.db"

ALP_STOP_THRESHOLD = 135.0
BA_CLOSED_THRESHOLD = 16.0
TYPE_EARLY_DELTA = 1.0
TYPE_DELAY_DELTA = -1.0
Y_AXIS_LO, Y_AXIS_HI = 100.0, 200.0

IGF1_RANGES = {
    "M": {3:(24,164),4:(32,176),5:(44,193),6:(55,215),7:(63,247),8:(72,292),9:(84,350),
          10:(99,423),11:(113,499),12:(125,557),13:(133,579),14:(138,570),15:(141,552),
          16:(142,543),17:(142,540),18:(142,526),19:(143,501),20:(142,470)},
    "F": {3:(40,227),4:(48,238),5:(56,252),6:(69,287),7:(89,357),8:(111,438),9:(133,517),
          10:(155,588),11:(175,638),12:(188,654),13:(193,643),14:(193,625),15:(192,614),
          16:(192,611),17:(191,599),18:(188,574),19:(182,539),20:(175,499)}
}

# =========================
# UI
# =========================
def apply_css():
    st.markdown("""
    <style>
      .block-container { padding-top: 2.2rem; }
      div[data-testid="stHorizontalBlock"] { gap: 6px !important; padding: 0 4px; }
div[data-testid="stHorizontalBlock"]::after{ content:""; display:block; height:1px; background: rgba(0,0,0,0.10); margin-top:-1px; }
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]{
  border: 1px solid rgba(0,0,0,0.10);
  border-bottom: 0;
  border-radius: 12px 12px 0 0;
  padding: 8px 14px !important;
  background: rgba(255,255,255,0.85);
  box-shadow: 0 6px 14px rgba(0,0,0,0.06);
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:has(input:checked){
  background: #ffffff;
  box-shadow: 0 10px 22px rgba(0,0,0,0.08);
  transform: translateY(1px);
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"] p{ margin:0; font-weight:700; }

      div[data-testid="column"] button{ width: 100%; }
      .stExpander{
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 10px 24px rgba(0,0,0,0.04);
        background: rgba(255,255,255,0.92);
      }
    
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(1){
  border-left: 4px solid rgba(59,130,246,0.8) !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(2){
  border-left: 4px solid rgba(239,68,68,0.8) !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(3){
  border-left: 4px solid rgba(16,185,129,0.8) !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(4){
  border-left: 4px solid rgba(245,158,11,0.85) !important;
}

/* ===== Premium mobile nav ===== */
@media (max-width: 640px){
  .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
  div[data-testid="stHorizontalBlock"] { gap: 8px !important; }
  div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]{
    padding: 10px 14px !important;
    border-radius: 14px !important;
    font-size: 16px !important;
  }
}
/* Make nav look like premium segmented tabs */
div[data-testid="stHorizontalBlock"]{
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 8px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.06);
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(10px);
}
div[data-testid="stHorizontalBlock"]::after{ display:none !important; }

div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]{
  border: 0 !important;
  border-radius: 14px !important;
  padding: 9px 14px !important;
  background: rgba(0,0,0,0.04) !important;
  box-shadow: none !important;
  transition: all 120ms ease;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"]:has(input:checked){
  background: #111827 !important;
  color: #fff !important;
  box-shadow: 0 10px 20px rgba(0,0,0,0.12) !important;
  transform: none !important;
}
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"] p{ font-weight: 800; }

/* Color accents per tab label (unselected) */
div[data-testid="stHorizontalBlock"] label[data-baseweb="radio"] p:contains("身長"){ }


/* ===== Main nav (radio) premium ===== */
div[data-testid="stRadio"] div[role="radiogroup"]{
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 16px;
  padding: 8px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.06);
}
div[data-testid="stRadio"] label[data-baseweb="radio"]{
  border-radius: 14px !important;
  padding: 10px 14px !important;
  background: rgba(0,0,0,0.04) !important;
  border-left: 4px solid rgba(0,0,0,0.0) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){
  background: rgba(255,255,255,0.98) !important;
  color: #111827 !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.12) !important;
  outline: 2px solid rgba(17,24,39,0.20);
}

div[data-testid="stRadio"] label[data-baseweb="radio"] p{ margin:0; font-weight:800; }
@media (max-width: 640px){
  div[data-testid="stRadio"] label[data-baseweb="radio"]{ font-size: 16px !important; }
}


div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(1){ border-left:4px solid rgba(59,130,246,0.85) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(2){ border-left:4px solid rgba(239,68,68,0.85) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(3){ border-left:4px solid rgba(16,185,129,0.85) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(4){ border-left:4px solid rgba(245,158,11,0.90) !important; }

div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(1):has(input:checked){ background: rgba(59,130,246,0.10) !important; outline-color: rgba(59,130,246,0.35) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(2):has(input:checked){ background: rgba(239,68,68,0.10) !important; outline-color: rgba(239,68,68,0.35) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(3):has(input:checked){ background: rgba(16,185,129,0.10) !important; outline-color: rgba(16,185,129,0.35) !important; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-child(4):has(input:checked){ background: rgba(245,158,11,0.12) !important; outline-color: rgba(245,158,11,0.40) !important; }

</style>
    """, unsafe_allow_html=True)

# =========================
# Utils
# =========================

def parse_menu_sections(menu_text: str):
    """Split menu text into sections by headings like '【上半身】' etc."""
    t = (menu_text or "").strip()
    if not t:
        return []
    # Normalize newlines
    t = t.replace('\r\n','\n').replace('\r','\n')
    # If it already contains bracket headings, split on them
    parts = re.split(r'(?=^【[^】]{1,20}】\s*$)', t, flags=re.M)
    sections = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^【([^】]{1,20})】\s*\n?(.*)$', p, flags=re.S)
        if m:
            title = m.group(1).strip()
            body = m.group(2).strip()
            sections.append((title, body))
        else:
            sections.append(("全体", p))
    return sections

def render_menu_blocks(menu_text: str):
    """Render larger, copy-friendly menu blocks."""
    st.markdown("""<style>
    /* Make textareas easier to read */
    div[data-testid="stTextArea"] textarea { font-size: 16px !important; line-height: 1.5 !important; }
    </style>""", unsafe_allow_html=True)

    st.markdown("#### 生成メニュー（見やすく／コピーしやすく）")
    full = (menu_text or "").strip()
    if not full:
        st.info("メニューがまだ生成されていません。")
        return

    # Copy-all
    st.text_area("（全文）", value=full, height=260, key="tr_menu_text_area")
    copy_button("メニューをコピー（全文）", full, key="copy_tr_menu_btn_all")
    st.caption("コピーしたら、スマホのメモやLINEの『自分だけのトーク』に保存しておくのがおすすめです。")

    secs = parse_menu_sections(full)
    if len(secs) <= 1:
        return

    for i, (title, body) in enumerate(secs, start=1):
        with st.expander(f"{title}（開く）", expanded=(title in ["上半身","下半身","体幹","４週間の進め方","4週間の進め方"])): 
            txt = f"【{title}】\n{body}".strip()
            st.text_area("", value=txt, height=220, key=f"tr_menu_sec_{i}")
            copy_button(f"{title}をコピー", txt, key=f"copy_tr_menu_sec_{i}")


def now_jst():
    return datetime.now(TZ)

def iso(dt):
    return dt.astimezone(TZ).isoformat()

def copy_button(label: str, text_to_copy: str, key: str):
    """One-click copy to clipboard (Streamlit)."""
    # Escape for JS template literal
    t = (text_to_copy or "")
    t = t.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
<button id='{key}' style='padding:0.45rem 0.8rem;border:1px solid #ddd;border-radius:10px;background:#fff;cursor:pointer;'>
  {label}
</button>
<script>
const btn = document.getElementById('{key}');
btn.addEventListener('click', async () => {{
  try {{
    await navigator.clipboard.writeText(`{t}`);
    const prev = btn.innerText;
    btn.innerText = 'コピーしました';
    setTimeout(()=>{{ btn.innerText = prev; }}, 1200);
  }} catch (e) {{
    btn.innerText = 'コピー失敗';
    setTimeout(()=>{{ btn.innerText = '{label}'; }}, 1500);
  }}
}});
</script>
"""
    components.html(html, height=55)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def years_between(d1, d2) -> float:
    return (pd.Timestamp(d2) - pd.Timestamp(d1)).days / 365.25

def nz(x):
    try:
        v = float(x)
        return None if v == 0 else v
    except Exception:
        return None


def _parse_date_maybe(v):
    if v is None:
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(v, fmt).date()
            except Exception:
                pass
    return None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================
# Login (test)
# =========================
def users_db():
    conn = sqlite3.connect(USERS_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_users_db():
    conn = users_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            pw_salt TEXT NOT NULL,
            pw_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def verify_user(username: str, password: str) -> bool:
    u = (username or "").strip()
    conn = users_db()
    row = conn.execute("SELECT pw_salt, pw_hash FROM users WHERE username=?", (u,)).fetchone()
    conn.close()
    if not row:
        return False
    salt, pw_hash = row
    return _hash_pw(password, salt) == pw_hash

def create_user(username: str, password: str) -> str | None:
    u = (username or "").strip()
    if not u or not password:
        return "IDとパスワードは必須です。"
    conn = users_db()
    exists = conn.execute("SELECT 1 FROM users WHERE username=?", (u,)).fetchone()
    if exists:
        conn.close()
        return "そのIDはすでに使われています。"
    salt = secrets.token_hex(16)
    pw_hash = _hash_pw(password, salt)
    conn.execute("INSERT INTO users(username, pw_salt, pw_hash, created_at) VALUES(?,?,?,?)",
                 (u, salt, pw_hash, iso(now_jst())))
    conn.commit()
    conn.close()
    return None

def login_panel() -> str | None:
    st.markdown("## ログイン（テスト段階）")
    t = st.tabs(["ログイン", "初回登録"])
    with t[0]:
        u = st.text_input("ID", key="login_id")
        p = st.text_input("パスワード", type="password", key="login_pw")
        if st.button("ログイン", type="primary"):
            if verify_user(u, p):
                st.session_state["user"] = u.strip()
                st.rerun()
            else:
                st.error("IDまたはパスワードが違います。")
    with t[1]:
        u = st.text_input("新規ID", key="reg_id")
        p1 = st.text_input("新規パスワード", type="password", key="reg_pw1")
        p2 = st.text_input("新規パスワード（確認）", type="password", key="reg_pw2")
        if st.button("登録する", type="primary"):
            if p1 != p2:
                st.error("パスワードが一致しません。")
            else:
                err = create_user(u, p1)
                if err:
                    st.error(err)
                else:
                    st.success("登録しました。ログインしてください。")
    return st.session_state.get("user")

# =========================
# Data DB
# =========================
def data_db():
    conn = sqlite3.connect(DATA_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_data_db():
    conn = data_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots(
            code_hash TEXT NOT NULL,
            kind TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(code_hash, kind)
        );
        CREATE TABLE IF NOT EXISTS records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            code_hash TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_records_codehash ON records(code_hash);
    """)
    conn.commit()
    conn.close()

def save_snapshot(code_hash: str, kind: str, payload: dict):
    conn = data_db()
    conn.execute(
        "INSERT INTO snapshots(code_hash, kind, updated_at, payload_json) VALUES(?,?,?,?) "
        "ON CONFLICT(code_hash, kind) DO UPDATE SET updated_at=excluded.updated_at, payload_json=excluded.payload_json",
        (code_hash, kind, iso(now_jst()), json.dumps(payload, ensure_ascii=False, default=str))
    )
    conn.commit()
    conn.close()

def load_snapshot(code_hash: str, kind: str):
    conn = data_db()
    row = conn.execute("SELECT payload_json FROM snapshots WHERE code_hash=? AND kind=?", (code_hash, kind)).fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None

def save_record(code_hash: str, kind: str, payload: dict, result: dict):
    conn = data_db()
    conn.execute(
        "INSERT INTO records(created_at, code_hash, kind, payload_json, result_json) VALUES(?,?,?,?,?)",
        (iso(now_jst()), code_hash, kind,
         json.dumps(payload, ensure_ascii=False, default=str),
         json.dumps(result, ensure_ascii=False, default=str))
    )
    conn.commit()
    conn.close()

def load_records(code_hash: str, limit: int = 200):
    conn = data_db()
    rows = conn.execute(
        "SELECT id, created_at, kind, payload_json, result_json FROM records WHERE code_hash=? ORDER BY id DESC LIMIT ?",
        (code_hash, limit)
    ).fetchall()
    conn.close()
    out = []
    for rid, created_at, kind, p, r in rows:
        try:
            out.append({
                "id": rid,
                "created_at": created_at,
                "kind": kind,
                "payload": json.loads(p),
                "result": json.loads(r),
            })
        except Exception:
            pass
    return out


def delete_snapshot(code_hash: str, kind: str) -> None:
    conn = sqlite3.connect(DATA_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM snapshots WHERE code_hash=? AND kind=?", (code_hash, kind))
    conn.commit()
    conn.close()

def delete_record_by_id(record_id: int) -> None:
    conn = sqlite3.connect(DATA_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM records WHERE id=?", (int(record_id),))
    conn.commit()
    conn.close()

def delete_latest_record(code_hash: str, kind: str) -> bool:
    conn = sqlite3.connect(DATA_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM records WHERE code_hash=? AND kind=? ORDER BY id DESC LIMIT 1", (code_hash, kind))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False
    rid = int(row[0])
    cur.execute("DELETE FROM records WHERE id=?", (rid,))
    conn.commit()
    conn.close()
    return True
def auto_fill_from_latest_records(code_hash: str):
    """基本情報入力後に、最新の保存記録をフォームに自動反映（初回のみ）"""
    if st.session_state.get("_auto_filled", False):
        return
    rows = load_records(code_hash, limit=200)
    if not rows:
        st.session_state["_auto_filled"] = True
        return

    # 最新の身長結果
    for r in rows:
        if r.get("kind") == "height_result":
            pl = r.get("payload") or {}
            # date fields may be string; keep as-is, date_input側でparse
            for k_map in [
                ("h_desired","desired_cm"),
                ("h_alp","alp"), ("h_ba","ba"), ("h_igf1","igf1"),
                ("h_t","testosterone"), ("h_e2","estradiol"),
                ("h_y1","h_y1"), ("h_y2","h_y2"), ("h_y3","h_y3"),
                ("h_w1","w_y1"), ("h_w2","w_y2"), ("h_w3","w_y3"),
                ("h_date_y1","date_y1"), ("h_date_y2","date_y2"), ("h_date_y3","date_y3"),
            ]:
                ui, pk = k_map
                if pk in pl and pl[pk] not in (None, "") and ui not in st.session_state:
                    st.session_state[ui] = pl[pk]
            break

    # 最新の貧血結果（未服用保存）
    for r in rows:
        if r.get("kind") in ("sports_anemia","anemia_baseline"):
            pl = r.get("payload") or {}
            for ui, pk in [("sa_hb","hb"), ("sa_ferr","ferritin"), ("sa_fe","fe"), ("sa_tibc","tibc"), ("sa_tsat","tsat")]:
                if pk in pl and pl[pk] not in (None, "") and ui not in st.session_state:
                    st.session_state[ui] = pl[pk]
            break

    st.session_state["_auto_filled"] = True



# =========================
# Per-user persistence (basic info & training)
# =========================
BASIC_INFO_KEYS = ["name_kana","sex_code","dob","sport"]

def save_basic_info_snapshot(code_hash: str):
    payload = {k: st.session_state.get(k) for k in BASIC_INFO_KEYS}
    if isinstance(payload.get("dob"), date):
        payload["dob"] = payload["dob"].isoformat()
    save_snapshot(code_hash, "basic_info", payload)

def load_basic_info_snapshot(code_hash: str) -> bool:
    pl = load_snapshot(code_hash, "basic_info")
    if not pl:
        return False
    if isinstance(pl.get("dob"), str):
        try:
            y,m,d = [int(x) for x in pl["dob"].split("-")]
            pl["dob"] = date(y,m,d)
        except Exception:
            pass
    for k in BASIC_INFO_KEYS:
        if k in pl and pl[k] is not None:
            st.session_state[k] = pl[k]
    # derive age
    if st.session_state.get("dob"):
        today = now_jst().date()
        st.session_state["age_years"] = float(years_between(st.session_state["dob"], today))
    return True

TRAINING_KEYS = ["tr_date","tr_type","tr_duration","tr_rpe","tr_focus","tr_notes"]

def save_training_latest(code_hash: str):
    payload = {k: st.session_state.get(k) for k in TRAINING_KEYS}
    if isinstance(payload.get("tr_date"), date):
        payload["tr_date"] = payload["tr_date"].isoformat()
    save_snapshot(code_hash, "training_latest", payload)
    save_record(code_hash, "training_log", payload, {"summary":"training_log"})

def load_training_latest(code_hash: str) -> bool:
    pl = load_snapshot(code_hash, "training_latest")
    if not pl:
        return False
    if isinstance(pl.get("tr_date"), str):
        try:
            y,m,d = [int(x) for x in pl["tr_date"].split("-")]
            pl["tr_date"] = date(y, m, d)
        except Exception:
            pass
    for k in TRAINING_KEYS:
        if k in pl and pl[k] is not None:
            st.session_state[k] = pl[k]
    return True

# =========================
# Shared demographics
# =========================

def _set_if_empty(k, v):
    if v is None or v == "":
        return
    if k not in st.session_state or st.session_state.get(k) in (None, "", 0, 0.0):
        st.session_state[k] = v

def auto_fill_latest_all_tabs(code_hash: str):
    """基本情報入力後に、保存済み最新データを各タブの入力欄へ自動反映（初回のみ）"""
    if st.session_state.get("_auto_filled_all", False):
        return
    # 必須：生年月日が入っているときだけ
    if not st.session_state.get("dob"):
        return

    # まず snapshots（下書き）を優先
    for kind, keys in [
        ("height_draft", ["h_desired","h_date_y1","h_date_y2","h_date_y3","h_y1","h_y2","h_y3","h_w1","h_w2","h_w3","h_alp","h_ba","h_igf1","h_t","h_e2"]),
        ("anemia_draft", ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end_test_type"]),
        ("meal_draft", ["meal_goal","meal_intensity","meal_weight","b_c","b_p","b_v","l_c","l_p","l_v","d_c","d_p","d_v"]),
    ]:
        try:
            pl = load_snapshot(code_hash, kind)
        except Exception:
            pl = None
        if pl:
            for k in keys:
                _set_if_empty(k, pl.get(k))

    # 次に records（結果）から
    rows = load_records(code_hash, limit=300)
    # Height
    for r in rows:
        if r.get("kind") == "height_result":
            pl = r.get("payload") or {}
            for ui, pk in [
                ("h_desired","desired_cm"),
                ("h_alp","alp"), ("h_ba","ba"), ("h_igf1","igf1"),
                ("h_t","testosterone"), ("h_e2","estradiol"),
                ("h_y1","h_y1"), ("h_y2","h_y2"), ("h_y3","h_y3"),
                ("h_w1","w_y1"), ("h_w2","w_y2"), ("h_w3","w_y3"),
                ("h_date_y1","date_y1"), ("h_date_y2","date_y2"), ("h_date_y3","date_y3"),
            ]:
                _set_if_empty(ui, pl.get(pk))
            break
    # Anemia
    for r in rows:
        if r.get("kind") in ("sports_anemia","anemia_baseline"):
            pl = r.get("payload") or {}
            for ui, pk in [("sa_hb","hb"),("sa_ferr","ferritin"),("sa_fe","fe"),("sa_tibc","tibc"),("sa_tsat","tsat")]:
                _set_if_empty(ui, pl.get(pk))
            break
    # Meal latest
    for r in rows:
        if r.get("kind") == "meal_day":
            pl = r.get("payload") or {}
            _set_if_empty("meal_goal", pl.get("goal"))
            _set_if_empty("meal_intensity", pl.get("intensity"))
            _set_if_empty("meal_weight", pl.get("weight"))
            break

    st.session_state["_auto_filled_all"] = True

def shared_demographics():
    st.markdown("### 基本情報")

    today = now_jst().date()
    dob_min = date(1900, 1, 1)

    c0, c1, c2, c3 = st.columns([1.4, 1.0, 1.2, 1.4])
    with c0:
        name_kana = st.text_input("名前（カタカナ）", value=st.session_state.get("name_kana",""), key="base_name_kana")
        if name_kana:
            st.session_state["name_kana"] = name_kana.strip()

    with c1:
        sex_ui = st.selectbox("性別", ["M（男）", "F（女）"],
                              index=0 if st.session_state.get("sex_code","M")=="M" else 1,
                              key="base_sex")

    with c2:
        dob_val = st.session_state.get("dob")
        dob = st.date_input("生年月日",
                            value=dob_val if isinstance(dob_val, date) else today,
                            min_value=dob_min, max_value=today,
                            key="base_dob")

    with c3:
        sport = st.selectbox("競技", SPORTS,
                             index=SPORTS.index(st.session_state.get("sport", SPORTS[0])),
                             key="base_sport")

    st.session_state["sex_code"] = "M" if sex_ui.startswith("M") else "F"
    st.session_state["sport"] = sport
    if dob:
        st.session_state["dob"] = dob
        st.session_state["age_years"] = float(years_between(dob, today))
        st.caption(f"年齢（概算）：{st.session_state['age_years']:.1f}歳")
    st.divider()
    c4, c5 = st.columns([1,1])
    with c4:
        if st.button("基本情報を保存", key="basic_save"):
            try:
                save_basic_info_snapshot(sha256_hex(st.session_state.get("user","")))
                st.success("基本情報を保存しました。")
            except Exception as e:
                st.error(f"保存に失敗: {e}")
    with c5:
        if st.button("基本情報を読み込み", key="basic_load"):
            try:
                ok = load_basic_info_snapshot(sha256_hex(st.session_state.get("user","")))
                if ok:
                    st.success("基本情報を読み込みました。")
                    st.rerun()
                else:
                    st.info("保存済みの基本情報がありません。")
            except Exception as e:
                st.error(f"読み込みに失敗: {e}")


# =========================
# Curve helpers
# =========================
@st.cache_data

def load_curve():
    df = pd.read_csv("boys_height_curve.csv")
    df = df.dropna(subset=["age"]).sort_values("age")
    return df

def interp_curve(df, col: str, age: np.ndarray):
    ages = df["age"].to_numpy(dtype=float)
    ys = df[col].to_numpy(dtype=float)
    aa = np.clip(age, ages.min(), ages.max())
    return np.interp(aa, ages, ys)

def fit_shift_offset(df, base_col: str, pts_age, pts_h, delta_shift: float):
    s = float(clamp(delta_shift, -2.0, 2.0))
    res = []
    for a, h in zip(pts_age, pts_h):
        y = interp_curve(df, base_col, np.array([a + s]))[0]
        res.append(h - y)
    b = float(np.median(res)) if res else 0.0
    return s, b

def plot_min_max_curves(df, s_min, b_min, s_max, b_max, pts_age, pts_h):
    ages = df["age"].to_numpy(dtype=float)
    y_min = interp_curve(df, "late", ages + s_min) + b_min
    y_max = interp_curve(df, "early", ages + s_max) + b_max
    chart_df = pd.DataFrame({
        "age": np.concatenate([ages, ages]),
        "height_cm": np.concatenate([y_max, y_min]),
        "curve": (["最大予測カーブ"]*len(ages)) + (["最小予測カーブ"]*len(ages))
    })
    line = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("age:Q", title="年齢（年）"),
        y=alt.Y("height_cm:Q", title="身長（cm）", scale=alt.Scale(domain=[Y_AXIS_LO, Y_AXIS_HI])),
        color=alt.Color("curve:N", scale=alt.Scale(domain=["最大予測カーブ","最小予測カーブ"], range=["red","blue"]))
    ).properties(height=320)
    if pts_age and pts_h:
        pts = alt.Chart(pd.DataFrame({"age": pts_age, "height_cm": pts_h})).mark_point(size=80).encode(x="age:Q", y="height_cm:Q")
        st.altair_chart(line+pts, use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)

# =========================
# IGF-1
# =========================
def igf1_range_for_age(sex_code: str, age_years: float):
    if age_years < 3 or age_years > 20:
        return None
    table = IGF1_RANGES["M" if sex_code=="M" else "F"]
    a0 = int(np.floor(age_years)); a1 = int(np.ceil(age_years))
    a0 = max(3, min(20, a0)); a1 = max(3, min(20, a1))
    lo0, hi0 = table[a0]; lo1, hi1 = table[a1]
    if a0 == a1:
        return float(lo0), float(hi0)
    t = (age_years - a0) / (a1 - a0)
    return float(lo0 + (lo1-lo0)*t), float(hi0 + (hi1-hi0)*t)

def igf1_classify(sex_code: str, age_years: float, igf1_value: float):
    rng = igf1_range_for_age(sex_code, age_years)
    if rng is None or igf1_value is None or igf1_value <= 0:
        return "不明", None, False
    lo, hi = rng
    if igf1_value < lo:
        return "低い", (lo, hi), False
    if igf1_value > hi:
        return "高い", (lo, hi), False
    low_normal = (igf1_value <= lo + 0.2*(hi-lo))
    return ("正常（下限寄り）" if low_normal else "正常"), (lo, hi), low_normal

# =========================
# OpenAI helpers
# =========================
def openai_client():
    k = (OPENAI_API_KEY or "").strip()
    if not k or k == "sk-REPLACE_ME":
        return None, "OPENAI_API_KEY を設定してください。"
    try:
        from openai import OpenAI
        return OpenAI(api_key=k), None
    except Exception as e:
        return None, str(e)

def analyze_meal_photo(img_bytes: bytes, meal_type: str):
    client, err = openai_client()
    if err:
        return None, err
    prompt = f"""これは{meal_type}の食事写真です。
食品群の量感を Aレベル（少/普/多）で推定し、JSONのみで返してください。
キー: carb, protein, veg, fat, fried_or_oily(true/false), dairy(true/false), fruit(true/false), confidence(0-1)
"""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role":"user","content":[
                {"type":"input_text","text":prompt},
                {"type":"input_image","image_url":"data:image/jpeg;base64,"+img_b64},
            ]}],
            text={"format":{"type":"json_object"}},
            temperature=0
        )
        txt = (resp.output_text or "").strip()
        if not txt:
            return None, "解析結果が空でした。"
        data = json.loads(txt)
        def norm(v): return v if v in ["少","普","多"] else "普"
        out = {
            "carb": norm(data.get("carb","普")),
            "protein": norm(data.get("protein","普")),
            "veg": norm(data.get("veg","普")),
            "fat": norm(data.get("fat","普")),
            "fried_or_oily": bool(data.get("fried_or_oily", False)),
            "dairy": bool(data.get("dairy", False)),
            "fruit": bool(data.get("fruit", False)),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
        }
        return out, None
    except Exception as e:
        return None, str(e)

def ai_text(system: str, user: str):
    client, err = openai_client()
    if err:
        return None, err
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2
        )
        return (resp.output_text or "").strip(), None
    except Exception as e:
        return None, str(e)

# =========================
# Pages
# =========================
def classify_type(delta: float):
    if delta >= TYPE_EARLY_DELTA:
        return "precocious", "早熟型"
    if delta <= TYPE_DELAY_DELTA:
        return "delayed", "遅発型"
    return "normal", "正常"

def height_page(code_hash: str):
    st.subheader("身長予測")
    # load/save buttons adjacent
    if st.button("記入データ読込", key="h_load_top"):
        payload = load_snapshot(code_hash, "height_draft")
        if payload:
            for k, v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if st.button("保存", key="h_save_top"):
        keys = ["h_desired","h_date_y1","h_date_y2","h_date_y3","h_y1","h_y2","h_y3","h_w1","h_w2","h_w3","h_alp","h_ba","h_igf1","h_t","h_e2"]
        save_snapshot(code_hash, "height_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")

    dob = st.session_state.get("dob")
    age = float(st.session_state.get("age_years", 0.0) or 0.0)
    sex_code = st.session_state.get("sex_code","M")
    if not dob or age <= 0:
        st.error("基本情報（生年月日）を入力してください。")
        return

    # default desired 175
    if ("h_desired" not in st.session_state) or (float(st.session_state.get("h_desired") or 0) <= 100.0):
        st.session_state["h_desired"] = 175.0
    desired = st.number_input("将来なりたい身長（cm）", 100.0, 230.0, step=0.1, key="h_desired")

    c = st.columns(3)
    ba = c[0].number_input("骨年齢（年）", 0.0, 25.0, step=0.1, key="h_ba")
    alp = c[1].number_input("ALP", 0.0, 5000.0, step=1.0, key="h_alp")
    igf1 = c[2].number_input("ソマトメジンC（IGF-1）", 0.0, 2000.0, step=1.0, key="h_igf1")
    testosterone = st.number_input("テストステロン（任意）", 0.0, 3000.0, step=1.0, key="h_t")
    estradiol = st.number_input("エストラジオール(E2)（任意）", 0.0, 2000.0, step=1.0, key="h_e2")

    igf_label, igf_rng, low_normal = igf1_classify(sex_code, age, nz(igf1))
    if igf_rng is not None:
        st.caption(f"IGF-1（自動判定）：{igf_label} / 基準 {igf_rng[0]:.0f}〜{igf_rng[1]:.0f}")

    df = load_curve()
    st.markdown("#### 直近3年（測定日・身長・体重）")
    col1, col2, col3 = st.columns(3)
    v = _parse_date_maybe(st.session_state.get("h_date_y1"))
    if v is not None:
        st.session_state["h_date_y1"] = v
    d1 = col1.date_input("測定日 3年前（任意）", key="h_date_y1")
    h1 = col1.number_input("身長 3年前(cm)", 0.0, 230.0, 0.0, 0.1, key="h_y1")
    w1 = col1.number_input("体重 3年前(kg)", 0.0, 200.0, 0.0, 0.1, key="h_w1")
    v = _parse_date_maybe(st.session_state.get("h_date_y2"))
    if v is not None:
        st.session_state["h_date_y2"] = v
    d2 = col2.date_input("測定日 2年前（任意）", key="h_date_y2")
    h2 = col2.number_input("身長 2年前(cm)", 0.0, 230.0, 0.0, 0.1, key="h_y2")
    w2 = col2.number_input("体重 2年前(kg)", 0.0, 200.0, 0.0, 0.1, key="h_w2")
    v = _parse_date_maybe(st.session_state.get("h_date_y3"))
    if v is not None:
        st.session_state["h_date_y3"] = v
    d3 = col3.date_input("測定日 最新（任意）", key="h_date_y3")
    h3 = col3.number_input("身長 最新(cm)", 0.0, 230.0, 0.0, 0.1, key="h_y3")
    w3 = col3.number_input("体重 最新(kg)", 0.0, 200.0, 0.0, 0.1, key="h_w3")

    pts_age, pts_h = [], []
    if nz(h1): pts_age.append(max(age-2,0)); pts_h.append(float(h1))
    if nz(h2): pts_age.append(max(age-1,0)); pts_h.append(float(h2))
    if nz(h3): pts_age.append(max(age,0)); pts_h.append(float(h3))
    if not pts_h:
        st.warning("身長データを入れてください。")
        return

    pred = pts_h[-1]
    type_code = "normal"
    type_jp = "正常"
    if nz(alp) is not None and float(alp) <= ALP_STOP_THRESHOLD:
        type_code, type_jp = "stop", "停止扱い"
        st.warning("ALP低値のため成長停止扱い")
    elif nz(ba) is not None and float(ba) > BA_CLOSED_THRESHOLD:
        type_code, type_jp = "closed", "閉鎖扱い"
        st.warning("骨年齢が進んでいるため閉鎖扱い")
    else:
        delta = float(ba) - age if nz(ba) is not None else 0.0
        type_code, type_jp = classify_type(delta)
        s_early,b_early = fit_shift_offset(df,"early",pts_age,pts_h,delta)
        s_late,b_late = fit_shift_offset(df,"late",pts_age,pts_h,delta)
        adult_age = float(df["age"].max())
        pred_early = interp_curve(df,"early",np.array([adult_age+s_early]))[0] + b_early
        pred_late  = interp_curve(df,"late", np.array([adult_age+s_late]))[0] + b_late
        pred = pred_early if type_code=="precocious" else (pred_late if type_code=="delayed" else pred_early)
        st.caption(f"予測最終身長レンジ：最大 {max(pred_early,pred_late):.1f} / 最小 {min(pred_early,pred_late):.1f} cm")
        if pred_early >= pred_late:
            plot_min_max_curves(df, s_late,b_late, s_early,b_early, pts_age,pts_h)
        else:
            plot_min_max_curves(df, s_early,b_early, s_late,b_late, pts_age,pts_h)
    st.success(f"推定最終身長：{pred:.1f} cm")
    st.write(f"将来なりたい身長との差：{(desired - pred):+.1f} cm")

    # feedback and consult
    reasons = []
    if igf_label == "低い" or low_normal:
        reasons.append("ソマトメジンC（IGF-1）が下限寄り/低値")
    if type_code == "precocious":
        reasons.append("早熟傾向（骨年齢が進んでいる可能性）")
    if reasons:
        st.info("以下の理由により、スポーツドクターに相談することで新たな選択肢が広がる可能性があります。")
        for r in reasons:
            st.write("• " + r)
    st.link_button("成長に関する相談（受診予約）", RESERVE_URL)

    
    st.divider()
    if st.button("記入データ読込", key="h_load_bottom"):
        payload = load_snapshot(code_hash, "height_draft")
        if payload:
            for k, v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if st.button("保存", key="h_save_bottom"):
        keys = ["h_desired","h_date_y1","h_date_y2","h_date_y3","h_y1","h_y2","h_y3","h_w1","h_w2","h_w3","h_alp","h_ba","h_igf1","h_t","h_e2"]
        save_snapshot(code_hash, "height_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")

    if st.button("結果保存（身長）", key="h_result_save"):
        save_record(code_hash, "height_result", {
            "desired_cm": desired, "alp": alp, "ba": ba, "igf1": igf1,
            "testosterone": testosterone, "estradiol": estradiol,
            "date_y1": d1, "date_y2": d2, "date_y3": d3,
            "h_y1": h1, "h_y2": h2, "h_y3": h3,
            "w_y1": w1, "w_y2": w2, "w_y3": w3,
            "pred_cm": pred, "type": type_jp
        }, {"summary":"height_result"})
        st.success("保存しました。")

def tsat_from_fe_tibc(fe, tibc):
    if fe is None or tibc is None or tibc <= 0:
        return None
    return 100.0 * fe / tibc

def render_riona_output(out: dict):
    p12 = out.get("12w") or {}
    p24 = out.get("24w") or {}
    st.markdown("### 12週予測")
    c = st.columns(4)
    c[0].metric("Hb", f"{p12.get('Hb','')}")
    c[1].metric("Fe", f"{p12.get('Fe','')}")
    c[2].metric("Ferritin", f"{p12.get('Ferritin','')}")
    c[3].metric("TSAT", f"{p12.get('TSAT','')}")
    if p12.get("alerts"):
        st.warning(" / ".join(p12["alerts"]))
    st.markdown("### 24週予測")
    c = st.columns(4)
    c[0].metric("Hb", f"{p24.get('Hb','')}")
    c[1].metric("Fe", f"{p24.get('Fe','')}")
    c[2].metric("Ferritin", f"{p24.get('Ferritin','')}")
    c[3].metric("TSAT", f"{p24.get('TSAT','')}")
    if p24.get("alerts"):
        st.warning(" / ".join(p24["alerts"]))


def estimate_endurance_gain(test_kind: str, baseline_value: float, hb_now: float, hb_pred: float, ferr_now: float | None, ferr_pred: float | None):
    """
    Very conservative heuristic for demo:
      - assume aerobic capacity roughly tracks Hb improvement and iron repletion.
      - returns (pred_value, pct_gain)
    """
    if baseline_value <= 0 or hb_now <= 0 or hb_pred <= 0:
        return None, None

    dhb = hb_pred - hb_now
    if dhb <= 0:
        return baseline_value, 0.0

    # base % gain from Hb change (scaled down)
    pct = (dhb / hb_now) * 0.6  # e.g. Hb +10% -> +6%
    # bonus if ferritin corrected to >=30
    if ferr_now is not None and ferr_pred is not None:
        if ferr_now < 30.0 and ferr_pred >= 30.0:
            pct += 0.02

    # cap
    pct = max(0.0, min(0.12, pct))  # 0-12%

    # if already very high baseline, dampen (ceiling effect)
    if test_kind == "yoyo" and baseline_value >= 2000:
        pct *= 0.6
    if test_kind == "shuttle" and baseline_value >= 130:
        pct *= 0.6

    return baseline_value * (1.0 + pct), pct

def anemia_page(code_hash: str):
    hb_v = ferr_v = fe_v = tibc_v = tsat_val = None
    st.subheader("貧血・リオナ")
    if st.button("記入データ読込", key="a_load_top"):
        payload = load_snapshot(code_hash, "anemia_draft")
        if payload:
            for k, v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if st.button("保存", key="a_save_top"):
        keys = ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end_test_type"]
        save_snapshot(code_hash, "anemia_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")

    sex_code = st.session_state.get("sex_code","M")
    age_default = float(st.session_state.get("age_years", 15.0) or 15.0)
    c1,c2,c3,c4,c5 = st.columns(5)
    hb = c1.number_input("Hb", 0.0, 20.0, 0.0, 0.1, key="sa_hb")
    ferr = c2.number_input("Ferritin", 0.0, 1000.0, 0.0, 1.0, key="sa_ferr")
    fe = c3.number_input("Fe", 0.0, 500.0, 0.0, 1.0, key="sa_fe")
    tibc = c4.number_input("TIBC", 0.0, 600.0, 0.0, 1.0, key="sa_tibc")
    tsat_override = c5.number_input("TSAT上書き(0=自動)", 0.0, 100.0, 0.0, 0.1, key="sa_tsat")

    st.markdown("#### 持久力テスト（任意）")
    end_test_type = st.selectbox("入力するテスト", ["シャトルラン（回数）", "Yo-Yo（距離m）"], index=0, key="end_test_type")
    end_current = st.number_input("現在の記録（回数 or 距離）", min_value=0.0, max_value=99999.0, value=float(st.session_state.get("end_current", 0.0) or 0.0), step=1.0, key="end_current")
    st.caption("※入力は任意。入力すると、Hb改善に伴う伸びを参考推定します（個人差あり）。")
    if st.button("結果保存（持久力）", key="save_endurance_baseline"):
        save_record(code_hash, "endurance_baseline", {"test": st.session_state.get("end_test_type",""), "current": float(st.session_state.get("end_current",0.0) or 0.0), "hb": float(hb_v or 0.0), "ferritin": float(ferr_v or 0.0), "tsat": float(tsat_val or 0.0)}, {"summary":"endurance_baseline"})
        st.success("保存しました。")
    hb_v,ferr_v,fe_v,tibc_v = nz(hb),nz(ferr),nz(fe),nz(tibc)
    tsat_val = tsat_from_fe_tibc(fe_v,tibc_v) if tsat_override==0 else float(tsat_override)
    taking = st.radio("リオナ服用中？", ["いいえ","はい"], horizontal=True, key="sa_riona") == "はい"

    if not taking:
        hb_low = 13.0 if sex_code=="M" else 12.0
        iron_def = (ferr_v is not None and ferr_v < 30.0) or (tsat_val is not None and tsat_val < 16.0)
        if hb_v is not None:
            if hb_v < hb_low and iron_def:
                st.error("鉄剤など医学的なフォローができる可能性がありますので、下記より受診をお勧めします")
            elif hb_v >= hb_low and iron_def:
                st.warning("潜在性鉄欠乏の可能性があります。必要なら受診をご検討ください。")
        st.link_button("スポーツ貧血の相談（受診予約）", RESERVE_URL)
        if st.button("結果保存（貧血）", key="a_result_save"):
            save_record(code_hash, "sports_anemia", {"hb":hb_v,"ferritin":ferr_v,"fe":fe_v,"tibc":tibc_v,"tsat":tsat_val}, {"summary":"sports_anemia"})
            st.success("保存しました。")
        return

    
    st.divider()
    if st.button("記入データ読込", key="a_load_bottom"):
        payload = load_snapshot(code_hash, "anemia_draft")
        if payload:
            for k, v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if st.button("保存", key="a_save_bottom"):
        keys = ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end_test_type"]
        save_snapshot(code_hash, "anemia_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")

    dose = st.number_input("用量 (mg/day)", value=500, step=50, key="r_dose")
    adherence = st.slider("服薬率", 0.0, 1.0, 0.9, 0.05, key="r_adher")
    if st.button("改善予測を表示", type="primary", key="r_pred_btn"):
        if tsat_val is None:
            st.error("TSATの計算に必要なFeとTIBCを入力してください。")
            return
        init_db()
        labs = Labs(hb=float(hb_v or 0), fe=float(fe_v or 0), ferritin=float(ferr_v or 0), tibc=float(tibc_v or 0), tsat=None)
        ctx = Ctx(dose_mg_day=int(dose), adherence=float(adherence), bleed=0.0, inflam=0.0)
        case_id, out = register_case(labs, ctx, note="sports_anemia", external_id="")
        st.session_state["r_case_id"] = case_id
        render_riona_output(out)

        # ---- 持久力テストの伸び（参考推定）----
        end_current = float(st.session_state.get("end_current", 0.0) or 0.0)
        end_test_type = st.session_state.get("end_test_type", "シャトルラン（回数）")
        hb0 = float(hb_v or 0.0)
        hb12 = float((out.get("12w") or {}).get("Hb", hb0) or hb0)
        hb24 = float((out.get("24w") or {}).get("Hb", hb0) or hb0)

        def predict_endurance(cur, hb_from, hb_to):
            if cur <= 0 or hb_from <= 0 or hb_to <= 0:
                return None, None
            dhb = max(0.0, hb_to - hb_from)
            pct = min(0.15, 0.03 * dhb)  # 仮係数（後で論文係数へ差替）
            return cur * (1.0 + pct), pct

        if end_current > 0 and hb0 > 0:
            p12, pct12 = predict_endurance(end_current, hb0, hb12)
            p24, pct24 = predict_endurance(end_current, hb0, hb24)
            st.markdown("### Hb改善に伴う持久力の伸び（参考推定）")
            st.caption(f"入力テスト：{end_test_type} / 現在：{end_current:.0f}")
            if p12 is not None:
                st.write(f"12週：{p12:.0f}（+{pct12*100:.1f}%）")
            if p24 is not None:
                st.write(f"24週：{p24:.0f}（+{pct24*100:.1f}%）")
            st.caption("※参考推定（個人差あり）。論文係数に基づく推定へ差し替え可能です。")
        else:
            st.caption("持久力テストの記録（回数/距離）を入れると、Hb改善に伴う伸びを表示します。")
        if st.button("予測を保存（リオナ）", key="save_riona_pred"):
            save_record(code_hash, "riona_pred", {"case_id": case_id, "dose": int(dose), "adherence": float(adherence)}, {"summary":"riona_pred", "out": out})
            st.success("保存しました。")
        st.info("実際に血液検査を希望の方は、下のボタンから予約をお願いします。")
        st.link_button("血液検査の実評価を希望（受診予約）", RESERVE_URL)

        st.divider()
        st.markdown("### 12週/24週 実測を入力（補正して再計算）")
        st.caption("通常はID入力不要です（直前の予測IDを自動使用）。別の検査結果を入力する場合のみIDを入力してください。")

        default_id = str(st.session_state.get("r_case_id","") or "")
        identifier = st.text_input("ID（通常は空欄でOK）", value="", key="r_follow_id")
        case_id_use = identifier.strip() or default_id

        horizon = st.selectbox("時点", [12,24], format_func=lambda x: "12週" if x==12 else "24週", key="r_follow_h")
        f1,f2,f3,f4 = st.columns(4)
        hb_m = f1.number_input("Hb 実測", value=0.0, step=0.1, key="r_hb_m")
        fe_m = f2.number_input("Fe 実測", value=0.0, step=1.0, key="r_fe_m")
        ferr_m = f3.number_input("Ferritin 実測", value=0.0, step=1.0, key="r_ferr_m")
        tibc_m = f4.number_input("TIBC 実測", value=0.0, step=1.0, key="r_tibc_m")

        if st.button("実測を保存して再計算", key="r_follow_save"):
            if not case_id_use:
                st.error("予測を先に実行してください。")
            else:
                cid = resolve_case_id(case_id_use)
                if not cid:
                    st.error("症例が見つかりません。")
                else:
                    res = add_followup(cid, horizon_weeks=int(horizon), hb=float(hb_m), fe=float(fe_m), ferritin=float(ferr_m), tibc=float(tibc_m))
                    ctx2 = Ctx(dose_mg_day=int(dose), adherence=float(adherence), bleed=0.0, inflam=0.0)
                    out2 = simulate_predictions_for_case(cid, ctx2)
                    st.success("保存・再計算しました。")
                    render_riona_output(out2)

                    # ローカル保存（実測＋再計算結果）
                    save_record(code_hash, "riona_followup", {
                        "case_id": cid,
                        "horizon": int(horizon),
                        "hb": hb_m, "fe": fe_m, "ferritin": ferr_m, "tibc": tibc_m
                    }, {"summary":"riona_followup", "out": out2, "auto": res.get("auto_calibration", {})})

def meal_estimate(c_level: str, p_level: str, v_level: str, fried: bool, dairy: bool, fruit: bool):
    mul = {"少":0.7,"普":1.0,"多":1.3}
    c = 60.0 * mul[c_level]
    p = 30.0 * mul[p_level]
    f = 10.0 * mul[p_level]
    veg_k = 80 if v_level=="多" else (50 if v_level=="普" else 20)
    if dairy:
        p += 8; f += 5; c += 10
    if fruit:
        c += 15
    if fried:
        f += 15; c += 5
    kcal = p*4 + c*4 + f*9 + veg_k
    return {"p":p,"c":c,"f":f,"kcal":kcal}

def meal_share(prefix: str):
    # Rough split for youth athletes
    return {"b": 0.25, "l": 0.35, "d": 0.40}.get(prefix, 0.33)

def rate_meal(prefix: str, est: dict, targets: dict):
    """Return (score:int, status:str, bullets:list[str]) based on kcal/P relative to allocated share."""
    share = meal_share(prefix)
    tk = max(1.0, float(targets.get("kcal", 0.0)) * share)
    tp = max(1.0, float(targets.get("p_g", 0.0)) * share)

    kcal = float(est.get("kcal", 0.0))
    p = float(est.get("p", 0.0))

    r_k = kcal / tk
    r_p = p / tp

    # Score: penalize kcal deviation and protein shortage more than excess
    pen_k = min(45.0, abs(r_k - 1.0) * 90.0)
    pen_p = 0.0
    if r_p < 1.0:
        pen_p = min(55.0, (1.0 - r_p) * 120.0)
    else:
        pen_p = min(15.0, (r_p - 1.0) * 25.0)

    score = int(max(0.0, min(100.0, 100.0 - pen_k - pen_p)))

    bullets = []
    if r_k < 0.85:
        bullets.append("エネルギーが少なめ（午後の集中・練習前後のパフォーマンス低下に注意）")
    elif r_k > 1.20:
        bullets.append("エネルギーが多め（他の食事で調整できればOK）")
    else:
        bullets.append("エネルギー量は概ね適正")

    if r_p < 0.85:
        bullets.append("たんぱく質が不足気味（成長・回復のために主菜を増やす）")
    elif r_p > 1.20:
        bullets.append("たんぱく質は十分（取り過ぎ自体は大きな問題になりにくい）")
    else:
        bullets.append("たんぱく質量は概ね適正")

    status = "目的に合っている" if score >= 75 else ("まずまず" if score >= 55 else "改善余地あり")
    return score, status, bullets


def kyushoku_template(age_years: float):
    # 小学生/中学生で推定
    if age_years < 12:
        return {"p":25.0,"c":90.0,"f":18.0,"kcal":650.0}
    return {"p":30.0,"c":105.0,"f":22.0,"kcal":750.0}

def compute_targets_pfc(weight_kg: float, age_years: float, sport: str, intensity: str, goal: str):
    if weight_kg <= 0:
        return None
    base = 45.0 if age_years < 12 else (50.0 if age_years < 15 else 48.0)
    sport_factor = {"サッカー":1.05,"ラグビー":1.10,"野球":1.00,"テニス":1.00,"水泳":1.08}.get(sport,1.0)
    intensity_factor = {"低":0.95,"中":1.00,"高":1.10}.get(intensity,1.0)
    goal_factor = {"増量":1.08,"維持":1.00,"回復":1.03}.get(goal,1.0)
    kcal = weight_kg * base * sport_factor * intensity_factor * goal_factor
    p_perkg = {"増量":1.8,"維持":1.6,"回復":2.0}.get(goal,1.6)
    p_g = p_perkg * weight_kg
    f_pct = 0.25 if goal in ["増量","維持"] else 0.28
    f_g = (kcal * f_pct) / 9.0
    c_g = max(0.0, kcal - p_g*4.0 - f_g*9.0) / 4.0
    return {"kcal":kcal, "p_g":p_g, "c_g":c_g, "f_g":f_g}

def eval_ratio(actual: float, target: float) -> str:
    if target <= 0:
        return "不明"
    r = actual / target
    if 0.90 <= r <= 1.10:
        return "目標達成"
    if 0.75 <= r < 0.90:
        return "やや不足"
    if r < 0.75:
        return "不足"
    if 1.10 < r <= 1.25:
        return "やや過剰"
    return "過剰"



def meal_block(prefix: str, title: str, enable_photo: bool, targets: dict):
    """
    食事1回分の入力（チェック式 + 写真AI）
    prefix: "b"/"l"/"d"
    """
    st.markdown(f"#### {title}")

    ai = st.session_state.get(f"{prefix}_ai")
    if enable_photo:
        up = st.file_uploader(f"{title}の写真（任意）", type=["jpg","jpeg","png"], key=f"{prefix}_photo")
        if up is not None:
            img_bytes = up.getvalue()
            st.image(up, caption="アップロード画像", use_container_width=True)

            if st.button("AIで推論（少/普/多）", key=f"{prefix}_ai_btn"):
                out, err = analyze_meal_photo(img_bytes, title)
                if err:
                    st.error("写真解析に失敗: " + err)
                    ai = None
                    st.session_state.pop(f"{prefix}_comment", None)
                    st.session_state.pop(f"{prefix}_score", None)
                    st.session_state.pop(f"{prefix}_status", None)
                    st.session_state.pop(f"{prefix}_bullets", None)
                else:
                    ai = out
                    st.session_state[f"{prefix}_ai"] = ai

                    # estimate + rating
                    est = meal_estimate(ai.get("carb","普"), ai.get("protein","普"), ai.get("veg","普"),
                                        bool(ai.get("fried_or_oily", False)), bool(ai.get("dairy", False)), bool(ai.get("fruit", False)))
                    score, status, bullets = rate_meal(prefix, est, targets)
                    st.session_state[f"{prefix}_score"] = score
                    st.session_state[f"{prefix}_status"] = status
                    st.session_state[f"{prefix}_bullets"] = bullets

                    # AI寸評（失敗時は簡易）
                    system = "You are a sports nutrition coach specializing in youth athletes. Output Japanese."
                    user = f"""{title}の写真推論（主食/主菜/野菜/脂質/乳製品/果物）からPFCとkcalを推定しました。
推定: kcal={est['kcal']:.0f}, P={est['p']:.0f}g, C={est['c']:.0f}g, F={est['f']:.0f}g
1日の目標: kcal={targets.get('kcal',0):.0f}, P={targets.get('p_g',0):.0f}g, C={targets.get('c_g',0):.0f}g, F={targets.get('f_g',0):.0f}g
この{title}は朝昼夕の配分を考えると、今の量が適切か、改善点を短い寸評（100〜140字）で書いてください。
出力は寸評のみ。"""
                    comment, e2 = ai_text(system, user)
                    if e2 or not comment:
                        comment = " / ".join(bullets)
                    st.session_state[f"{prefix}_comment"] = comment.strip()

                    st.success("推論が完了しました。")

        if ai:
            st.caption(f"AI推定: 主食={ai.get('carb','?')} 主菜={ai.get('protein','?')} 野菜={ai.get('veg','?')} 脂質={ai.get('fat','?')} (信頼度 {ai.get('confidence',0):.2f})")

    # 手入力（AIが無い/微調整用）
    c_level = st.radio("主食（炭水化物）", ["少","普","多"], horizontal=True, index=1, key=f"{prefix}_c")
    p_level = st.radio("主菜（たんぱく質）", ["少","普","多"], horizontal=True, index=1, key=f"{prefix}_p")
    v_level = st.radio("野菜", ["少","普","多"], horizontal=True, index=1, key=f"{prefix}_v")
    dairy = st.checkbox("乳製品あり", value=False, key=f"{prefix}_dairy")
    fruit = st.checkbox("果物あり", value=False, key=f"{prefix}_fruit")
    fried = st.checkbox("揚げ物/高脂質", value=False, key=f"{prefix}_fried")

    # 推定（AIがあればAI優先）
    if ai:
        est = meal_estimate(ai.get("carb","普"), ai.get("protein","普"), ai.get("veg","普"),
                            bool(ai.get("fried_or_oily", False)), bool(ai.get("dairy", False)), bool(ai.get("fruit", False)))
    else:
        est = meal_estimate(c_level, p_level, v_level, fried, dairy, fruit)

    # 表示（点数・栄養評価・寸評）
    score = st.session_state.get(f"{prefix}_score")
    status = st.session_state.get(f"{prefix}_status")
    bullets = st.session_state.get(f"{prefix}_bullets") or []
    comment = st.session_state.get(f"{prefix}_comment")

    st.markdown("##### 推定PFC / kcal")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("P", f"{est['p']:.0f} g")
    m2.metric("C", f"{est['c']:.0f} g")
    m3.metric("F", f"{est['f']:.0f} g")
    m4.metric("kcal", f"{est['kcal']:.0f}")

    if score is None or status is None:
        # AI推論前でも評価は出す（手入力ベース）
        s2, st2, bl2 = rate_meal(prefix, est, targets)
        score, status, bullets = s2, st2, bl2

    st.markdown(f"##### {title}スコア：**{int(score)} / 100**（{status}）")
    if bullets:
        for b in bullets:
            st.write("・" + b)
    if comment:
        st.markdown("##### 寸評")
        st.write(comment)

    return est
def meal_page(code_hash: str):
    st.subheader("食事ログ（1日チェック）")
    st.caption("朝・昼・夕で1日のPFCを推定します。昼は「給食（簡易）」または「通常（朝夕と同等）」を選べます。")

    # --- 保存/読込（食事ログ）---
    c1, c2 = st.columns(2)
    if c1.button("読込", key="meal_load_top"):
        payload = load_snapshot(code_hash, "meal_draft")
        if payload:
            for k, v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if c2.button("保存", key="meal_save_top"):
        keys = [
            "meal_goal", "meal_intensity", "meal_weight",
            "school_lunch", "l_menu", "l_kcal_simple", "l_p_simple", "l_c_simple", "l_f_simple",
            "b_c", "b_p", "b_v", "b_dairy", "b_fruit", "b_fried", "b_ai",
            "l_c", "l_p", "l_v", "l_dairy", "l_fruit", "l_fried", "l_ai",
            "d_c", "d_p", "d_v", "d_dairy", "d_fruit", "d_fried", "d_ai",
        ]
        save_snapshot(code_hash, "meal_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")

    sport = st.session_state.get("sport", SPORTS[0])
    age_years = float(st.session_state.get("age_years", 15.0) or 15.0)
    weight0 = float(st.session_state.get("latest_weight_kg", 0.0) or 0.0)

    top = st.columns(4)
    goal = top[0].selectbox("目的", ["増量","維持","回復"], index=1, key="meal_goal")
    intensity = top[1].selectbox("運動強度", ["低","中","高"], index=1, key="meal_intensity")
    weight = top[2].number_input("体重（kg）", 20.0, 150.0, value=weight0 if weight0>0 else 45.0, step=0.1, key="meal_weight")
    top[3].caption(f"競技：{sport} / 年齢：{age_years:.1f}")

    st.session_state["latest_weight_kg"] = float(weight)

    targets = compute_targets_pfc(weight, age_years, sport, intensity, goal)
    st.markdown("### 目標（P/F/C）")
    t1,t2,t3,t4 = st.columns(4)
    t1.metric("炭水化物", f"{targets['c_g']:.0f} g")
    t2.metric("たんぱく質", f"{targets['p_g']:.0f} g")
    t3.metric("脂質", f"{targets['f_g']:.0f} g")
    t4.metric("総カロリー", f"{targets['kcal']:.0f} kcal")


    with st.expander("朝食", expanded=True):
        b = meal_block("b", "朝食", True, targets)

    # --- 昼食（給食なら簡易、給食でないなら朝夕と同等に）---
    with st.expander("昼食", expanded=False):
        st.markdown("#### 昼食")
        is_school = st.checkbox("給食（学校の標準的な昼食）", value=True, key="l_is_school")
        if is_school:
            st.caption("給食の日は、ざっくり推定（kcal/PFC）にとどめます。メニューが分かれば入力してください。")
            menu = st.text_area("メニュー（分かる範囲で）", key="l_menu", placeholder="例：ごはん、鶏の照り焼き、みそ汁、牛乳…")
            lk = st.number_input("推定カロリー（kcal）", 0.0, 2000.0, value=650.0, step=10.0, key="l_kcal_simple")
            lp = st.number_input("たんぱく質（g）", 0.0, 200.0, value=25.0, step=1.0, key="l_p_simple")
            lc = st.number_input("炭水化物（g）", 0.0, 400.0, value=90.0, step=1.0, key="l_c_simple")
            lf = st.number_input("脂質（g）", 0.0, 200.0, value=18.0, step=1.0, key="l_f_simple")
            l = {"p": float(lp), "c": float(lc), "f": float(lf), "kcal": float(lk), "menu": menu, "mode": "school"}
        else:
            st.caption("給食でない日は、朝食・夕食と同じように写真AI＋詳細推定で入力します。")
            l = meal_block("l", "昼食", True, targets)
            l["mode"] = "normal"

        # 昼食のAIコメント（しっかり）
        if st.button("昼食のAIコメント（しっかり）", key="l_ai_comment_btn"):
            # 目標との差分を昼の一言に落とす
            targets_local = targets  # meal_page内のtargetsを参照
            # ここでは昼食単体と、昼までの累計でコメントを作る
            p_l = float(l.get("p", 0.0) or 0.0)
            c_l = float(l.get("c", 0.0) or 0.0)
            f_l = float(l.get("f", 0.0) or 0.0)
            k_l = float(l.get("kcal", 0.0) or 0.0)
            menu_txt = l.get("menu", "") if isinstance(l, dict) else ""
            system = "You are a sports nutrition coach for junior athletes. Output Japanese. Be specific with grams/portions. No long preface."
            user = f"""目的: {goal}
運動強度: {intensity}
体重: {weight} kg
1日の目標: kcal={targets_local['kcal']:.0f}, C={targets_local['c_g']:.0f}g, P={targets_local['p_g']:.0f}g, F={targets_local['f_g']:.0f}g

昼食（推定）:
- kcal: {k_l:.0f}
- C/P/F: {c_l:.0f}g / {p_l:.0f}g / {f_l:.0f}g
- メニュー: {menu_txt if menu_txt else "不明（写真/入力ベース）"}
お願い:
- 昼食の評価（良い点/改善点）を短く
- 今日は“夕食でどう帳尻を合わせるか”を具体量で提案（例：ごはん何g、肉/魚何g、牛乳/ヨーグルト量）
- もし不足が大きければ、間食案（1〜2個）も提案（コンビニで買えるレベル）
- 文章は見出し＋箇条書き中心で、読みやすく
"""
            text, err = ai_text(system, user)
            if err:
                st.error("AIコメントに失敗: " + err)
            else:
                st.session_state["l_ai_comment_text"] = text
        if st.session_state.get("l_ai_comment_text"):
            st.markdown("##### AIコメント")
            st.write(st.session_state["l_ai_comment_text"])

    with st.expander("夕食", expanded=True):

        d = meal_block("d", "夕食", True, targets)

    tot_p = b["p"] + l["p"] + d["p"]
    tot_c = b["c"] + l["c"] + d["c"]
    tot_f = b["f"] + l["f"] + d["f"]
    tot_k = b["kcal"] + l["kcal"] + d["kcal"]

    st.markdown("### 1日の推定と評価")
    r_p = eval_ratio(tot_p, targets["p_g"])
    r_c = eval_ratio(tot_c, targets["c_g"])
    r_f = eval_ratio(tot_f, targets["f_g"])
    r_k = eval_ratio(tot_k, targets["kcal"])
    e1,e2,e3,e4 = st.columns(4)
    e1.metric("炭水化物", f"{tot_c:.0f} g", r_c)
    e2.metric("たんぱく質", f"{tot_p:.0f} g", r_p)
    e3.metric("脂質", f"{tot_f:.0f} g", r_f)
    e4.metric("総カロリー", f"{tot_k:.0f} kcal", r_k)

    
    with st.expander("📅 食事ログ（カレンダー）", expanded=False):
        rows = load_records(code_hash, limit=500)
        meals = [r for r in rows if r.get("kind") == "meal_day"]
        if not meals:
            st.caption("まだ保存がありません。")
        else:
            # 日付ごとに集計（最新30日）
            data = []
            for r in meals:
                try:
                    dt = r.get("created_at","")[:10]  # YYYY-MM-DD
                    pl = r.get("payload") or {}
                    rt = (pl.get("ratings") or {})
                    data.append({
                        "date": dt,
                        "p": rt.get("p",""),
                        "c": rt.get("c",""),
                        "f": rt.get("f",""),
                        "kcal": rt.get("kcal","")
                    })
                except Exception:
                    pass
            dfm = pd.DataFrame(data).dropna()
            if dfm.empty:
                st.caption("ログが読み取れませんでした。")
            else:
                dfm = dfm.sort_values("date")
                dfm = dfm.drop_duplicates(subset=["date"], keep="last")
                dfm_tail = dfm.tail(31).reset_index(drop=True)
                st.dataframe(dfm_tail, use_container_width=True, hide_index=True)
                st.caption("※日付ごとに最新の食事ログ評価を表示しています（直近約1ヶ月）。")

    if st.button("結果保存（食事ログ）", key="meal_save"):
        save_record(code_hash, "meal_day",
                    {"goal": goal, "intensity": intensity, "weight": weight, "targets": targets,
                     "breakfast": b, "lunch": l, "dinner": d,
                     "total": {"p": tot_p, "c": tot_c, "f": tot_f, "kcal": tot_k},
                     "ratings": {"p": r_p, "c": r_c, "f": r_f, "kcal": r_k}},
                    {"summary":"meal_day"})
        st.success("保存しました。")


    st.divider()
    if st.button("記入データ読込", key="meal_load_bottom"):
        payload = load_snapshot(code_hash, "meal_draft")
        if payload:
            for k,v in payload.items():
                st.session_state[k] = v
            st.success("読み込みました。")
            st.rerun()
        else:
            st.info("保存データがありません。")
    if st.button("保存", key="meal_save_bottom"):
        keys = ["meal_goal","meal_intensity","meal_weight","b_c","b_p","b_v","b_dairy","b_fruit","b_fried","l_kyu","l_c","l_p","l_v","l_dairy","l_fruit","l_fried","d_c","d_p","d_v","d_dairy","d_fruit","d_fried"]
        save_snapshot(code_hash, "meal_draft", {k: st.session_state.get(k) for k in keys})
        st.success("保存しました。")


def advice_page(code_hash: str):
    st.subheader("🤖 Aiアドバイス")
    st.markdown("""<style>
    /* Make tabs easier to find */
    div[data-baseweb="tab"] button{font-size:16px !important; padding:10px 14px !important;}
    div[data-baseweb="tab-list"]{gap:6px;}
    </style>""", unsafe_allow_html=True)
    sport = st.session_state.get("sport", SPORTS[0])

    # ---- Training log (per-user latest + history) ----
    with st.expander("📝 トレーニング（保存・最新読み込み）", expanded=True):
        st.session_state.setdefault("tr_date", now_jst().date())
        st.session_state.setdefault("tr_type", "チーム練習")
        st.session_state.setdefault("tr_duration", 0)
        st.session_state.setdefault("tr_rpe", 5)
        st.session_state.setdefault("tr_focus", "")
        st.session_state.setdefault("tr_notes", "")

        st.date_input("日付", value=st.session_state.get("tr_date"), key="tr_date")
        st.selectbox(
            "種類",
            ["チーム練習","試合","筋力（上半身）","筋力（下半身）","スプリント","持久走","リカバリー","その他"],
            index=0,
            key="tr_type"
        )
        st.number_input(
            "時間（分）",
            min_value=0, max_value=600,
            step=5,
            key="tr_duration"
        )
        st.slider("主観的きつさ（RPE 1-10）", 1, 10, int(st.session_state.get("tr_rpe", 5) or 5), key="tr_rpe")
        st.text_input("主目的（例：スプリント/当たり負け改善/持久力）", value=st.session_state.get("tr_goal_text",""), key="tr_goal_text")
        st.text_area("内容メモ（セット数・距離・本数など）", value=st.session_state.get("tr_notes",""), height=120, key="tr_notes")

        cA, cB, cD, cC = st.columns([1,1,1,2])
        with cA:
            if st.button("保存", key="tr_log_save"):
                try:
                    save_training_latest(code_hash)
                    st.success("保存しました。")
                except Exception as e:
                    st.error(f"保存に失敗: {e}")
        with cB:
            if st.button("最新を読み込み", key="tr_log_load"):
                try:
                    ok = load_training_latest(code_hash)
                    if ok:
                        st.success("最新のトレーニングを読み込みました。")
                        st.rerun()
                    else:
                        st.info("保存データがありません。")
                except Exception as e:
                    st.error(f"読み込みに失敗: {e}")
        
        with cD:
            if st.button("削除（最新）", key="tr_log_delete"):
                try:
                    delete_snapshot(code_hash, "training_latest")
                    delete_latest_record(code_hash, "training_log")
                    # also clear current inputs to defaults
                    st.session_state["tr_duration"] = 0
                    st.session_state["tr_rpe"] = 5
                    st.session_state["tr_notes"] = ""
                    st.success("最新の保存データを削除しました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"削除に失敗: {e}")
        with cC:
            try:
                hist = load_records(code_hash, limit=30)
                hist = [h for h in hist if h.get("kind")=="training_log"][:5]
            except Exception:
                hist = []
            if hist:
                st.caption("直近の保存（最大5件）")
                for h in hist:
                    pl = h.get("payload") or {}
                    d = pl.get("tr_date","")
                    st.write(f"- {d} / {pl.get('tr_type','')} / {pl.get('tr_duration','')}分 / RPE{pl.get('tr_rpe','')}")

    # ---- 端末保存（CSV/カレンダー） ----
    with st.expander("📱 トレーニング記録を端末に保存／カレンダーで見る", expanded=False):
        try:
            recs = load_records(code_hash, limit=400)
            recs = [r for r in recs if r.get("kind") == "training_log"]
        except Exception:
            recs = []

        if not recs:
            st.info("まだトレーニング記録がありません（上で「保存」を押すと蓄積されます）。")
        else:
            rows = []
            for r in recs:
                pl = r.get("payload") or {}
                rows.append({
                    "date": str(pl.get("tr_date", "")),
                    "type": str(pl.get("tr_type", "")),
                    "duration_min": pl.get("tr_duration", ""),
                    "rpe": pl.get("tr_rpe", ""),
                    "goal": pl.get("tr_goal_text", pl.get("tr_focus", "")) or "",
                    "notes": str(pl.get("tr_notes", "")),
                })
            df = pd.DataFrame(rows)

            st.markdown("##### 🗑️ 記録の削除")
            dates = [d for d in df["date"].dropna().astype(str).tolist() if d]
            if dates:
                target_date = st.selectbox("削除したい日付", sorted(list(set(dates)), reverse=True), key="tr_delete_date")
                if st.button("この日付の最新記録を削除", key="tr_delete_by_date"):
                    try:
                        # newest record with that date
                        for r in recs:
                            pl = r.get("payload") or {}
                            if str(pl.get("tr_date", "")) == target_date:
                                rid = r.get("id")
                                if rid is not None:
                                    delete_record_by_id(rid)
                                    st.success("削除しました。")
                                    st.rerun()
                        st.warning("削除対象が見つかりませんでした。")
                    except Exception as e:
                        st.error(f"削除に失敗: {e}")
            else:
                st.caption("削除できる記録がありません。")

            st.markdown("##### ⬇️ 端末に保存")
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSVとして保存（端末に残す）",
                data=csv_bytes,
                file_name="training_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # iCalendar (.ics)
            def _ics_escape(s: str) -> str:
                s = str(s or "")
                return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

            ics_lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Kiwi//TrainingLog//JA"]
            for r in recs:
                pl = r.get("payload") or {}
                d = str(pl.get("tr_date", ""))
                if not d:
                    continue
                try:
                    y, m, dd = [int(x) for x in d.split("-")]
                    dt = datetime(y, m, dd, 9, 0, tzinfo=JST)
                except Exception:
                    continue
                summary = f"TR: {pl.get('tr_type','')}"
                desc = f"{pl.get('tr_duration','')}分 / RPE{pl.get('tr_rpe','')}\n{pl.get('tr_notes','')}"
                uid = f"{r.get('id','')}-{code_hash}@kiwi"
                ics_lines += [
                    "BEGIN:VEVENT",
                    f"UID:{_ics_escape(uid)}",
                    f"DTSTAMP:{dt.strftime('%Y%m%dT%H%M%SZ')}",
                    f"DTSTART:{dt.strftime('%Y%m%dT%H%M%S')}",
                    f"SUMMARY:{_ics_escape(summary)}",
                    f"DESCRIPTION:{_ics_escape(desc)}",
                    "END:VEVENT",
                ]
            ics_lines.append("END:VCALENDAR")
            ics_bytes = "\n".join(ics_lines).encode("utf-8")
            st.download_button(
                "カレンダー用(.ics)で保存",
                data=ics_bytes,
                file_name="training_log.ics",
                mime="text/calendar",
                use_container_width=True,
            )

            st.markdown("##### 📅 アプリ内カレンダー（一覧）")
            # very simple month filter
            today = datetime.now(JST).date()
            ym_options = sorted(list(set([d[:7] for d in dates if len(d) >= 7])), reverse=True)
            default_ym = today.strftime("%Y-%m")
            if default_ym not in ym_options and ym_options:
                default_ym = ym_options[0]
            ym = st.selectbox("表示する月", ym_options or [default_ym], index=0, key="tr_cal_month")
            if ym:
                cal_df = df[df["date"].astype(str).str.startswith(ym)].copy()
                cal_df = cal_df.sort_values("date", ascending=True)
                st.dataframe(cal_df, use_container_width=True, hide_index=True)

    # ---- Tabs ----

    # -----------------
    # トレーニング
    # -----------------
    with t1:
        st.markdown("### 筋トレメニュー提案")
        st.caption("体重や筋力の情報から、上半身・下半身・体幹をバランスよく提案します。")

        w = st.number_input("体重（kg）", min_value=20.0, max_value=150.0,
                            value=float(st.session_state.get("latest_weight_kg", 45.0) or 45.0),
                            step=0.1, key="tr_weight")
        st.session_state["latest_weight_kg"] = float(w)

        bench1rm = st.number_input("ベンチプレス最大（推定1回の重さ kg・任意）", min_value=0.0, max_value=300.0,
                                   value=float(st.session_state.get("tr_bench1rm", 0.0) or 0.0),
                                   step=0.5, key="tr_bench1rm")

        squat_est = round(w * 1.2, 1)
        st.caption(f"スクワット（重りを使う場合の目安）: 体重×1.2 ≈ {squat_est} kg（フォーム優先）")

        equipment = st.selectbox("使える器具", ["自重中心（道具なし）", "ダンベル/チューブあり", "バーベル（ベンチ・スクワット可能）"],
                                 index=0, key="tr_equipment")
        days = st.selectbox("週あたりの筋トレ日数", [1,2,3,4], index=2, key="tr_days")
        focus = st.selectbox("今の目的（筋トレ）", ["強くなる", "スピード・跳躍", "怪我予防", "疲労回復を優先"], index=0, key="tr_menu_focus")

        if st.button("AIでメニューを作る", type="primary", key="tr_ai"):
            system = "You are a strength & conditioning coach specializing in youth athletes. Output concise Japanese."
            user = f"""競技: {sport}
体重: {w} kg
ベンチプレス最大(推定1RM): {bench1rm if bench1rm>0 else '不明'} kg
スクワット目安: {squat_est} kg（体重から推定）
器具: {equipment}
週の筋トレ日数: {days}
目的: {focus}

要件:
- 上半身/下半身/体幹に分ける
- 1回あたり30〜45分
- ジュニアなのでフォーム・安全最優先（重すぎない）
- 重りが使える場合はベンチプレスやスクワットの「目安重量(kg)」も提案
- 自重中心の場合は負荷の上げ方（回数/テンポ/片脚など）を提案
- 4週間の進め方（1〜4週の変化）を短く
出力は見出し＋箇条書きで。"""
            text, err = ai_text(system, user)
            if err:
                st.error("AI提案に失敗: " + err)
            else:
                st.session_state["tr_menu_text"] = text
                render_menu_blocks(text)

        if st.button("トレーニングログを保存", key="tr_inputs_save"):
            save_record(code_hash, "training_inputs",
                        {"sport": sport, "weight": w, "bench1rm": bench1rm, "squat_est": squat_est,
                         "equipment": equipment, "days": days, "focus": focus},
                        {"summary": "training_inputs"})
            st.success("保存しました。")

    # -----------------
    # 怪我
    # -----------------
    with t2:
        st.markdown("### 怪我のチェック")
        st.caption("痛む場所を選ぶと質問が増えます。最後にAIがコメントします。")

        cols = st.columns(3)
        locs = []
        loc_list = ["頭/首", "肩", "肘", "手首/手", "背中/腰", "股関節/鼠径部", "太もも", "ハムストリング", "膝", "足首", "踵/足底"]
        for i, loc in enumerate(loc_list):
            with cols[i % 3]:
                if st.checkbox(loc, key=f"inj_loc_{loc}"):
                    locs.append(loc)

        pain = st.slider("痛み（0-10）", 0, 10, 0, key="inj_pain")
        st.caption("例：0=痛みなし / 2-3=違和感 / 4-5=動かすと痛い / 6-7=練習が難しい / 8-10=日常生活もつらい")

        onset = st.selectbox("きっかけ", ["急に（ひねった・ぶつけた・着地で痛い）", "少しずつ（使いすぎ・疲れ）"], index=0, key="inj_onset")
        swelling = st.checkbox("腫れがある", key="inj_swelling")
        bruise = st.checkbox("内出血がある", key="inj_bruise")
        numb = st.checkbox("しびれ・感覚の違和感がある", key="inj_numb")
        fever = st.checkbox("熱がある", key="inj_fever")
        weight_bearing = st.selectbox("体重をかけられる？（足の痛みがある場合）", ["問題なし", "少し痛いが可能", "ほぼ無理"], index=0, key="inj_bearing")

        extra = {}
        if locs:
            st.markdown("#### 追加の質問（選んだ場所に応じて）")
            for loc in locs:
                with st.expander(f"{loc} の追加質問", expanded=False):
                    if loc in ["膝", "足首", "股関節/鼠径部"]:
                        extra[f"{loc}_giving_way"] = st.checkbox("踏ん張るとガクっとする/抜ける感じがある", key=f"inj_{loc}_give")
                        extra[f"{loc}_locking"] = st.checkbox("引っかかる/動かしにくい感じがある", key=f"inj_{loc}_lock")
                    if loc in ["肩", "肘", "手首/手"]:
                        extra[f"{loc}_throw"] = st.checkbox("投げる/打つ動作で強く痛む", key=f"inj_{loc}_throw")
                        extra[f"{loc}_weak"] = st.checkbox("力が入りにくい", key=f"inj_{loc}_weak")
                    if loc in ["背中/腰"]:
                        extra[f"{loc}_legpain"] = st.checkbox("脚の方に痛み/しびれが走る", key=f"inj_{loc}_rad")
                    extra[f"{loc}_worse"] = st.selectbox("一番つらい動き", ["走る", "ジャンプ", "切り返し", "蹴る", "投げる", "日常動作"], index=0, key=f"inj_{loc}_worse")

        st.markdown("### 直ぐにできる対応")
        st.write("• **痛みの出る動きは行わない**（痛みが出ない範囲での活動に切り替える）")
        st.write("• **冷やす**：氷や保冷剤をタオルで包んで、10〜15分を1日に数回")
        st.write("• **押さえる**：腫れているなら、包帯やサポーターで軽く固定（きつすぎない）")
        st.write("• **高くする**：足のケガなら、座って足をクッションで少し高くする")
        st.write("• 痛みが強い/腫れが増える/しびれ/体重をかけられない/熱がある時は、早めに相談が安心です。")

        if st.button("AIコメントを出す", type="primary", key="inj_ai"):
            system = "You are a sports medicine assistant for youth athletes. Output Japanese. Avoid the phrase '受診の目安'. Be kind and clear."
            user = f"""競技: {sport}
痛い場所: {", ".join(locs) if locs else "未選択"}
痛みスケール(0-10): {pain}
きっかけ: {onset}
腫れ: {swelling}
内出血: {bruise}
しびれ: {numb}
熱: {fever}
荷重: {weight_bearing}
追加情報: {json.dumps(extra, ensure_ascii=False)}

お願い:
- 整形外科医に伝わるように、以下の形式で出力
  1) まとめ（部位/発症様式/痛みの強さ/腫れ・内出血・しびれ・荷重/悪化動作）
  2) 考えやすい鑑別（3〜5個、可能性の理由を短く）
  3) 直ぐにできる対応（冷やし方/固定/痛くない範囲での代替運動）
  4) 相談を急いだ方がよいサイン（箇条書き）
- “受診の目安”という言葉は使わない
- 文章は短め、箇条書き中心
"""
            text, err = ai_text(system, user)
            if err:
                st.error("AIコメントに失敗: " + err)
            else:
                st.write(text)

        if st.button("怪我ログを保存", key="inj_save"):
            save_record(code_hash, "injury_log",
                        {"sport": sport, "locations": locs, "pain": pain, "onset": onset,
                         "swelling": swelling, "bruise": bruise, "numb": numb, "fever": fever,
                         "bearing": weight_bearing, "extra": extra},
                        {"summary": "injury_log"})
            st.success("保存しました。")

    # -----------------
    # 睡眠
    # -----------------
    with t3:
        st.markdown("### 睡眠")
        wake = st.time_input("起床時刻", value=time(6,0))
        sleep_h = st.number_input("昨日の睡眠時間（時間）", 0.0, 16.0, 8.0, 0.25)
        screen = st.number_input("就寝前のスマホ・ゲーム時間（分）", 0, 300, 60, 5)
        score = 100
        if sleep_h < 8.0:
            score -= 20
        if screen >= 90:
            score -= 15
        score = max(0, min(100, score))
        st.write(f"睡眠スコア（簡易）：{score}/100")
        if st.button("睡眠ログを保存", key="sl_save"):
            save_record(code_hash, "sleep_log",
                        {"wake": str(wake), "sleep_h": float(sleep_h), "screen": int(screen), "score": int(score)},
                        {"summary": "sleep_log"})
            st.success("保存しました。")




    # -----------------
    # サッカー動画（YouTube検索）
    # -----------------
    with t4:
        if sport != "サッカー":
            st.caption("このタブはサッカー選手向けです。競技がサッカーの場合に使ってください。")
        else:
            st.markdown("### やりたいプレーからおすすめ動画")
            st.caption("例：裏抜け / 1対1突破 / ハーフスペースの受け方 / ビルドアップ / 守備の間合い / カウンターの判断 など")
            style = st.text_area("やりたいプレー・課題（できるだけ具体的に）", height=120, key="soccer_style")
            if st.button("おすすめ動画リンクを作る", type="primary", key="soccer_make_links"):
                system = "You are a soccer coach. Produce 5 Japanese YouTube search queries. Output one per line, no extra text."
                user = f"テーマ: {style}"
                text, err = ai_text(system, user)
                if err:
                    st.error("AIに失敗: " + err)
                else:
                    queries = [q.strip("-• 	") for q in (text or "").splitlines() if q.strip()]
                    st.markdown("#### YouTube検索リンク")
                    import urllib.parse
                    for q in queries[:5]:
                        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(q)
                        st.markdown(f"- [{q}]({url})")


def main():
    st.set_page_config(page_title="Height & Riona (Rebuild Stable)", layout="wide")
    apply_css()
    init_users_db()
    init_data_db()

    user = st.session_state.get("user")
    if not user:
        user = login_panel()
        if not user:
            return

    code_hash = sha256_hex(user)

    # per-user saved data
    try:
        load_basic_info_snapshot(code_hash)
    except Exception:
        pass
    try:
        load_training_latest(code_hash)
    except Exception:
        pass

    shared_demographics()
    auto_fill_latest_all_tabs(code_hash)
    auto_fill_from_latest_records(code_hash)

    st.markdown("### 画面選択")
    with st.container():
        nav = st.radio("", ["身長予測","貧血・リオナ","食事ログ","Aiアドバイス"], horizontal=True, key="nav_main")
    if nav == "身長予測":
        height_page(code_hash)
    elif nav == "貧血・リオナ":
        anemia_page(code_hash)
    elif nav == "食事ログ":
        meal_page(code_hash)
    elif nav == "Aiアドバイス":
        advice_page(code_hash)
    else:
        advice_page(code_hash)

if __name__ == "__main__":
    main()
