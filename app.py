# -*- coding: utf-8 -*-
import os
import sqlite3
import hashlib
import secrets
import threading
import time
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

# =========================
# Branding
# =========================
def render_brand_header():
    try:
        cols = st.columns([1, 4])
        with cols[0]:
            st.image("assets/logo.png", width=72)
        with cols[1]:
            st.markdown('<div style="font-size:28px;font-weight:800;line-height:1.1;">ジュニアス</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:rgba(0,0,0,0.55);font-size:14px;margin-top:2px;">ジュニアアスリートの成長とコンディションを支えるメディカルサポート</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown("## ジュニアス")

from core import init_db, Labs, Ctx, register_case, add_followup, resolve_case_id, simulate_predictions_for_case

# =========================
# APIキー（Streamlit Secrets推奨）
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# =========================
# Config
# =========================
TZ = timezone(timedelta(hours=9))
JST = TZ  # alias

# =========================
# AIコメントの永続化（翌日・別端末でも残す）
# - st.session_stateに加えて、DB(snapshots)に保存します
# - ブラウザ/端末を変えても、同じIDでログインすれば復元できます
AI_PERSIST_KEYS = [
    "tr_menu_text",      # 筋トレメニュー
    "sl_ai_text",        # 睡眠AIアドバイス
    "inj_ai_text",       # 怪我AIコメント
    "l_ai_comment_text", # 食事（昼）のAIコメント（給食/非給食共通で使う想定）
]

def _ai_cache_load(code_hash: str) -> dict:
    d = load_snapshot(code_hash, "ai_cache") or {}
    if isinstance(d, dict):
        return d
    return {}

def _ai_cache_save(code_hash: str, cache: dict) -> None:
    if not isinstance(cache, dict):
        return
    save_snapshot(code_hash, "ai_cache", cache)

def restore_ai_cache_to_session(code_hash: str) -> None:
    cache = _ai_cache_load(code_hash)
    for k in AI_PERSIST_KEYS:
        v = cache.get(k)
        if v:
            st.session_state.setdefault(k, v)

def persist_ai_cache_from_session(code_hash: str) -> None:
    cache = _ai_cache_load(code_hash)
    changed = False
    for k in AI_PERSIST_KEYS:
        v = st.session_state.get(k)
        if v and cache.get(k) != v:
            cache[k] = v
            changed = True
    if changed:
        _ai_cache_save(code_hash, cache)

def download_text_button(label: str, text: str, filename: str, key: str):
    if not text:
        return
    st.download_button(
        label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        key=key
    )
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



def clipboard_copy_button(label: str, text: str, key: str, height: int = 56):
    """ワンタップでクリップボードにコピー（スマホでも使いやすい）"""
    try:
        payload = json.dumps(text or "", ensure_ascii=False)
    except Exception:
        payload = json.dumps(str(text or ""), ensure_ascii=False)
    components.html(
        f"""
        <div style="margin: 6px 0 10px 0;">
          <button id="{key}" style="
            width: 100%;
            padding: 14px 14px;
            font-size: 18px;
            font-weight: 700;
            border-radius: 14px;
            border: 1px solid #d0d0d0;
            background: white;
            cursor: pointer;
          ">📋 {label}</button>
        </div>
        <script>
          const btn = document.getElementById("{key}");
          if(btn) {{
            btn.onclick = async () => {{
              try {{
                await navigator.clipboard.writeText({payload});
                btn.innerText = "✅ コピーしました";
                setTimeout(()=>{{ btn.innerText = "📋 {label}"; }}, 1400);
              }} catch(e) {{
                btn.innerText = "⚠️ コピーできません";
                setTimeout(()=>{{ btn.innerText = "📋 {label}"; }}, 1600);
              }}
            }};
          }}
        </script>
        """,
        height=height,
    )

def _find_jams_logo_path():
    candidates = [
        "JAMSロゴ.png",
        "JAMSロゴ.png",
        "assets/JAMSロゴ.png",
        "assets/JAMSロゴ.png",
        "static/JAMSロゴ.png",
        "static/JAMSロゴ.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def render_login_brand():
    p = _find_jams_logo_path()
    st.markdown("<div style='text-align:center; margin-top:24px; margin-bottom:18px;'>", unsafe_allow_html=True)
    if p:
        st.image(p, width=280)
    st.markdown("<h2 style='margin:12px 0 0 0;'>プライベートスポーツドクター</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555; font-size:14px; margin-top:6px;'>Junior Athlete Medical Support</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def jams_logo_footer():
    p = _find_jams_logo_path()
    if not p:
        return
    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        st.image(p, width=180)


def apply_css():
    st.markdown("""
    <style>
      /* === Mobile-first readability (40代でも迷わず押せる) === */
      html, body, [class*="css"] { font-size: 17px; }
      @media (max-width: 640px){
        html, body, [class*="css"] { font-size: 19px; }
        .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
      }

      /* Buttons */
      .stButton > button {
        font-size: 18px !important;
        font-weight: 800 !important;
        padding: 0.85rem 0.9rem !important;
        border-radius: 14px !important;
        min-height: 56px !important;
      }
      @media (max-width: 640px){
        .stButton > button{
          font-size: 20px !important;
          min-height: 64px !important;
          padding: 1.0rem 1.0rem !important;
          border-radius: 16px !important;
        }
      }

      /* Inputs */
      .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div{
        font-size: 18px !important;
        min-height: 52px !important;
      }
      @media (max-width: 640px){
        .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div{
          font-size: 20px !important;
          min-height: 56px !important;
        }
      }

      /* Section headings */
      h1, h2, h3 { letter-spacing: -0.01em; }
      h2 { font-size: 1.45rem !important; }
      @media (max-width: 640px){
        h2 { font-size: 1.55rem !important; }
      }

      /* Make radio/checkbox labels easier to tap */
      label[data-baseweb="checkbox"], label[data-baseweb="radio"] { padding-top: 10px; padding-bottom: 10px; }

      /* Menu: keep 2 buttons per row, big and tappable */
      .km-menu-title{ font-size: 22px; font-weight: 900; margin: 6px 0 8px 0; }
      @media (max-width: 640px){ .km-menu-title{ font-size: 24px; } }
      .km-menu-sub{ color: rgba(0,0,0,0.65); font-size: 15px; margin-bottom: 10px; }
      @media (max-width: 640px){ .km-menu-sub{ font-size: 16px; } }

      .km-bigbtn .stButton > button{
        min-height: 78px !important;
        font-size: 20px !important;
      }
      @media (max-width: 640px){
        .km-bigbtn .stButton > button{
          min-height: 92px !important;
          font-size: 22px !important;
        }
      }

      /* Back-to-menu button should also be large */
      .km-navbtn .stButton > button{
        min-height: 60px !important;
        font-size: 18px !important;
      }
      @media (max-width: 640px){
        .km-navbtn .stButton > button{
          min-height: 68px !important;
          font-size: 20px !important;
        }
      }
    
      .block-container { padding-top: 2.2rem; }
      div[data-id="stHorizontalBlock"] { gap: 6px !important; padding: 0 4px; }
div[data-id="stHorizontalBlock"]::after{ content:""; display:block; height:1px; background: rgba(0,0,0,0.10); margin-top:-1px; }
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]{
  border: 1px solid rgba(0,0,0,0.10);
  border-bottom: 0;
  border-radius: 12px 12px 0 0;
  padding: 8px 14px !important;
  background: rgba(255,255,255,0.85);
  box-shadow: 0 6px 14px rgba(0,0,0,0.06);
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:has(input:checked){
  background: #ffffff;
  box-shadow: 0 10px 22px rgba(0,0,0,0.08);
  transform: translateY(1px);
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"] p{ margin:0; font-weight:700; }

      div[data-id="column"] button{ width: 100%; }
      .stExpander{
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 10px 24px rgba(0,0,0,0.04);
        background: rgba(255,255,255,0.92);
      }
    
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(1){
  border-left: 4px solid rgba(59,130,246,0.8) !important;
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(2){
  border-left: 4px solid rgba(239,68,68,0.8) !important;
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(3){
  border-left: 4px solid rgba(16,185,129,0.8) !important;
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:nth-child(4){
  border-left: 4px solid rgba(245,158,11,0.85) !important;
}

/* ===== Premium mobile nav ===== */
@media (max-width: 640px){
  .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
  div[data-id="stHorizontalBlock"] { gap: 8px !important; }
  div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]{
    padding: 10px 14px !important;
    border-radius: 14px !important;
    font-size: 16px !important;
  }
}
/* Make nav look like premium segmented tabs */
div[data-id="stHorizontalBlock"]{
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
div[data-id="stHorizontalBlock"]::after{ display:none !important; }

div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]{
  border: 0 !important;
  border-radius: 14px !important;
  padding: 9px 14px !important;
  background: rgba(0,0,0,0.04) !important;
  box-shadow: none !important;
  transition: all 120ms ease;
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"]:has(input:checked){
  background: #111827 !important;
  color: #fff !important;
  box-shadow: 0 10px 20px rgba(0,0,0,0.12) !important;
  transform: none !important;
}
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"] p{ font-weight: 800; }

/* Color accents per tab label (unselected) */
div[data-id="stHorizontalBlock"] label[data-baseweb="radio"] p:contains("身長"){ }


/* ===== Main nav (radio) premium ===== */
div[data-id="stRadio"] div[role="radiogroup"]{
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 16px;
  padding: 8px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.06);
}
div[data-id="stRadio"] label[data-baseweb="radio"]{
  border-radius: 14px !important;
  padding: 10px 14px !important;
  background: rgba(0,0,0,0.04) !important;
  border-left: 4px solid rgba(0,0,0,0.0) !important;
}
div[data-id="stRadio"] label[data-baseweb="radio"]:has(input:checked){
  background: rgba(255,255,255,0.98) !important;
  color: #111827 !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.12) !important;
  outline: 2px solid rgba(17,24,39,0.20);
}

div[data-id="stRadio"] label[data-baseweb="radio"] p{ margin:0; font-weight:800; }
@media (max-width: 640px){
  div[data-id="stRadio"] label[data-baseweb="radio"]{ font-size: 16px !important; }
}


div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(1){ border-left:4px solid rgba(59,130,246,0.85) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(2){ border-left:4px solid rgba(239,68,68,0.85) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(3){ border-left:4px solid rgba(16,185,129,0.85) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(4){ border-left:4px solid rgba(245,158,11,0.90) !important; }

div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(1):has(input:checked){ background: rgba(59,130,246,0.10) !important; outline-color: rgba(59,130,246,0.35) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(2):has(input:checked){ background: rgba(239,68,68,0.10) !important; outline-color: rgba(239,68,68,0.35) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(3):has(input:checked){ background: rgba(16,185,129,0.10) !important; outline-color: rgba(16,185,129,0.35) !important; }
div[data-id="stRadio"] label[data-baseweb="radio"]:nth-child(4):has(input:checked){ background: rgba(245,158,11,0.12) !important; outline-color: rgba(245,158,11,0.40) !important; }


/* === Mobile menu (Calomil-ish) === */
.km-wrap{max-width:760px;margin:0 auto;}
.km-card{border:1px solid rgba(0,0,0,0.08); border-radius:16px; padding:12px 14px; background:rgba(255,255,255,0.92); box-shadow:0 1px 6px rgba(0,0,0,0.04);}
.km-muted{color:rgba(0,0,0,0.55); font-size:0.85rem;}
.km-title{font-weight:700; font-size:1.05rem; margin:0 0 6px 0;}
.km-grid button[kind="secondary"], .km-grid button[kind="primary"]{width:100%;}
.km-bigbtn button{height:68px !important; border-radius:16px !important; font-weight:800 !important; font-size:18px !important;}
.km-bigbtn .stButton>button{width:100%;}
@media (max-width: 640px){
  .km-bigbtn button{height:76px !important; font-size:20px !important;}
}
.km-topbar{display:flex; gap:8px; align-items:center; justify-content:space-between; margin:8px 0 14px;}
.km-navbtn .stButton>button{border-radius:14px; padding:10px 12px; width:100%;}
.km-bottom{position:sticky; bottom:0; z-index:10; padding:10px 0 8px 0; background:linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.65), rgba(255,255,255,0));}
.km-thumb img{border-radius:12px !important;}

</style>
    """, unsafe_allow_html=True)

# =========================
# Utils
# =========================

def now_jst():
    return datetime.now(TZ)


# =========================
# Streak & Medal (Duolingo-like)
# =========================
MEDALS = [
    (30, "🏆 スペシャル"),
    (14, "🥇 ゴールド"),
    (7,  "🥈 シルバー"),
    (3,  "🥉 ブロンズ"),
]



def calc_daily_targets(weight_kg: float, goal: str) -> dict:
    """ざっくりの1日目標（kcal/P/C/F）を算出。
    goal: 'maintain'/'bulk'/'diet' など（UI表示名でもOK）
    - diet: -2kg/月 ≒ -500kcal/日を目安（成長期は下げすぎ防止の下限あり）
    戻り値は 'kcal','p','c','f' を必ず含み、互換のため 'p_g','c_g','f_g' も同梱。
    """
    try:
        w = float(weight_kg)
    except Exception:
        w = 0.0
    if w <= 0:
        w = float(st.session_state.get("profile_weight_kg") or 0.0) or 45.0

    # プロフィールから年齢/性別/身長を推定
    sex = str(st.session_state.get("pf_sex") or st.session_state.get("sex") or "M")
    try:
        h = float(st.session_state.get("pf_height") or st.session_state.get("height_cm") or 165.0)
    except Exception:
        h = 165.0
    dob = st.session_state.get("pf_dob") or st.session_state.get("dob")

    # 年齢推定（失敗してもOK）
    try:
        age = 16.0
        if dob:
            from datetime import datetime, date
            if isinstance(dob, str):
                d = datetime.fromisoformat(dob).date()
            elif hasattr(dob, "year"):
                d = dob
            else:
                d = None
            if d:
                today = now_jst().date()
                age = (today - d).days / 365.25
    except Exception:
        age = 16.0

    # BMR (Mifflin-St Jeor)
    s_const = 5 if sex.upper().startswith("M") else -161
    bmr = 10.0*w + 6.25*h - 5.0*age + s_const

    # 活動係数（アスリート寄りのざっくり）
    try:
        activity = float(st.session_state.get("activity_factor") or 1.6)
    except Exception:
        activity = 1.6
    tdee = bmr * activity

    g = str(goal or "").lower()
    if ("diet" in g) or ("ダイエット" in g) or ("減量" in g):
        kcal = tdee - 500.0
        p_g = 1.8 * w
        f_g = 0.8 * w
    elif ("bulk" in g) or ("増量" in g):
        kcal = tdee + 300.0
        p_g = 1.8 * w
        f_g = 1.0 * w
    else:  # maintain / default
        kcal = tdee
        p_g = 1.6 * w
        f_g = 0.9 * w

    # 成長期の下限（下げすぎ防止）
    if age < 18:
        kcal_floor = max(1600.0, 30.0*w)  # 目安
    else:
        kcal_floor = max(1200.0, 18.0*w)
    kcal = max(kcal, kcal_floor)

    # carbs remainder
    kcal_p = p_g * 4.0
    kcal_f = f_g * 9.0
    c_g = max(0.0, (kcal - kcal_p - kcal_f) / 4.0)

    kcal_r = float(round(kcal))
    p_r = float(round(p_g))
    f_r = float(round(f_g))
    c_r = float(round(c_g))

    return {
        "kcal": kcal_r,
        "p": p_r,
        "c": c_r,
        "f": f_r,
        # backward compatible keys
        "p_g": p_r,
        "c_g": c_r,
        "f_g": f_r,
        "age": float(round(age, 1)),
        "activity_factor": float(activity),
    }

def calc_medal(streak: int) -> str:
    for days, name in MEDALS:
        if streak >= days:
            return name
    return "—"

def update_streak_on_save(code_hash: str):
    """Call this after any daily 'save' action (training/meal/sleep/injury).
    Stores streak and medal in snapshots so it persists across days/devices."""
    try:
        today = now_jst().date().isoformat()
        last = load_snapshot(code_hash, "streak_last_date")
        streak = int(load_snapshot(code_hash, "streak_count") or 0)

        if last == today:
            pass
        else:
            if last:
                try:
                    last_d = date.fromisoformat(str(last))
                except Exception:
                    last_d = None
            else:
                last_d = None

            if last_d and last_d == (now_jst().date() - timedelta(days=1)):
                streak += 1
            else:
                streak = 1

            save_snapshot(code_hash, "streak_last_date", today)
            save_snapshot(code_hash, "streak_count", streak)

        save_snapshot(code_hash, "streak_medal", calc_medal(streak))
    except Exception:
        # streak should never break core features
        return

def render_streak_medal(code_hash: str):
    streak = int(load_snapshot(code_hash, "streak_count") or 0)
    medal  = load_snapshot(code_hash, "streak_medal") or "—"
    st.markdown(
        f"""
        <div style="padding:12px 14px;border-radius:16px;
                    background:#fff7ed;border:1px solid #fed7aa;">
          <div style="font-size:16px;font-weight:700;">🔥 連続 {streak} 日</div>
          <div style="font-size:18px;margin-top:6px;font-weight:600;">{medal}</div>
          <div style="color:#666;font-size:13px;margin-top:4px;">
            1日にどれか1つ記録できたらカウントされます
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
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


# -------------------------
# UI helpers: header/logo & AI comment persistence display
# -------------------------

def jams_logo_header():
    """Show JAMS logo at the top of the page if available."""
    p = _find_jams_logo_path()
    if not p:
        return
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(p, width=220)

def premium_css():
    """Lightweight premium-ish UI (kids-friendly, readable)."""
    st.markdown(
        """
<style>
/* Larger base font for kids */
html, body, [class*="css"]  { font-size: 16px !important; }

/* Make tab labels bigger & easier to tap */
div[data-baseweb="tab"] button {
  font-size: 16px !important;
  padding: 12px 14px !important;
  border-radius: 14px !important;
}

/* AI highlight card */
.ai-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255, 250, 235, 0.9);
  box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}
.ai-title {
  font-weight: 700;
  font-size: 16px;
  margin-bottom: 8px;
}
.ai-text { font-size: 15px; line-height: 1.7; white-space: pre-wrap; }

/* Section header card */
.section-card {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 18px;
  padding: 14px 16px;
  background: #ffffff;
  box-shadow: 0 8px 18px rgba(0,0,0,0.05);
  margin: 10px 0 14px 0;
}
.section-card h2 { margin: 0; font-size: 18px; }
.section-card p { margin: 6px 0 0 0; color: rgba(0,0,0,0.6); }

/* Buttons: slightly rounded */
button[kind="secondary"], button[kind="primary"] { border-radius: 12px !important; }
</style>
        """,
        unsafe_allow_html=True,
    )

def ai_highlight_box(title: str, text: str):
    if not text:
        return
    st.markdown(
        f"""<div class="ai-card">
  <div class="ai-title">✨ {title}</div>
  <div class="ai-text">{text}</div>
</div>""",
        unsafe_allow_html=True,
    )


def normalize_training_headings(text: str) -> str:
    """
    筋トレメニュー内の見出しをすべて同一フォント・同一サイズに統一する
    - Markdown見出し（### 上半身トレーニング 等）も除去
    - 表記ゆれ（上半身 / 上半身トレーニング 等）も吸収
    """
    if not text:
        return text

    head_keywords = [
        "上半身",
        "下半身",
        "体幹",
        "4週間の進め方",
        "４週間の進め方",
        "4週間",
        "４週間",
    ]

    lines = text.splitlines()
    out = []

    for line in lines:
        raw = line.strip()
        raw = raw.lstrip("#").strip()
        raw_clean = raw.strip("【】[]()（）:：・- ")

        matched = None
        for kw in head_keywords:
            if kw in raw_clean:
                matched = raw_clean
                break

        if matched:
            out.append(
                (
                    "<div style=\""
                    "font-weight:800;"
                    "font-size:18px;"
                    "margin:14px 0 8px 0;"
                    "padding:6px 0;"
                    "border-bottom:2px solid rgba(0,0,0,0.08);"
                    "\">"
                    f"{matched}"
                    "</div>"
                )
            )
        else:
            out.append(line)

    return "\n".join(out)


def strip_html_simple(s: str) -> str:
    """Very small HTML stripper for saving text to logs (prevents <div> tags from appearing)."""
    if not s:
        return s
    # common line breaks
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    # remove tags
    s = re.sub(r"<[^>]+>", "", s)
    # unescape entities
    try:
        import html as _html
        s = _html.unescape(s)
    except Exception:
        pass
    # normalize newlines
    s = s.replace("\r\n", "\n")
    return s.strip()




def saved_ai_footer(items):
    """Footer area where saved comments are shown + copy buttons."""
    st.markdown("---")
    st.subheader("📌 保存したAIコメント")
    shown = False
    for item in items:
        key = item.get("key")
        title = item.get("title", key)
        text = st.session_state.get(key, "") if key else ""
        if not text:
            continue
        shown = True
        with st.expander(title, expanded=False):
            copy_button("このコメントをコピー", text, key=f"copy_{key}")
            download_text_button("TXTで保存", text, filename=f"{title}.txt", key=f"dl_{key}")
            st.caption("コピーしたら、スマホのメモやLINEの『自分だけのトーク』に保存しておくのがおすすめです。")
            st.text_area("内容", value=text, height=180, key=f"ta_{key}")
    if not shown:
        st.info("保存済みのAIコメントはまだありません。AIでコメントを作るとここに残ります。")



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
# Login ()
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


DB_LOCK = threading.Lock()

def _connect_db():
    # Streamlit Cloud: concurrent reruns can hit sqlite locks. Use timeout + busy_timeout + WAL.
    conn = sqlite3.connect(DATA_DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

# =========================
# Data DB
# =========================
def data_db():
    return _connect_db()
def _with_db_retry(fn, *, attempts: int = 3, sleep_s: float = 0.15):
    last = None
    for i in range(attempts):
        try:
            with DB_LOCK:
                return fn()
        except sqlite3.OperationalError as e:
            last = e
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(sleep_s * (i + 1))
                continue
            raise
    if last:
        raise last



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
    def _op():
        conn = data_db()
        try:
            conn.execute(
                "INSERT INTO snapshots(code_hash, kind, updated_at, payload_json) VALUES(?,?,?,?) "
                "ON CONFLICT(code_hash, kind) DO UPDATE SET updated_at=excluded.updated_at, payload_json=excluded.payload_json",
                (code_hash, kind, iso(now_jst()), json.dumps(payload, ensure_ascii=False, default=str))
            )
            conn.commit()
        finally:
            conn.close()
    return _with_db_retry(_op)

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



# =====================
# Meal (by date) persistence helpers
# =====================
def _meal_date_key(d) -> str:
    """Return YYYY-MM-DD for a date or date-like."""
    if isinstance(d, date):
        return d.isoformat()
    try:
        # already string
        return str(d)
    except Exception:
        return now_jst().date().isoformat()

def meal_snapshot_kind(d) -> str:
    return f"meal_day_{_meal_date_key(d)}"

def save_meal_day_snapshot(code_hash: str, d, payload: dict):
    # payload should include "date" (YYYY-MM-DD)
    save_snapshot(code_hash, meal_snapshot_kind(d), payload)

def load_meal_day_snapshot(code_hash: str, d):
    return load_snapshot(code_hash, meal_snapshot_kind(d))

def meal_draft_kind(d) -> str:
    return f"meal_draft_{_meal_date_key(d)}"

def save_meal_day_draft(code_hash: str, d, payload: dict):
    save_snapshot(code_hash, meal_draft_kind(d), payload)

def load_meal_day_draft(code_hash: str, d):
    return load_snapshot(code_hash, meal_draft_kind(d))





# =========================
# Global Weight Sync (profile -> all tabs)
# =========================
WEIGHT_KEYS = ["pf_weight", "meal_weight", "tr_weight", "h_w3"]

def _get_profile_snapshot(code_hash: str) -> dict:
    return load_snapshot(code_hash, "profile") or {}

def _get_profile_weight_kg_from_snapshot(prof: dict) -> float:
    for k in ("weight_kg", "weight", "wt"):
        v = prof.get(k)
        if v is None:
            continue
        try:
            w = float(v)
            if 10.0 <= w <= 200.0:
                return w
        except Exception:
            pass
    return 0.0

def _set_profile_weight_kg_in_snapshot(code_hash: str, w: float):
    prof = _get_profile_snapshot(code_hash)
    prof["weight_kg"] = float(w)
    save_snapshot(code_hash, "profile", prof)

def _is_manual(key: str) -> bool:
    return bool(st.session_state.get(f"{key}__manual", False))

def _mark_manual(key: str):
    st.session_state[f"{key}__manual"] = True

def _set_global_weight(code_hash: str, w: float, *, write_back_profile: bool = True):
    """Update global weight (profile_weight_kg) and persist to profile snapshot.

    IMPORTANT:
    Do NOT write into other widget keys here. Streamlit forbids mutating a widget's session_state
    key after that widget has been instantiated in the current run.
    Propagation to other tabs is handled safely at the very top of the script via
    _sync_weight_defaults_before_render(), which runs BEFORE any widgets are created.
    """
    try:
        w = float(w)
    except Exception:
        return
    if not (10.0 <= w <= 200.0):
        return

    st.session_state["profile_weight_kg"] = w
    st.session_state["la_weight_kg"] = w  # backward-compat

    if write_back_profile:
        _set_profile_weight_kg_in_snapshot(code_hash, w)

def _sync_weight_defaults_before_render(code_hash: str, *, fallback: float = 45.0):
    """Call this early in main() before routing/UI to ensure all tabs use profile weight as baseline."""
    prof = _get_profile_snapshot(code_hash)
    w_prof = _get_profile_weight_kg_from_snapshot(prof)
    if w_prof <= 0:
        w_prof = float(st.session_state.get("profile_weight_kg") or 0) or fallback

    # set global weight if not already set
    if float(st.session_state.get("profile_weight_kg") or 0) <= 0:
        st.session_state["profile_weight_kg"] = w_prof
    st.session_state["la_weight_kg"] = float(st.session_state["profile_weight_kg"])

    # seed widget keys BEFORE they are created (safe). If a key was manually edited, keep it.
    for k in WEIGHT_KEYS:
        if k not in st.session_state or float(st.session_state.get(k) or 0) <= 0:
            st.session_state[k] = float(st.session_state["profile_weight_kg"])
        elif (not _is_manual(k)) and k != "pf_weight":
            # keep in sync for auto-derived keys
            st.session_state[k] = float(st.session_state["profile_weight_kg"])

def _weight_on_change(code_hash: str, key: str, *, write_back_profile: bool = True):
    """on_change callback for weight inputs."""
    _mark_manual(key)
    w = st.session_state.get(key)
    _set_global_weight(code_hash, w, write_back_profile=write_back_profile)


def save_record(code_hash: str, kind: str, payload: dict, result: dict):
    def _op():
        conn = data_db()
        try:
            conn.execute(
                "INSERT INTO records(created_at, code_hash, kind, payload_json, result_json) VALUES(?,?,?,?,?)",
                (iso(now_jst()), code_hash, kind,
                 json.dumps(payload, ensure_ascii=False, default=str),
                 json.dumps(result, ensure_ascii=False, default=str))
            )
            conn.commit()
        finally:
            conn.close()
    return _with_db_retry(_op)

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
    def _op():
        conn = data_db()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM snapshots WHERE code_hash=? AND kind=?", (code_hash, kind))
            conn.commit()
        finally:
            conn.close()
    _with_db_retry(_op)

def delete_record_by_id(record_id: int) -> None:
    conn = sqlite3.connect(DATA_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM records WHERE id=?", (int(record_id),))
    conn.commit()
    conn.close()

def delete_la_record(code_hash: str, kind: str) -> bool:
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
def auto_fill_from_la_records(code_hash: str):
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
                ("h_t","osterone"), ("h_e2","estradiol"),
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

def save_training_la(code_hash: str):
    payload = {k: st.session_state.get(k) for k in TRAINING_KEYS}
    if isinstance(payload.get("tr_date"), date):
        payload["tr_date"] = payload["tr_date"].isoformat()
    save_snapshot(code_hash, "training_la", payload)
    save_record(code_hash, "training_log", payload, {"summary":"training_log"})

def load_training_la(code_hash: str) -> bool:
    pl = load_snapshot(code_hash, "training_la")
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

def auto_fill_la_all_tabs(code_hash: str):
    """基本情報入力後に、保存済み最新データを各タブの入力欄へ自動反映（初回のみ）"""
    if st.session_state.get("_auto_filled_all", False):
        return
    # 必須：生年月日が入っているときだけ
    if not st.session_state.get("dob"):
        return

    # まず snapshots（下書き）を優先
    for kind, keys in [
        ("height_draft", ["h_desired","h_date_y1","h_date_y2","h_date_y3","h_y1","h_y2","h_y3","h_w1","h_w2","h_w3","h_alp","h_ba","h_igf1","h_t","h_e2"]),
        ("anemia_draft", ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end__type"]),
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
                ("h_t","osterone"), ("h_e2","estradiol"),
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
    # Meal la
    for r in rows:
        if r.get("kind") == "meal_day":
            pl = r.get("payload") or {}
            _set_if_empty("meal_goal", pl.get("goal"))
            _set_if_empty("meal_intensity", pl.get("intensity"))
            _set_if_empty("meal_weight", pl.get("weight"))
            break

    st.session_state["_auto_filled_all"] = True

def shared_demographics():
    jams_logo_header()
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
    # 基本情報ボタン（縦並び：読み込み → 保存）
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

    if st.button("基本情報を保存", key="basic_save"):
        try:
            save_basic_info_snapshot(sha256_hex(st.session_state.get("user","")))
            st.success("基本情報を保存しました。")
        except Exception as e:
            st.error(f"保存に失敗: {e}")


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



def ai_text(system: str, user: str, *, model: str = "gpt-4.1-mini", temperature: float = 0.3, max_output_tokens: int = 700):
    """テキスト生成ヘルパー。成功時 (text, None) / 失敗時 ("", err)"""
    client, err = openai_client()
    if err or client is None:
        return "", err or "no client"
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system or ""}]},
                {"role": "user", "content": [{"type": "input_text", "text": user or ""}]},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return (resp.output_text or "").strip(), None
    except Exception as e:
        return "", str(e)




def analyze_meal_photo(img_bytes: bytes, meal_type: str):
    """
    食事写真を解析して、量感（少/普/多）と特徴、食事内容の要約を返す。
    返却: dict {is_food, carb, protein, veg, fat, fried_or_oily, dairy, fruit, items, note, confidence}
    """
    client, err = openai_client()
    if err:
        return None, err

    prompt = f"""画像が「食事の写真」かどうかをまず判定してください。
食事でない場合は is_food=false とし、他の推定は空 or 低信頼で返してください。

食事の場合:
- 主食/主菜/野菜の量感を Aレベル（少/普/多）で推定
- 揚げ物・油っぽさ、乳製品、果物の有無を推定
- 料理名や食材を items に箇条書きで（推測でOK）
- note に短い要約を1文で
- confidence は0-1

JSONのみで返してください。キー:
is_food(boolean), carb, protein, veg, fat("少/普/多"),
fried_or_oily(boolean), dairy(boolean), fruit(boolean),
items(array of string), note(string), confidence(number)
"""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ],
            }],
            temperature=0.2,
            max_output_tokens=600,
        )
        text = (resp.output_text or "").strip()
        # JSON抽出（余計な文字が混じる場合に備える）
        m = re.search(r'\{.*\}', text, flags=re.S)
        j = m.group(0) if m else text
        data = json.loads(j)
        # normalize
        data.setdefault("is_food", True)
        data.setdefault("confidence", 0.0)
        return data, None
    except Exception as e:
        return None, str(e)


def merge_meal_analyses(items: list[dict]) -> dict:
    """
    複数枚の食事写真解析結果を統合する。
    - 量感（少/普/多）は多数決（同票は「普」寄り）
    - 有無フラグは OR
    - items/note は統合
    """
    if not items:
        return {"is_food": False, "confidence": 0.0}

    def vote_level(key: str) -> str:
        vals = [d.get(key) for d in items if d.get(key) in ("少", "普", "多")]
        if not vals:
            return "普"
        counts = {"少": 0, "普": 0, "多": 0}
        for v in vals:
            counts[v] += 1
        # 同票は普を優先
        best = max(counts.items(), key=lambda kv: (kv[1], 1 if kv[0] == "普" else 0))[0]
        return best

    merged = {
        "is_food": True,
        "confidence": max(float(d.get("confidence") or 0.0) for d in items),
        "carb": vote_level("carb"),
        "protein": vote_level("protein"),
        "veg": vote_level("veg"),
        "fat": vote_level("fat"),
        "fried_or_oily": any(bool(d.get("fried_or_oily")) for d in items),
        "dairy": any(bool(d.get("dairy")) for d in items),
        "fruit": any(bool(d.get("fruit")) for d in items),
    }

    # items を統合（重複除去、順序保持）
    seen = set()
    merged_items = []
    for d in items:
        lst = d.get("items") or []
        if isinstance(lst, str):
            lst = [s.strip() for s in lst.split("\n") if s.strip()]
        for s in lst:
            s = str(s).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            merged_items.append(s)
    merged["items"] = merged_items

    notes = []
    for d in items:
        n = (d.get("note") or "").strip()
        if n and n not in notes:
            notes.append(n)
    merged["note"] = " / ".join(notes)[:500]

    return merged


def ai_comment_for_meal(meal_title: str, est: dict, targets: dict):
    """短い食事コメントを生成。est/targetsは {p,c,f,kcal} を想定。"""
    def _num(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return float(d)

    ek = _num(est.get("kcal"))
    ep = _num(est.get("p"))
    ec = _num(est.get("c"))
    ef = _num(est.get("f"))

    tk = _num(targets.get("kcal"))
    tp = _num(targets.get("p"))
    tc = _num(targets.get("c"))
    tf = _num(targets.get("f"))

    system = (
        "You are a nutrition coach for youth athletes in Japan. "
        "Be concise and practical. Output Japanese. "
        "Do not mention 'AI' or uncertainties. "
        "Avoid medical diagnosis. "
        "Use 3-6 bullet points. "
    )
    user = f"""食事: {meal_title}
推定: kcal={ek:.0f}, P={ep:.1f}g, C={ec:.1f}g, F={ef:.1f}g
目標(1日): kcal={tk:.0f}, P={tp:.1f}g, C={tc:.1f}g, F={tf:.1f}g

この食事について、次の観点でコメントして:
- 良い点
- 足りない/多い場合の調整案（食材例）
- 次の食事で意識する一言
"""
    text, err = ai_text(system, user)
    if err:
        raise RuntimeError(err)
    return (text or "").strip()


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
    osterone = st.number_input("テストステロン（任意）", 0.0, 3000.0, step=1.0, key="h_t")
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
    w3 = col3.number_input("体重 最新(kg)", 0.0, 200.0,
                        value=float(st.session_state.get("h_w3") or st.session_state.get("profile_weight_kg") or 0.0),
                        step=0.1, key="h_w3",
                        on_change=lambda: _weight_on_change(code_hash, "h_w3", write_back_profile=False))

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
            "osterone": osterone, "estradiol": estradiol,
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


def estimate_endurance_gain(_kind: str, baseline_value: float, hb_now: float, hb_pred: float, ferr_now: float | None, ferr_pred: float | None):
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
    if _kind == "yoyo" and baseline_value >= 2000:
        pct *= 0.6
    if _kind == "shuttle" and baseline_value >= 130:
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
        keys = ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end__type"]
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
    end__type = st.selectbox("入力するテスト", ["シャトルラン（回数）", "Yo-Yo（距離m）"], index=0, key="end__type")
    end_current = st.number_input("現在の記録（回数 or 距離）", min_value=0.0, max_value=99999.0, value=float(st.session_state.get("end_current", 0.0) or 0.0), step=1.0, key="end_current")
    st.caption("※入力は任意。入力すると、Hb改善に伴う伸びを参考推定します（個人差あり）。")
    if st.button("結果保存（持久力）", key="save_endurance_baseline"):
        save_record(code_hash, "endurance_baseline", {"": st.session_state.get("end__type",""), "current": float(st.session_state.get("end_current",0.0) or 0.0), "hb": float(hb_v or 0.0), "ferritin": float(ferr_v or 0.0), "tsat": float(tsat_val or 0.0)}, {"summary":"endurance_baseline"})
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
        keys = ["sa_hb","sa_ferr","sa_fe","sa_tibc","sa_tsat","sa_riona","end_current","end__type"]
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
        end__type = st.session_state.get("end__type", "シャトルラン（回数）")
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
            st.caption(f"入力テスト：{end__type} / 現在：{end_current:.0f}")
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
    """1日のPFC目標をざっくり推定する（スマホ入力向けの簡易ロジック）。

    goal:
      - 増量 / 維持 / 回復 / ダイエット（-2kg/月 目安）
    """
    if weight_kg <= 0:
        return None

    # ベース（成長期は少し高め、成人はやや低め）
    base = 45.0 if age_years < 12 else (50.0 if age_years < 15 else 48.0)

    sport_factor = {"サッカー": 1.05, "ラグビー": 1.10, "野球": 1.00, "テニス": 1.00, "水泳": 1.08}.get(sport, 1.0)
    intensity_factor = {"低": 0.95, "中": 1.00, "高": 1.10}.get(intensity, 1.0)

    # まず維持カロリーの粗推定
    maint_kcal = weight_kg * base * sport_factor * intensity_factor

    if goal == "ダイエット":
        # -2kg/月 ≒ -500kcal/日 の目安（個人差あり）
        kcal = maint_kcal - 500.0

        # 成長期の下げ過ぎ防止：最低ライン（ざっくり）
        # 体重×30kcal を下回らないようにする（年齢が若いほど安全側）
        min_kcal = weight_kg * (32.0 if age_years < 15 else 30.0)
        kcal = max(kcal, min_kcal)

        # 筋量維持優先でたんぱく質は高め
        p_perkg = 2.0
        f_pct = 0.25
    else:
        goal_factor = {"増量": 1.08, "維持": 1.00, "回復": 1.03}.get(goal, 1.0)
        kcal = maint_kcal * goal_factor
        p_perkg = {"増量": 1.8, "維持": 1.6, "回復": 2.0}.get(goal, 1.6)
        f_pct = 0.25 if goal in ["増量", "維持"] else 0.28

    p_g = p_perkg * weight_kg
    f_g = (kcal * f_pct) / 9.0
    c_g = max(0.0, kcal - p_g * 4.0 - f_g * 9.0) / 4.0
    return {"kcal": kcal, "p_g": p_g, "c_g": c_g, "f_g": f_g}

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
    食事1回分の入力（写真 + ざっくり量選択 + AI推定）
    - 写真は st.file_uploader（スマホではカメラ/アルバム選択へ）
    - 量はユーザーが「少/普/多」を選択（写真で伝わりづらい時の補正にも使う）
    - 「AIで写真から初期値セット」で、選択肢の初期値を自動入力（カロミル風）
    """
    st.markdown(f"#### {title}")

    # --- 現在値（ユーザー選択） ---
    def _init_sel(k: str, default):
        if k not in st.session_state:
            st.session_state[k] = default

    _init_sel(f"{prefix}_sel_carb", "普")
    _init_sel(f"{prefix}_sel_protein", "普")
    _init_sel(f"{prefix}_sel_veg", "普")
    _init_sel(f"{prefix}_sel_fat", "普")
    _init_sel(f"{prefix}_sel_fried", False)
    _init_sel(f"{prefix}_sel_dairy", False)
    _init_sel(f"{prefix}_sel_fruit", False)

    img_bytes = None

    # --- 写真（任意） ---
    if enable_photo:
        ups = st.file_uploader(
            f"{title}の写真（カメラ/アルバム）",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            key=f"{prefix}_photos",
            accept_multiple_files=True,
        )
        if ups:
            # 複数枚サムネ（小さめ）
            cols = st.columns(min(3, len(ups)))
            img_list = []
            for i, f in enumerate(ups):
                try:
                    b = f.getvalue()
                except Exception:
                    b = None
                if not b:
                    continue
                img_list.append(b)
                with cols[i % len(cols)]:
                    st.image(b, width=120)
            # 最初の1枚を代表として拡大表示
            if img_list and st.button("拡大表示", key=f"{prefix}_photo_zoom_grid"):
                st.image(img_list[0], use_container_width=True)


            # 小サムネ（場所を取りすぎない）
            st.image(img_bytes, caption=None, width=160)

            # 拡大（ページ内）
            if st.button("拡大表示", key=f"{prefix}_photo_zoom_single"):
                st.image(img_bytes, caption=None, use_container_width=True)

            # AIで初期値セット（写真から、少/普/多を推測）
            if st.button("AIで写真から初期値セット", key=f"{prefix}_ai_set_btn"):
                if not require_premium_ai(code_hash):
                    return st.session_state.get(est_key) or {"p":0.0,"c":0.0,"f":0.0,"kcal":0.0}
                # 複数枚の結果をまとめて、少/普/多をざっくり推測
                results = []
                for b in img_list:
                    out1, err1 = analyze_meal_photo(b, title)
                    if err1 or (out1 is None):
                        continue
                    results.append(out1)
                if not results:
                    st.error("写真解析に失敗しました。別の写真でお試しください。")
                else:
                    # 食事判定：過半数が食事であること
                    food_votes = sum(1 for r in results if bool(r.get("is_food")))
                    conf_max = max(float(r.get("confidence") or 0.0) for r in results)
                    if food_votes < (len(results) / 2) and conf_max >= 0.35:
                        st.error("この画像は食事写真として判定できませんでした。食事が写る写真でお願いします。")
                    else:
                        # mode（多数決）
                        def _mode(key, default="普"):
                            vals = [r.get(key) for r in results if r.get(key) in ("少","普","多")]
                            if not vals:
                                return default
                            return max(set(vals), key=vals.count)
                        st.session_state[f"{prefix}_sel_carb"] = _mode("carb", st.session_state[f"{prefix}_sel_carb"])
                        st.session_state[f"{prefix}_sel_protein"] = _mode("protein", st.session_state[f"{prefix}_sel_protein"])
                        st.session_state[f"{prefix}_sel_veg"] = _mode("veg", st.session_state[f"{prefix}_sel_veg"])
                        st.session_state[f"{prefix}_sel_fat"] = _mode("fat", st.session_state[f"{prefix}_sel_fat"])
                        st.session_state[f"{prefix}_fried"] = any(bool(r.get("fried_or_oily")) for r in results)
                        st.session_state[f"{prefix}_dairy"] = any(bool(r.get("dairy")) for r in results)
                        st.session_state[f"{prefix}_fruit"] = any(bool(r.get("fruit")) for r in results)
                        # items/note
                        items = []
                        for r in results:
                            for it in (r.get("items") or []):
                                if isinstance(it, str) and it and it not in items:
                                    items.append(it)
                        note = " / ".join([r.get("note") for r in results if isinstance(r.get("note"), str) and r.get("note")][:2])
                        st.session_state[f"{prefix}_ai_items"] = items
                        st.session_state[f"{prefix}_ai_note"] = note
                        st.success("AIが写真から量を推測しました（必要なら下の調整で微修正できます）。")


            # 食事判定ガード（非食事の誤爆対策）
                    conf = float(out.get("confidence", 0.0) or 0.0)
                    if conf < 0.35:
                        st.warning("食事写真として判定できませんでした。食事が写るように撮り直すか、下の量選択で入力してください。")
                    else:
                        st.session_state[f"{prefix}_sel_carb"] = out.get("carb", "普")
                        st.session_state[f"{prefix}_sel_protein"] = out.get("protein", "普")
                        st.session_state[f"{prefix}_sel_veg"] = out.get("veg", "普")
                        st.session_state[f"{prefix}_sel_fat"] = out.get("fat", "普")
                        st.session_state[f"{prefix}_sel_fried"] = bool(out.get("fried_or_oily", False))
                        st.session_state[f"{prefix}_sel_dairy"] = bool(out.get("dairy", False))
                        st.session_state[f"{prefix}_sel_fruit"] = bool(out.get("fruit", False))
                        st.success("量の初期値をセットしました（必要なら下で調整してください）")
                        # 古い評価をクリア
                        st.session_state.pop(f"{prefix}_comment", None)

    # --- 量選択（カロミル風：写真 + ざっくり量で推測） ---
    st.caption("写真だけで伝わりにくい時は、下の「少/普/多」でざっくり補正してください。")
    c1, c2 = st.columns(2)
    with c1:
        carb = st.selectbox("主食（ごはん/パン/麺）", ["少", "普", "多"],
                            index=["少", "普", "多"].index(st.session_state[f"{prefix}_sel_carb"]),
                            key=f"{prefix}_sel_carb")
        protein = st.selectbox("主菜（肉/魚/卵/豆）", ["少", "普", "多"],
                               index=["少", "普", "多"].index(st.session_state[f"{prefix}_sel_protein"]),
                               key=f"{prefix}_sel_protein")
        veg = st.selectbox("野菜", ["少", "普", "多"],
                           index=["少", "普", "多"].index(st.session_state[f"{prefix}_sel_veg"]),
                           key=f"{prefix}_sel_veg")
    with c2:
        fat = st.selectbox("油もの（揚げ物/マヨ/ドレ）", ["少", "普", "多"],
                           index=["少", "普", "多"].index(st.session_state[f"{prefix}_sel_fat"]),
                           key=f"{prefix}_sel_fat")
        fried = st.toggle("揚げ物・油多め", value=bool(st.session_state[f"{prefix}_sel_fried"]), key=f"{prefix}_sel_fried")
        dairy = st.toggle("乳製品あり", value=bool(st.session_state[f"{prefix}_sel_dairy"]), key=f"{prefix}_sel_dairy")
        fruit = st.toggle("果物あり", value=bool(st.session_state[f"{prefix}_sel_fruit"]), key=f"{prefix}_sel_fruit")

    # 推定（ユーザー選択を反映）
    est = meal_estimate(carb, protein, veg, bool(fried), bool(dairy), bool(fruit))

    # 表示（推定値）
    st.markdown("##### 推定PFC / kcal")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P", f"{est['p']:.0f} g")
    m2.metric("C", f"{est['c']:.0f} g")
    m3.metric("F", f"{est['f']:.0f} g")
    m4.metric("kcal", f"{est['kcal']:.0f}")

    # 1食コメント（必要な時だけ）
    if st.button("この食事のAIコメント", key=f"{prefix}_ai_comment_btn"):
        try:
            comment = ai_comment_for_meal(title, est, targets)
            st.session_state[f"{prefix}_comment"] = comment
        except Exception as e:
            st.session_state[f"{prefix}_comment"] = "コメント生成に失敗しました: " + str(e)

    comment = st.session_state.get(f"{prefix}_comment")
    if comment:
        st.markdown("##### AIコメント")
        st.write(comment)

    # meal_pageが保存できる形で返す
    payload = {
        "p": float(est["p"]),
        "c": float(est["c"]),
        "f": float(est["f"]),
        "kcal": float(est["kcal"]),
        "menu": "",
        "ai_levels": {"carb": carb, "protein": protein, "veg": veg, "fat": fat, "fried": bool(fried), "dairy": bool(dairy), "fruit": bool(fruit)},
        "sel": {
            "carb": carb, "protein": protein, "veg": veg, "fat": fat,
            "fried": bool(fried), "dairy": bool(dairy), "fruit": bool(fruit)
        },
    }
    return payload



def estimate_macros_from_levels(ai_levels: dict, weight_kg: float, goal: str) -> dict:
    """AIの量感（少/普/多）からP/C/F/kcalをざっくり推定。
    現状は meal_estimate をベースにし、items/note/levels を付与する。
    """
    if not isinstance(ai_levels, dict):
        ai_levels = {}
    carb = ai_levels.get("carb", "普") if ai_levels.get("carb") in ("少", "普", "多") else "普"
    protein = ai_levels.get("protein", "普") if ai_levels.get("protein") in ("少", "普", "多") else "普"
    veg = ai_levels.get("veg", "普") if ai_levels.get("veg") in ("少", "普", "多") else "普"
    # fat は meal_estimate では直接の係数に使っていないが、隠しUIで調整するので保持する
    fat_level = ai_levels.get("fat", "普") if ai_levels.get("fat") in ("少", "普", "多") else "普"
    fried = bool(ai_levels.get("fried_or_oily") or ai_levels.get("fried"))
    dairy = bool(ai_levels.get("dairy"))
    fruit = bool(ai_levels.get("fruit"))

    est = meal_estimate(carb, protein, veg, fried, dairy, fruit)
    est["levels"] = {
        "carb": carb,
        "protein": protein,
        "veg": veg,
        "fat": fat_level,
        "fried": fried,
        "dairy": dairy,
        "fruit": fruit,
    }
    # optional extras
    if isinstance(ai_levels.get("items"), list):
        est["items"] = ai_levels.get("items")
    else:
        est["items"] = []
    est["note"] = str(ai_levels.get("note") or "")
    return est


def _meal_ui(prefix: str, title: str, targets: dict, allow_school: bool = False):
    """食事（朝/昼/夕）のUI。
    - 写真は複数枚選択 → 「選択した写真を追加」で確定（重複防止）
    - 追加済みは小サムネ表示、1枚ずつ削除可能
    - AI解析は追加済み写真に対して実行
    - 変更・追加は隠しUIで補正（少/普/多）
    """
    photos_key = f"{prefix}_photos_store"   # list[dict(hash, bytes)]
    ai_key = f"{prefix}_ai"
    est_key = f"{prefix}_est"
    comment_key = f"{prefix}_comment"
    last_batch_key = f"{prefix}_last_batch_id"

    if photos_key not in st.session_state:
        st.session_state[photos_key] = []

    # 昼のみ：給食チェック
    if allow_school:
        is_school = st.checkbox("給食（写真なし）", key=f"{prefix}_school", value=bool(st.session_state.get(f"{prefix}_school") or False))
        if is_school:
            st.info("給食の日はチェックのみでOKです。必要なら後から写真を追加できます。")
            # 給食は簡易テンプレ（年齢でざっくり）
            age_years = float(st.session_state.get("age_years") or 12.0)
            est = kyushoku_template(age_years)
            st.session_state[est_key] = {"p": est["p"], "c": est["c"], "f": est["f"], "kcal": est["kcal"], "menu": "school", "items": ["給食"], "note": "給食（写真なし）", "levels": {}}

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("kcal", f"{est['kcal']:.0f}")
            c2.metric("タンパク質(g)", f"{est['p']:.0f}")
            c3.metric("炭水化物(g)", f"{est['c']:.0f}")
            c4.metric("脂質(g)", f"{est['f']:.0f}")

            if st.button("この食事のAIコメント", key=f"{prefix}_school_comment_btn"):
                try:
                    st.session_state[comment_key] = ai_comment_for_meal(title, st.session_state[est_key], targets)
                except Exception as e:
                    st.error(f"コメント生成に失敗しました: {e}")

            if st.session_state.get(comment_key):
                st.markdown("##### AIコメント")
                st.write(st.session_state[comment_key])

            return st.session_state[est_key]

    # ----------------------------
    # 写真アップロード（複数枚）
    # ----------------------------
    with st.container(border=True):
        ups = st.file_uploader(
            "食事の写真アップロード（カメラ/アルバム）",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            accept_multiple_files=True,
            key=f"{prefix}_up_multi",
        )

        staged = []
        if ups:
            for f in ups:
                try:
                    b = f.getvalue()
                except Exception:
                    b = None
                if not b:
                    continue
                h = hashlib.sha1(b).hexdigest()
                staged.append((h, b))

        batch_hash = None
        if staged:
            batch_hash = hashlib.sha1((",".join([h for h, _ in staged])).encode("utf-8")).hexdigest()

        cA, cB = st.columns([1, 1])
        with cA:
            add_clicked = st.button("選択した写真を追加", key=f"{prefix}_add_btn", disabled=not bool(staged))
        with cB:
            clear_clicked = st.button("すべて削除", key=f"{prefix}_clear_all_btn", disabled=(len(st.session_state[photos_key]) == 0))

        if clear_clicked:
            st.session_state[photos_key] = []
            st.session_state.pop(ai_key, None)
            st.session_state.pop(est_key, None)
            st.session_state.pop(comment_key, None)
            st.success("写真をすべて削除しました。")

        if add_clicked and staged and batch_hash:
            if st.session_state.get(last_batch_key) == batch_hash:
                st.info("この選択はすでに取り込み済みです（重複追加はしません）。")
            else:
                existing_hashes = set([p.get("hash") for p in st.session_state[photos_key]])
                new_items = []
                for h, b in staged:
                    if h in existing_hashes:
                        continue
                    new_items.append({"hash": h, "bytes": b})
                if new_items:
                    st.session_state[photos_key].extend(new_items)
                    # 最新6枚まで
                    st.session_state[photos_key] = st.session_state[photos_key][-6:]
                    st.success(f"{len(new_items)}枚を追加しました。")
                else:
                    st.info("すべて既に追加済みの写真でした。")
                st.session_state[last_batch_key] = batch_hash

        # ---- サムネ表示＋削除 ----
        photos = st.session_state.get(photos_key) or []
        if photos:
            st.caption("追加済み写真（小サムネ）")
            cols = st.columns(min(3, len(photos)))
            for i, p in enumerate(list(photos)):
                with cols[i % len(cols)]:
                    st.image(p["bytes"], width=120)
                    if st.button("削除", key=f"{prefix}_del_{p['hash']}"):
                        st.session_state[photos_key] = [x for x in st.session_state[photos_key] if x.get("hash") != p["hash"]]
                        # 解析結果はリセット（再解析）
                        st.session_state.pop(ai_key, None)
                        st.session_state.pop(est_key, None)
                        st.session_state.pop(comment_key, None)
                        st.success("削除しました。")
                        st.rerun()
        else:
            st.caption("写真を選んだあと「選択した写真を追加」を押すとサムネが出ます。")

        # ---- AI解析 ----
        can_analyze = len(st.session_state.get(photos_key) or []) > 0
        if can_analyze:
            if st.button("AI食事解析", key=f"{prefix}_analyze_btn"):
                img_bytes_list = [p["bytes"] for p in (st.session_state.get(photos_key) or [])]
                valid = []
                last_err = None
                with st.spinner("AIで解析中..."):
                    for b in img_bytes_list:
                        out1, err1 = analyze_meal_photo(b, title)
                        if err1:
                            last_err = err1
                            continue
                        # 食事判定と信頼度
                        if out1 and bool(out1.get("is_food")) and float(out1.get("confidence") or 0.0) >= 0.35:
                            valid.append(out1)
                if not valid:
                    st.error("この画像は食事写真として解析できませんでした。食事が写るように撮り直してください。")
                    if last_err:
                        st.caption(f"詳細: {last_err}")
                    st.session_state.pop(ai_key, None)
                    st.session_state.pop(est_key, None)
                    st.session_state.pop(comment_key, None)
                else:
                    merged_data = merge_meal_analyses(valid)
                    st.session_state[ai_key] = merged_data
                    w = float(st.session_state.get("meal_weight") or st.session_state.get("profile_weight_kg") or 45.0)
                    goal = st.session_state.get("meal_goal", "維持")
                    est = estimate_macros_from_levels(merged_data, w, goal)
                    st.session_state[est_key] = est
                    st.success("解析が完了しました。")
        else:
            st.caption("写真を追加すると「AI食事解析」ボタンが表示されます。")

    # ----------------------------
    # 結果表示（推定）
    # ----------------------------
    est = st.session_state.get(est_key)
    if est:
        st.markdown("##### 推定結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("kcal", f"{float(est.get('kcal',0)):.0f}")
        c2.metric("タンパク質(g)", f"{float(est.get('p',0)):.0f}")
        c3.metric("炭水化物(g)", f"{float(est.get('c',0)):.0f}")
        c4.metric("脂質(g)", f"{float(est.get('f',0)):.0f}")

        items = est.get("items") or []
        if items:
            st.caption("推定された内容: " + " / ".join([str(x) for x in items[:10]]))
        if est.get("note"):
            st.caption("補足: " + str(est.get("note")))

        if st.button("この食事のAIコメント", key=f"{prefix}_comment_btn"):
            try:
                st.session_state[comment_key] = ai_comment_for_meal(title, est, targets)
            except Exception as e:
                st.error(f"コメント生成に失敗しました: {e}")

        if st.session_state.get(comment_key):
            st.markdown("##### AIコメント")
            st.write(st.session_state[comment_key])

        # ---- 変更・追加（隠しUI） ----
        with st.expander("変更・追加（必要なときだけ）"):
            lv = (est.get("levels") or {}) if isinstance(est, dict) else {}
            carb = st.selectbox("主食の量", ["少", "普", "多"], index=["少", "普", "多"].index(lv.get("carb", "普")), key=f"{prefix}_adj_carb")
            protein = st.selectbox("主菜の量", ["少", "普", "多"], index=["少", "普", "多"].index(lv.get("protein", "普")), key=f"{prefix}_adj_protein")
            veg = st.selectbox("野菜の量", ["少", "普", "多"], index=["少", "普", "多"].index(lv.get("veg", "普")), key=f"{prefix}_adj_veg")
            fat = st.selectbox("脂質（全体）", ["少", "普", "多"], index=["少", "普", "多"].index(lv.get("fat", "普")), key=f"{prefix}_adj_fat")
            fried = st.checkbox("揚げ物/油っぽい", value=bool(lv.get("fried", False)), key=f"{prefix}_adj_fried")
            dairy = st.checkbox("乳製品あり", value=bool(lv.get("dairy", False)), key=f"{prefix}_adj_dairy")
            fruit = st.checkbox("果物あり", value=bool(lv.get("fruit", False)), key=f"{prefix}_adj_fruit")
            if st.button("再計算", key=f"{prefix}_recalc_btn"):
                new_est = meal_estimate(carb, protein, veg, fried, dairy, fruit)
                new_est["items"] = est.get("items") or []
                new_est["note"] = est.get("note") or ""
                new_est["levels"] = {"carb": carb, "protein": protein, "veg": veg, "fat": fat, "fried": fried, "dairy": dairy, "fruit": fruit}
                st.session_state[est_key] = new_est
                st.session_state.pop(comment_key, None)
                st.success("更新しました。")

        return st.session_state.get(est_key) or {"p": 0.0, "c": 0.0, "f": 0.0, "kcal": 0.0}

    return {"p": 0.0, "c": 0.0, "f": 0.0, "kcal": 0.0}


def meal_page(code_hash: str):
    st.subheader("🍽️ 食事管理（写真→AI解析）")
    st.caption("朝・昼・夕の写真をアップロードして、AIが内容を推測してフィードバックします（目安）。昼が給食の場合はチェックのみ。")

    # 記録する日付（過去にさかのぼって入力できます）
    st.session_state.setdefault("meal_date", now_jst().date())
    meal_date = st.date_input("日付", value=st.session_state.get("meal_date"), key="meal_date")

    # 日付を変えたら復元フラグをリセット（写真は新規にする）
    if st.session_state.get("_meal_last_date") != _meal_date_key(meal_date):
        st.session_state["_meal_last_date"] = _meal_date_key(meal_date)
        st.session_state["_meal_day_restored_once"] = False

        # 写真は日付ごとに分ける：日付変更時は写真だけクリア（文章ログや目標は残す）
        for pref in ["b", "l", "d"]:
            # 追加済み写真ストア
            st.session_state[f"{pref}_photos_store"] = []
            # 選択中アップロード（file_uploader）の状態もリセット
            st.session_state.pop(f"{pref}_photos", None)
            # 重複検知/一時バッチ
            st.session_state.pop(f"{pref}_last_batch_id", None)
        # 表示用の拡大ボタン状態なども日付で持たない
        st.session_state.pop("__copy_buffer", None)




    # 体重はプロフィールから初期値（ウィジェット作成前に同期済み）
    w = float(st.session_state.get("meal_weight") or st.session_state.get("profile_weight_kg") or 45.0)

    # ---- 今日の保存済み食事ログをセッションへ復元（ウィジェット生成前に行う）----
    snap_day = load_meal_day_snapshot(code_hash, meal_date)
    snap_draft = load_meal_day_draft(code_hash, meal_date) or None
    # 下書き（途中保存）があれば優先して復元/表示
    snap_base = snap_draft or snap_day
    if snap_base and not st.session_state.get("_meal_day_restored_once", False):
        st.session_state["_meal_day_restored_once"] = True
        try:
            if snap_base.get("meal_goal") is not None:
                st.session_state["meal_goal"] = snap_base.get("meal_goal")
            if snap_base.get("meal_weight") is not None:
                st.session_state["meal_weight"] = snap_base.get("meal_weight")
            # 各食事の推定結果やコメント（写真以外）を復元して、途中再開しやすくする
            for pref in ["b", "l", "d"]:
                info = snap_base.get(pref) or {}
                if isinstance(info, dict):
                    if info.get("est") is not None:
                        st.session_state[f"{pref}_est"] = info.get("est")
                    if info.get("ai") is not None:
                        st.session_state[f"{pref}_ai"] = info.get("ai")
                    if info.get("comment") is not None:
                        st.session_state[f"{pref}_comment"] = info.get("comment")
                    if "school" in info:
                        st.session_state[f"{pref}_school"] = info.get("school")
            # 量の補正（少/普/多）など
            if isinstance(snap_base.get("levels"), dict):
                lv = snap_base.get("levels") or {}
                for pref in ["b","l","d"]:
                    for k in ["sel_carb","sel_protein","sel_veg","sel_fat","sel_fried","sel_dairy","sel_fruit"]:
                        kk = f"{pref}_{k}"
                        if kk in lv:
                            st.session_state[kk] = lv.get(kk)

        except Exception:
            pass


    goal = st.selectbox("目的", ["増量", "維持", "回復", "ダイエット"], key="meal_goal", index=1)
    targets = calc_daily_targets(w, goal)

    # ---- 今日の保存済み食事ログ（表示のみ：ログアウトしても残ります）----
    if snap_base:
        with st.expander("✅ 選択した日の保存済み食事ログ（ログアウトしても残ります）", expanded=False):
            if snap_draft:
                st.info("この日付には「途中保存」があります。続きから入力して、最後に『今日の食事ログを保存』を押してください。")
            st.write("※ 写真は復元しません（容量・安定性のため）。AI推定結果とコメント、合計は復元します。")
            total_s = (snap_base.get("total") or {})
            targets_s = (snap_base.get("targets") or {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("kcal", f"{float(total_s.get('kcal',0)):.0f}", delta=f"{(float(total_s.get('kcal',0))-float(targets_s.get('kcal',0))):+.0f}")
            c2.metric("タンパク質(g)", f"{float(total_s.get('p',0)):.0f}", delta=f"{(float(total_s.get('p',0))-float(targets_s.get('p',targets_s.get('p_g',0)))):+.0f}")
            c3.metric("炭水化物(g)", f"{float(total_s.get('c',0)):.0f}", delta=f"{(float(total_s.get('c',0))-float(targets_s.get('c',targets_s.get('c_g',0)))):+.0f}")
            c4.metric("脂質(g)", f"{float(total_s.get('f',0)):.0f}", delta=f"{(float(total_s.get('f',0))-float(targets_s.get('f',targets_s.get('f_g',0)))):+.0f}")

            # 各食事のAIコメント（保存済み）を表示
            for pref, title in [("b", "朝食"), ("l", "昼食"), ("d", "夕食")]:
                info = (snap_base.get(pref) or {})
                if not isinstance(info, dict):
                    continue
                ai_val = info.get("ai")
                if ai_val is None:
                    ai_txt = ""
                elif isinstance(ai_val, dict):
                    note = str(ai_val.get("note") or "").strip()
                    items = ai_val.get("items") or []
                    def _lv(k):
                        v = ai_val.get(k)
                        return v if v in ("少","普","多") else None
                    lv_c = _lv("carb"); lv_p = _lv("protein"); lv_v = _lv("veg"); lv_f = _lv("fat")
                    parts = []
                    if any([lv_c, lv_p, lv_v, lv_f]):
                        parts.append("推定（目安）: " + " / ".join([x for x in [
                            f"炭水化物={lv_c}" if lv_c else "",
                            f"タンパク質={lv_p}" if lv_p else "",
                            f"野菜={lv_v}" if lv_v else "",
                            f"脂質={lv_f}" if lv_f else "",
                        ] if x]))
                    flags = []
                    if ai_val.get("fried_or_oily") is True: flags.append("揚げ物/油多め")
                    if ai_val.get("dairy") is True: flags.append("乳製品あり")
                    if ai_val.get("fruit") is True: flags.append("果物あり")
                    if flags:
                        parts.append("補足: " + " / ".join(flags))
                    if isinstance(items, list) and items:
                        parts.append("推定された品目: " + " / ".join([str(x) for x in items[:30]]))
                    if note:
                        parts.append("コメント: " + note)
                    ai_txt = "\\n".join([p for p in parts if p])

                elif isinstance(ai_val, list):
                    ai_txt = json.dumps(ai_val, ensure_ascii=False, default=str)
                else:
                    ai_txt = str(ai_val)
                ai_txt = ai_txt.strip()
                # 表示用の整形（"\n" などのエスケープが残っていたら改行に戻す）
                ai_txt = ai_txt.replace("\\n", "\n")
                ai_txt = ai_txt.replace("／ｎ", "\n").replace("/ｎ", "\n").replace("／n", "\n").replace("/n", "\n")
                ai_txt = ai_txt.replace("\r\n", "\n")

                cmt_val = info.get("comment")
                if cmt_val is None:
                    user_cmt = ""
                else:
                    user_cmt = str(cmt_val).strip()

                if ai_txt or user_cmt:
                    st.markdown("---")
                    st.markdown(f"#### {title}")
                if user_cmt:
                    st.markdown("**メモ（本人/保護者）**")
                    st.write(user_cmt)
                if ai_txt:
                    st.markdown("**AIコメント**")
                    st.write(ai_txt)
                    if st.button("AIコメントをコピー", key=f"copy_saved_ai_{pref}", use_container_width=True):
                        st.session_state["__copy_buffer"] = ai_txt
            if st.session_state.get("__copy_buffer"):
                with st.expander("📋 コピー用テキスト", expanded=False):
                    st.code(st.session_state.get("__copy_buffer") or "", language="text")




    st.caption(
        f"目標（1日）: kcal {targets.get('kcal',0):.0f} / "
        f"タンパク質 {targets.get('p', targets.get('p_g',0)):.0f}g / "
        f"炭水化物 {targets.get('c', targets.get('c_g',0)):.0f}g / "
        f"脂質 {targets.get('f', targets.get('f_g',0)):.0f}g"
    )

    tabs = st.tabs(["朝食", "昼食", "夕食"])
    with tabs[0]:
        b = _meal_ui("b", "朝食", targets, allow_school=False)
    with tabs[1]:
        l = _meal_ui("l", "昼食", targets, allow_school=True)
    with tabs[2]:
        d = _meal_ui("d", "夕食", targets, allow_school=False)

    total = {
        "p": float(b.get("p", 0)) + float(l.get("p", 0)) + float(d.get("p", 0)),
        "c": float(b.get("c", 0)) + float(l.get("c", 0)) + float(d.get("c", 0)),
        "f": float(b.get("f", 0)) + float(l.get("f", 0)) + float(d.get("f", 0)),
        "kcal": float(b.get("kcal", 0)) + float(l.get("kcal", 0)) + float(d.get("kcal", 0)),
    }

    st.divider()
    st.markdown("### 今日の合計（目安）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("kcal", f"{total['kcal']:.0f}", delta=f"{(total['kcal'] - targets.get('kcal', 0)):+.0f}")
    c2.metric("タンパク質(g)", f"{total['p']:.0f}", delta=f"{(total['p'] - targets.get('p', targets.get('p_g', 0))):+.0f}")
    c3.metric("炭水化物(g)", f"{total['c']:.0f}", delta=f"{(total['c'] - targets.get('c', targets.get('c_g', 0))):+.0f}")
    c4.metric("脂質(g)", f"{total['f']:.0f}", delta=f"{(total['f'] - targets.get('f', targets.get('f_g', 0))):+.0f}")

    cA, cB = st.columns(2)
    with cA:
        if st.button("途中保存（後で続きから）", key="meal_save_draft"):
            try:
                levels = {}
                for pref in ["b","l","d"]:
                    for k in ["sel_carb","sel_protein","sel_veg","sel_fat","sel_fried","sel_dairy","sel_fruit"]:
                        kk = f"{pref}_{k}"
                        if kk in st.session_state:
                            levels[kk] = st.session_state.get(kk)
                save_meal_day_draft(code_hash, meal_date, {
                    "date": _meal_date_key(meal_date),
                    "meal_goal": goal,
                    "meal_weight": float(st.session_state.get("meal_weight") or w),
                    "targets": targets,
                    "total": total,
                    "levels": levels,
                    "b": {"est": st.session_state.get("b_est"), "ai": st.session_state.get("b_ai"), "comment": st.session_state.get("b_comment"), "school": bool(st.session_state.get("b_school") or False)},
                    "l": {"est": st.session_state.get("l_est"), "ai": st.session_state.get("l_ai"), "comment": st.session_state.get("l_comment"), "school": bool(st.session_state.get("l_school") or False)},
                    "d": {"est": st.session_state.get("d_est"), "ai": st.session_state.get("d_ai"), "comment": st.session_state.get("d_comment"), "school": bool(st.session_state.get("d_school") or False)},
                })
                st.success("途中保存しました。この日付を開けば続きから再開できます。")
                st.rerun()
            except Exception as e:
                st.error(f"途中保存に失敗: {e}")
    with cB:
        if st.button("今日の食事ログを保存", key="meal_save_simple"):
            try:
                save_record(code_hash, "meal_log", {"date": _meal_date_key(meal_date), "b": b, "l": l, "d": d, "total": total, "targets": targets}, {"summary": "meal_log"})
                # 今日のログ（AI推定・コメント・合計）をスナップショットに保存（ログアウトしても復元可）
                save_meal_day_snapshot(code_hash, meal_date, {
                    "date": _meal_date_key(meal_date),
                    "meal_goal": goal,
                    "meal_weight": float(st.session_state.get("meal_weight") or w),
                    "targets": targets,
                    "total": total,
                    "b": {
                        "est": st.session_state.get("b_est"),
                        "ai": st.session_state.get("b_ai"),
                        "comment": st.session_state.get("b_comment"),
                        "school": bool(st.session_state.get("b_school") or False),
                    },
                    "l": {
                        "est": st.session_state.get("l_est"),
                        "ai": st.session_state.get("l_ai"),
                        "comment": st.session_state.get("l_comment"),
                        "school": bool(st.session_state.get("l_school") or False),
                    },
                    "d": {
                        "est": st.session_state.get("d_est"),
                        "ai": st.session_state.get("d_ai"),
                        "comment": st.session_state.get("d_comment"),
                        "school": bool(st.session_state.get("d_school") or False),
                    },
                })
    
                # 旧来の簡易復元（フォーム用のフラットキー）も保存
                save_snapshot(code_hash, "meal_draft", {
                    "meal_goal": goal,
                    "meal_weight": float(st.session_state.get("meal_weight") or w),
                    "meal_intensity": st.session_state.get("meal_intensity"),
                    "b_c": st.session_state.get("b_c"),
                    "b_p": st.session_state.get("b_p"),
                    "b_v": st.session_state.get("b_v"),
                    "l_c": st.session_state.get("l_c"),
                    "l_p": st.session_state.get("l_p"),
                    "l_v": st.session_state.get("l_v"),
                    "d_c": st.session_state.get("d_c"),
                    "d_p": st.session_state.get("d_p"),
                    "d_v": st.session_state.get("d_v"),
                })
    
                update_streak_on_save(code_hash)
                try:
                    delete_snapshot(code_hash, meal_draft_kind(meal_date))
                except Exception:
                    pass
                st.success("保存しました。")
            except Exception as e:
                st.error(f"保存に失敗: {e}")


def exercise_prescription_page(code_hash: str):
    st.subheader("🏋️ 運動処方")
    render_streak_medal(code_hash)
    sport = st.session_state.get("sport", SPORTS[0])
    # ---- Training log (per-user la + history) ----
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
                # 主目的（プリセット＋自由入力）
        goal_opts = ["スプリント", "当たり負け改善", "持久力", "低酸素トレーニング", "リカバリー", "技術練習", "その他（自由入力）"]
        cur_goal = (st.session_state.get("tr_goal_text") or "").strip()
        default_idx = 0
        if cur_goal in goal_opts:
            default_idx = goal_opts.index(cur_goal)
        elif cur_goal:
            default_idx = goal_opts.index("その他（自由入力）")
        goal_sel = st.selectbox("主目的", goal_opts, index=default_idx, key="tr_goal_sel")
        if goal_sel == "その他（自由入力）":
            st.text_input("主目的（自由入力）", value=cur_goal, key="tr_goal_text")
        else:
            st.session_state["tr_goal_text"] = goal_sel
        st.text_area("内容メモ（セット数・距離・本数など）", value=st.session_state.get("tr_notes",""), height=120, key="tr_notes")

        cA, cB, cD, cC = st.columns([1,1,1,2])
        with cA:
            if st.button("保存", key="tr_log_save"):
                try:
                    save_training_la(code_hash)
                    st.success("保存しました。")
                except Exception as e:
                    st.error(f"保存に失敗: {e}")
        with cB:
            if st.button("最新を読み込み", key="tr_log_load"):
                try:
                    ok = load_training_la(code_hash)
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
                    delete_snapshot(code_hash, "training_la")
                    delete_la_record(code_hash, "training_log")
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
    st.markdown("### 筋トレメニュー提案")
    st.caption("体重や筋力の情報から、上半身・下半身・体幹をバランスよく提案します。")

    w = st.number_input("体重（kg）", min_value=20.0, max_value=150.0,
                        value=float(st.session_state.get("tr_weight") or st.session_state.get("profile_weight_kg") or 45.0),
                        step=0.1, key="tr_weight",
                        on_change=lambda: _weight_on_change(code_hash, "tr_weight", write_back_profile=True))
    _set_global_weight(code_hash, w, write_back_profile=True)

    bench1rm = st.number_input("ベンチプレス最大（推定1回の重さ kg・任意）", min_value=0.0, max_value=300.0,
                               value=float(st.session_state.get("tr_bench1rm", 0.0) or 0.0),
                               step=0.5, key="tr_bench1rm")

    squat_est = round(w * 1.2, 1)
    st.caption(f"スクワット（重りを使う場合の目安）: 体重×1.2 ≈ {squat_est} kg（フォーム優先）")

    equipment = st.selectbox("使える器具", ["自重中心（道具なし）", "ダンベル/チューブあり", "バーベル（ベンチ・スクワット可能）"],
                             index=0, key="tr_equipment")
    days = st.selectbox("週あたりの筋トレ日数", [1,2,3,4], index=2, key="tr_days")
    focus = st.selectbox("筋トレの目的", ["バルクアップ", "スピード・跳躍", "怪我予防", "疲労回復を優先"], index=0, key="tr_menu_focus")

    st.text_area("追加コメント（例：もう少しきつく／重量を重く／休憩を短く）", key="tr_menu_adjust", height=80)

    if st.button("AIでメニューを作る", type="primary", key="tr_ai"):
        if not require_premium_ai(code_hash):
            return
        system = "You are a strength & conditioning coach specializing in youth athletes. Output concise Japanese."
        user = f"""競技: {sport}
    体重: {w} kg
    ベンチプレス最大(推定1RM): {bench1rm if bench1rm>0 else '不明'} kg
    スクワット目安: {squat_est} kg（体重から推定）
    器具: {equipment}
    週の筋トレ日数: {days}
    目的: {focus}
    追加コメント: {st.session_state.get("tr_menu_adjust","")}

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
            html_menu = normalize_training_headings(text)
            plain_menu = strip_html_simple(html_menu)
            st.session_state["tr_menu_text"] = plain_menu           # 保存用（HTMLなし）
            st.session_state["tr_menu_text_html"] = html_menu       # 表示用（見出し装飾あり）
            ai_highlight_box("🏋️ 筋トレメニュー（生成結果）", html_menu)

            # きつくしたい場合の再生成（40代でも迷わない）
            if st.button("このメニューをもう少しきつくして再生成", key="tr_menu_make_harder"):
                st.session_state["tr_menu_adjust"] = (st.session_state.get("tr_menu_adjust","").strip() + "\n" +
                    "全体的に少しきつくしてください。可能なら重量を上げ（目安：1RMの70〜85%）、回数やセットを微増し、休憩を短くしてください。フォームが崩れるなら無理しない注意も入れてください。").strip()
                st.rerun()


    if st.button("トレーニングログを保存", key="tr_inputs_save"):
        save_record(code_hash, "training_inputs",
                    {"sport": sport, "weight": w, "bench1rm": bench1rm, "squat_est": squat_est,
                     "equipment": equipment, "days": days, "focus": focus},
                    {"summary": "training_inputs"})
        st.success("保存しました。")

        # -----------------
        # 怪我
        # -----------------
    jams_logo_footer()
    # --- 保存済みAIコメント（コピーはここから） ---
    saved_ai_footer([
        {"key": "tr_menu_text", "title": "🏋️ 運動処方：筋トレメニュー"},
    ])




def coldflu_page(code_hash: str):
    st.subheader("🤒 風邪・インフルエンザ相談（診断ではありません）")

    if not is_premium(code_hash):
        premium_gate(code_hash, "このページはプレミアムで利用できます。")
        return

    sport = st.session_state.get("sport", SPORTS[0])

    st.caption("症状を整理して、クリニックへ相談するための文章を作ります。診断は行いません。")

    st.markdown("### 症状の入力")
    st.session_state.setdefault("cf_onset", now_jst().date())
    onset = st.date_input("いつから？", value=st.session_state.get("cf_onset"), key="cf_onset")
    temp_max = st.number_input("最高体温（℃）", min_value=34.0, max_value=43.0, value=float(st.session_state.get("cf_temp_max") or 37.5), step=0.1, key="cf_temp_max")
    fever_days = st.selectbox("発熱の経過", ["上がってきた", "下がってきた", "横ばい", "発熱なし"], index=0, key="cf_fever_trend")

    st.markdown("### 症状チェック")
    cols = st.columns(2)
    with cols[0]:
        sore = st.checkbox("のど痛", key="cf_sore")
        cough = st.checkbox("咳", key="cf_cough")
        runny = st.checkbox("鼻水/鼻づまり", key="cf_runny")
        headache = st.checkbox("頭痛", key="cf_headache")
        chill = st.checkbox("悪寒", key="cf_chill")
    with cols[1]:
        fatigue = st.checkbox("強いだるさ", key="cf_fatigue")
        muscle = st.checkbox("筋肉痛/関節痛", key="cf_muscle")
        nausea = st.checkbox("吐き気/嘔吐", key="cf_nausea")
        diarrhea = st.checkbox("下痢", key="cf_diarrhea")
        appetite = st.checkbox("食欲低下", key="cf_appetite")

    st.markdown("### 状況")
    school_outbreak = st.checkbox("学校・チームで流行している", key="cf_outbreak")
    family = st.checkbox("家族に発熱者がいる", key="cf_family")
    breathing = st.checkbox("息が苦しい/呼吸がつらい", key="cf_breathing")
    hydration = st.checkbox("水分が取れない", key="cf_hydration")
    note = st.text_area("メモ（自由記載）", key="cf_note", height=90)

    if st.button("AIで相談文を作る", type="primary", key="cf_ai"):
        system = "You are a sports medicine clinician. Output Japanese. Do NOT diagnose. Avoid definitive statements. Be concise and structured."
        user = f"""競技: {sport}
発症: {onset}
最高体温: {temp_max}℃
発熱の経過: {fever_days}
症状: のど痛={sore}, 咳={cough}, 鼻={runny}, 頭痛={headache}, 悪寒={chill}, だるさ={fatigue}, 筋肉痛/関節痛={muscle}, 吐き気/嘔吐={nausea}, 下痢={diarrhea}, 食欲低下={appetite}
状況: 流行={school_outbreak}, 家族発熱={family}
危険そうなサイン: 呼吸苦={breathing}, 水分不可={hydration}
メモ: {note}

要件:
- 診断はしない（「〜の可能性」まで）
- 形式は以下
  1) まとめ（発症/発熱/主症状/周囲状況）
  2) 考えやすい状態（診断ではない）
  3) 自宅でできる対応
  4) 相談を急いだ方がよいサイン（箇条書き）
- “受診の目安”という言葉は使わない
"""
        text, err = ai_text(system, user)
        if err:
            st.error("AIに失敗: " + err)
        else:
            st.session_state["cf_ai_text"] = text
            ai_highlight_box("🤒 相談文（保存されます）", text)

            st.markdown("### 📲 公式LINEに貼る（プレミアム）")
            st.text_area("LINE貼り付け用テキスト", text, height=220, key="cf_line_text")
            clipboard_copy_button("LINEに貼る文章をコピー", text, key="cf_copy_line_btn")
            if "LINE_OFFICIAL_URL" in globals() and LINE_OFFICIAL_URL:
                st.link_button("公式LINEを開く", LINE_OFFICIAL_URL)

    # 保存済み（コピーはここから）
    saved_ai_footer([
        {"key": "cf_ai_text", "title": "🤒 風邪/インフル：相談文"},
    ])


def injury_page(code_hash: str):
    st.subheader("🩹 怪我")
    sport = st.session_state.get("sport", SPORTS[0])

    st.markdown("### 痛む場所を選んでください（頭 → 足先）")
    st.caption("まずは主な痛みを1つ選びます。必要なら2つ目も追加できます。最後にAIが整形します。")

    # 競技で多い部位（表示の補助）
    sport_hint = {
        "サッカー": "膝/足首/ハムストリング/股関節（鼠径部）/踵（足底）",
        "バスケットボール": "足首/膝/踵（足底）",
        "野球": "肩/肘/手首/腰",
        "陸上": "ハムストリング/ふくらはぎ/足首",
    }
    for k, v in sport_hint.items():
        if k in str(sport):
            st.info(f"この競技で多い部位の例：{v}")
            break

    LOCS = [
        "頭（顔）", "首",
        "肩", "肘", "手首", "手指",
        "胸/肋骨",
        "背中", "腰",
        "股関節/鼠径部",
        "太もも前", "太もも後（ハムストリング）",
        "膝",
        "すね", "ふくらはぎ",
        "足首",
        "踵/足底",
        "足（足背/足趾）",
    ]

    primary = st.selectbox("主な痛む場所", LOCS, index=0, key="inj_primary_loc")

    add_second = st.checkbox("2つ目の場所もある", key="inj_add_second")
    secondary = None
    if add_second:
        secondary = st.selectbox("2つ目の痛む場所", [x for x in LOCS if x != primary], index=0, key="inj_secondary_loc")

    locs = [primary] + ([secondary] if secondary else [])

    # ----------------------------
    # 共通質問
    # ----------------------------
    st.markdown("### 共通の質問")
    pain = st.slider("痛み（0-10）", 0, 10, 0, key="inj_pain")
    st.caption("例：0=痛みなし / 2-3=違和感 / 4-5=動かすと痛い / 6-7=練習が難しい / 8-10=日常生活もつらい")

    onset = st.selectbox("きっかけ", ["急に（ひねった・ぶつけた・着地で痛い）", "少しずつ（使いすぎ・疲れ）"], index=0, key="inj_onset")
    swelling = st.checkbox("腫れがある", key="inj_swelling")
    bruise = st.checkbox("内出血がある", key="inj_bruise")
    numb = st.checkbox("しびれ・感覚の違和感がある", key="inj_numb")
    fever = st.checkbox("熱がある", key="inj_fever")

    # 下肢が含まれる場合だけ荷重を聞く
    lower_limb = any(x in locs for x in ["股関節/鼠径部", "太もも前", "太もも後（ハムストリング）", "膝", "すね", "ふくらはぎ", "足首", "踵/足底", "足（足背/足趾）"])
    weight_bearing = st.selectbox(
        "体重をかけられる？（足の痛みがある場合）",
        ["問題なし", "少し痛いが可能", "ほぼ無理"],
        index=0,
        key="inj_bearing"
    ) if lower_limb else "（対象外）"

    extra = {}

    # ----------------------------
    # 部位別（よくあるスポーツ外傷を中心に）
    # ----------------------------
    st.markdown("### 追加の質問（選んだ場所に応じて）")
    for loc in locs:
        with st.expander(f"{loc} の追加質問", expanded=False):

            # どの部位でも「一番つらい動き」は聞く
            extra[f"{loc}_worse"] = st.selectbox(
                "一番つらい動き",
                ["特になし", "走る", "ジャンプ", "切り返し", "蹴る", "投げる", "日常動作"],
                index=0,
                key=f"inj_{loc}_worse"
            )

            # 肩・肘・手首・手指（投球/打撃）
            if loc in ["肩", "肘", "手首", "手指"]:
                extra[f"{loc}_throw"] = st.checkbox("投げる/打つ動作で強く痛む", key=f"inj_{loc}_throw")
                extra[f"{loc}_weak"] = st.checkbox("力が入りにくい", key=f"inj_{loc}_weak")
                extra[f"{loc}_night"] = st.checkbox("夜間痛がある/じっとしていても痛む", key=f"inj_{loc}_night")

            # 背中・腰
            if loc in ["背中", "腰"]:
                extra[f"{loc}_legpain"] = st.checkbox("脚の方に痛み/しびれが走る", key=f"inj_{loc}_rad")
                extra[f"{loc}_extend"] = st.checkbox("反ると痛い", key=f"inj_{loc}_extend")
                extra[f"{loc}_flex"] = st.checkbox("前屈で痛い", key=f"inj_{loc}_flex")

            # 股関節/鼠径部
            if loc in ["股関節/鼠径部"]:
                extra[f"{loc}_kick"] = st.checkbox("蹴る/切り返しで痛い", key=f"inj_{loc}_kick")
                extra[f"{loc}_adduct"] = st.checkbox("内もも（内転筋）を押すと痛い", key=f"inj_{loc}_adduct")
                extra[f"{loc}_limp"] = st.checkbox("走ると跛行（びっこ）になる", key=f"inj_{loc}_limp")

            # 太もも前/ハム
            if loc in ["太もも前", "太もも後（ハムストリング）"]:
                extra[f"{loc}_sudden_pop"] = st.checkbox("走った/蹴った瞬間に『ブチッ/ピキッ』とした感じがあった", key=f"inj_{loc}_pop")
                extra[f"{loc}_stretch_pain"] = st.checkbox("伸ばすと痛い", key=f"inj_{loc}_stretch")
                extra[f"{loc}_contract_pain"] = st.checkbox("力を入れると痛い", key=f"inj_{loc}_contract")
                extra[f"{loc}_walking_pain"] = st.checkbox("歩くだけでも痛い", key=f"inj_{loc}_walk")

            # 膝
            if loc in ["膝"]:
                extra[f"{loc}_giving_way"] = st.checkbox("踏ん張るとガクっとする/抜ける感じがある", key=f"inj_{loc}_give")
                extra[f"{loc}_locking"] = st.checkbox("引っかかる/動かしにくい感じがある", key=f"inj_{loc}_lock")
                extra[f"{loc}_stairs"] = st.checkbox("階段で痛い", key=f"inj_{loc}_stairs")
                extra[f"{loc}_swollen"] = st.checkbox("膝が水がたまった感じに腫れる", key=f"inj_{loc}_eff")

            # すね
            if loc in ["すね"]:
                extra[f"{loc}_diffuse"] = st.checkbox("広い範囲がズーンと痛い（走ると増える）", key=f"inj_{loc}_diff")
                extra[f"{loc}_point"] = st.checkbox("一点を押すと強く痛い", key=f"inj_{loc}_point")

            # ふくらはぎ
            if loc in ["ふくらはぎ"]:
                extra[f"{loc}_tightness"] = st.checkbox("つっぱる/攣りそうな感じが強い", key=f"inj_{loc}_tight")
                extra[f"{loc}_push_off_pain"] = st.checkbox("つま先立ち（蹴り出し）で痛い", key=f"inj_{loc}_push")
                extra[f"{loc}_localized"] = st.selectbox("痛い場所の中心", ["中央", "内側", "外側", "アキレス腱寄り"], index=0, key=f"inj_{loc}_spot")

            # 足首
            if loc in ["足首"]:
                extra[f"{loc}_twist_in"] = st.checkbox("内側にひねった（内返し）", key=f"inj_{loc}_inv")
                extra[f"{loc}_twist_out"] = st.checkbox("外側にひねった（外返し）", key=f"inj_{loc}_ev")
                extra[f"{loc}_bearing"] = st.selectbox("今の荷重", ["問題なし", "少し痛いが可能", "ほぼ無理"], index=0, key=f"inj_{loc}_bearing2")

            # 踵/足底
            if loc in ["踵/足底"]:
                extra[f"{loc}_morning"] = st.checkbox("朝一歩目が特に痛い", key=f"inj_{loc}_am")
                extra[f"{loc}_spike"] = st.checkbox("スパイク/靴で悪化する", key=f"inj_{loc}_shoe")

            # 足（足背/足趾）
            if loc in ["足（足背/足趾）"]:
                extra[f"{loc}_toe"] = st.checkbox("足趾を動かすと痛い", key=f"inj_{loc}_toe")
                extra[f"{loc}_swelling"] = st.checkbox("足の甲が腫れている", key=f"inj_{loc}_sw")

            # 頭/首（赤旗）
            if loc in ["頭（顔）", "首"]:
                extra[f"{loc}_headache"] = st.checkbox("頭痛がある", key=f"inj_{loc}_hd")
                extra[f"{loc}_nausea"] = st.checkbox("吐き気/嘔吐がある", key=f"inj_{loc}_nv")
                extra[f"{loc}_dizzy"] = st.checkbox("めまい/ふらつきがある", key=f"inj_{loc}_dz")

    st.markdown("### 直ぐにできる対応")
    st.write("• **痛みの出る動きは行わない**（痛みが出ない範囲での活動に切り替える）")
    st.write("• **冷やす**：氷や保冷剤をタオルで包んで、10〜15分を1日に数回")
    st.write("• **押さえる**：腫れているなら、包帯やサポーターで軽く固定（きつすぎない）")
    st.write("• **高くする**：足のケガなら、座って足をクッションで少し高くする")
    st.write("• 痛みが強い/腫れが増える/しびれ/体重をかけられない/熱がある時は、早めに相談が安心です。")

    if st.button("AIコメントを出す", type="primary", key="inj_ai"):
        if not require_premium_ai(code_hash):
            return
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
            st.session_state["inj_ai_text"] = text
            ai_highlight_box("🩹 怪我AIコメント（保存されます）", text)

            if is_premium(code_hash):
                st.markdown("### 📲 公式LINEに貼る（プレミアム）")
                st.caption("下の文章をコピーして、公式LINEのトークに貼り付けてください。")
                st.text_area("LINE貼り付け用テキスト", text, height=220, key="inj_line_text")
                clipboard_copy_button("LINEに貼る文章をコピー", text, key="inj_copy_line_btn")
                if "LINE_OFFICIAL_URL" in globals() and LINE_OFFICIAL_URL:
                    st.link_button("公式LINEを開く", LINE_OFFICIAL_URL)

            else:
                st.info("公式LINE連動（コピー＆起動）はプレミアムで利用できます。")

            st.caption("※コピーやTXT保存は、ページ最下部の『保存したAIコメント』から行えます。")

    if st.button("怪我ログを保存", key="inj_save"):
        save_record(code_hash, "injury_log",
                    {"sport": sport, "locations": locs, "pain": pain, "onset": onset,
                     "swelling": swelling, "bruise": bruise, "numb": numb, "fever": fever,
                     "bearing": weight_bearing, "extra": extra},
                    {"summary": "injury_log"})
        st.success("保存しました。")

    jams_logo_footer()
    saved_ai_footer([
        {"key": "inj_ai_text", "title": "🩹 怪我：AIコメント"},
    ])


def sleep_page(code_hash: str):
    st.subheader("😴 睡眠")

    sport = st.session_state.get("sport", SPORTS[0])

    st.markdown("### 昨日の睡眠")

    # --- 入力 ---
    sleep_h = st.number_input(
        "睡眠時間（時間）",
        0.0, 16.0, 8.0, 0.25,
        help="成長期は8〜10時間が目安です"
    )

    wake_quality = st.selectbox(
        "今朝の目覚めはどうだった？",
        ["😴 まだ眠い", "😐 まあまあ", "🙂 すっきり", "😄 とても良い"],
        help="起きたときの回復感を直感で選んでください"
    )

    screen = st.number_input(
        "就寝前のスマホ・ゲーム時間（分）",
        0, 300, 60, 5
    )

    # --- スコア計算 ---
    WAKE_SCORE = {
        "😴 まだ眠い": 5,
        "😐 まあまあ": 10,
        "🙂 すっきり": 15,
        "😄 とても良い": 20,
    }

    score = 0

    # 睡眠時間（最大40点）
    if sleep_h >= 9:
        score += 40
    elif sleep_h >= 8:
        score += 35
    elif sleep_h >= 7:
        score += 25
    else:
        score += 15

    # 目覚め（最大20点）
    score += WAKE_SCORE[wake_quality]

    # スクリーン時間（最大40点）
    if screen <= 30:
        score += 40
    elif screen <= 60:
        score += 30
    elif screen <= 90:
        score += 20
    else:
        score += 10

    score = int(max(0, min(100, score)))

    st.metric("睡眠スコア", f"{score} / 100")

    # --- AIアドバイス ---
    if st.button("AIで睡眠アドバイスを作る", key="sl_ai_make"):
        if not require_premium_ai(code_hash):
            return
        system = (
            "You are a sports medicine clinician and youth athlete performance coach. "
            "Give practical, safe, and kind sleep advice in Japanese. "
            "Use short bullets."
        )
        user = f"""
競技: {sport}
睡眠時間: {sleep_h}時間
起床時の目覚め: {wake_quality}
就寝前スクリーン時間: {screen}分
睡眠スコア: {score}/100

要件:
- 1) 評価（良い点）
- 2) 気になる点
- 3) 今日からできる改善を2〜3個
- 4) 明日の練習・試合への一言
文章はやさしく、子どもにも分かる表現で。
"""

        text, err = ai_text(system, user)
        if err:
            st.error("AIに失敗しました")
        else:
            st.session_state["sl_ai_text"] = text

    if st.session_state.get("sl_ai_text"):
        ai_highlight_box("😴 睡眠AIアドバイス（保存されます）", st.session_state["sl_ai_text"])
        st.caption("※コピーやTXT保存はページ最下部の『保存したAIコメント』から行えます。")

    # --- 保存 ---
    if st.button("睡眠ログを保存", key="sl_save"):
        save_record(
            code_hash,
            "sleep_log",
            {
                "sleep_h": float(sleep_h),
                "wake_quality": wake_quality,
                "screen": int(screen),
                "score": score,
            },
            {"summary": "sleep_log"}
        )
        update_streak_on_save(code_hash)
        st.success("保存しました。")




    # -----------------
    # サッカー動画（YouTube検索）
    # -----------------
    jams_logo_footer()
    # --- 保存済みAIコメント（コピーはここから） ---
    saved_ai_footer([
        {"key": "sl_ai_text", "title": "😴 睡眠：AIアドバイス"},
    ])


def soccer_video_page(code_hash: str):
    st.subheader("🎥 サッカー動画")
    sport = st.session_state.get("sport", SPORTS[0])
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
    jams_logo_footer()


# =========================
# Mobile navigation (Profile -> Menu -> Pages)
# =========================

APP_PAGES = [
    ("exercise", "🏋️ 運動処方"),
    ("meal", "🍽 食事管理"),
    ("height", "📏 身長予測"),
    ("anemia", "🩸 スポーツ貧血"),
    ("injury", "🩹 怪我の相談"),
    ("coldflu", "🤒 風邪・インフル相談（プレミアム）"),
    ("sleep", "😴 睡眠の質"),
    ("soccer", "🎥 サッカー動画検索"),
    ("profile", "👤 個人情報"),
]

LINE_OFFICIAL_URL = (os.getenv("KIWI_LINE_OFFICIAL_URL", "").strip() or "https://line.me/R/ti/p/@983prujv")  # 公式LINE
LINE_PREFILL_TEXT = os.getenv("KIWI_LINE_PREFILL_TEXT", "怪我の相談（アプリ）: ").strip()

def _route_get():
    return st.session_state.get("route", "")

def _route_set(r: str):
    st.session_state["route"] = r

def _nav_to_menu():
    _route_set("menu")
    st.rerun()

def _nav_button_to_menu(position: str = "top"):
    # position is only for key uniqueness
    st.markdown('<div class="km-navbtn">', unsafe_allow_html=True)
    if st.button("⬅️ 機能選択へ戻る", key=f"to_menu_{position}", use_container_width=True):
        _nav_to_menu()
    st.markdown("</div>", unsafe_allow_html=True)


# =====================
# Plan (basic / premium)
# =====================
def get_plan(code_hash: str) -> str:
    d = load_snapshot(code_hash, "plan") or {}
    tier = (d.get("tier") or "basic").strip().lower()
    return "premium" if tier == "premium" else "basic"

def set_plan(code_hash: str, tier: str):
    tier = (tier or "basic").strip().lower()
    if tier not in ("basic", "premium"):
        tier = "basic"
    save_snapshot(code_hash, "plan", {"tier": tier, "updated_at": iso(now_jst())})

def is_premium(code_hash: str) -> bool:
    return get_plan(code_hash) == "premium"

def premium_gate(code_hash: str, label: str = "この機能はプレミアムで利用できます"):
    st.info(label)
    st.caption("プレミアムにすると、AIアドバイス・風邪/インフル相談・公式LINE連動が使えます。")

def require_premium_ai(code_hash: str) -> bool:
    """Return True if premium, else show notice and return False."""
    if is_premium(code_hash):
        return True
    premium_gate(code_hash, "AIアドバイス機能はプレミアムで利用できます。")
    return False


def _load_profile(code_hash: str) -> dict:
    d = load_snapshot(code_hash, "profile") or {}
    if isinstance(d, dict):
        return d
    return {}

def _save_profile(code_hash: str, payload: dict):
    save_snapshot(code_hash, "profile", payload)


def _sync_profile_to_session(code_hash: str, prof: dict | None = None):
    """Load profile snapshot (if needed) and sync key fields into st.session_state
    so other pages can use them as defaults (dob/age/sex/weight/height).
    """
    if prof is None:
        prof = _load_profile(code_hash) or {}
    # birth -> dob + age_years
    b = (prof.get("birth") or "").strip()
    dob = None
    try:
        if b:
            dob = date.fromisoformat(b)
    except Exception:
        dob = None

    if dob:
        st.session_state["dob"] = dob
        # age in years (JST date basis)
        today = datetime.now(timezone(timedelta(hours=9))).date()
        age_days = (today - dob).days
        st.session_state["age_years"] = max(0.0, age_days / 365.25)
    # sex -> sex_code
    sex = (prof.get("sex") or "").strip()
    if sex == "男":
        st.session_state["sex_code"] = "M"
    elif sex == "女":
        st.session_state["sex_code"] = "F"

    # defaults for weight/height used by multiple pages
    try:
        w = float(prof.get("weight_kg") or 0.0)
    except Exception:
        w = 0.0
    if w > 0:
        # Always treat profile as the source of truth
        st.session_state["profile_weight_kg"] = float(w)
        st.session_state["la_weight_kg"] = float(w)

        # Clear "manual" flags so other tabs can re-seed from updated profile on the next rerun
        for _k in WEIGHT_KEYS:
            if _k != "pf_weight":
                st.session_state.pop(f"{_k}__manual", None)

    try:
        h = float(prof.get("height_cm") or 0.0)
    except Exception:
        h = 0.0
        # seed tab weights (only if not manually edited)
        for k in WEIGHT_KEYS:
            if k not in st.session_state or float(st.session_state.get(k) or 0.0) <= 0.0:
                st.session_state[k] = float(st.session_state["profile_weight_kg"])

    try:
        h = float(prof.get("height_cm") or 0.0)
    except Exception:
        h = 0.0
    if ("la_height_cm" not in st.session_state) or float(st.session_state.get("la_height_cm") or 0.0) <= 0.0:
        if h > 0:
            st.session_state["la_height_cm"] = h


def profile_top_page(code_hash: str):
    render_brand_header()
    st.markdown('<div class="km-wrap">', unsafe_allow_html=True)
    st.markdown("## 基礎情報（最初に1回）")
    prof = _load_profile(code_hash)

    with st.container():
        st.markdown('<div class="km-card">', unsafe_allow_html=True)
        # 最小限：スマホで入力しやすい項目だけ
        name = st.text_input("名前（ニックネーム可）", value=prof.get("name",""), key="pf_name")
        sex = st.selectbox("性別", ["未選択","男","女"], index=["未選択","男","女"].index(prof.get("sex","未選択") if prof.get("sex","未選択") in ["未選択","男","女"] else "未選択"), key="pf_sex")
        import datetime as _dt
        _b = (prof.get("birth","") or "").strip()
        try:
            _b_date = _dt.date.fromisoformat(_b) if _b else _dt.date(2010,1,1)
        except Exception:
            _b_date = _dt.date(2010,1,1)
        birth = st.date_input("生年月日", value=_b_date, min_value=_dt.date(1900,1,1), max_value=_dt.date.today(), key="pf_birth")
        _h0 = float(prof.get("height_cm") or 0.0)
        _w0 = float(prof.get("weight_kg") or 0.0)
        if _h0 < 50.0:
            _h0 = 150.0
        if _w0 < 10.0:
            _w0 = 40.0

        height_cm = st.number_input("身長（cm）", min_value=50.0, max_value=230.0, value=_h0, step=0.1, key="pf_height")
        weight_kg = st.number_input("体重（kg）", min_value=10.0, max_value=200.0, value=_w0, step=0.1, key="pf_weight")

        # プラン（販売版ではStripe連動に置き換え）
        tier = get_plan(code_hash)
        tier_label = "プレミアム" if tier=="premium" else "ベーシック"
        sel = st.radio("プラン", ["ベーシック", "プレミアム"], index=1 if tier=="premium" else 0, horizontal=True, key="pf_plan")
        if sel == "プレミアム":
            set_plan(code_hash, "premium")
        else:
            set_plan(code_hash, "basic")

        st.markdown('<div class="km-muted">※入力後は自動保存され、リセットしない限りこの情報で進みます。</div>', unsafe_allow_html=True)

        # 自動保存（毎回）
        payload = {
            "name": (name or "").strip(),
            "sex": sex,
            "birth": str(birth) if birth else "",
            "height_cm": float(height_cm or 0.0),
            "weight_kg": float(weight_kg or 0.0),
        }
        _save_profile(code_hash, payload)
        _sync_profile_to_session(code_hash, payload)

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("次へ（機能を選ぶ）", type="primary", use_container_width=True, key="pf_next"):
                _route_set("menu")
                st.rerun()
        with colB:
            if st.button("基礎情報をリセット", use_container_width=True, key="pf_reset"):
                delete_snapshot(code_hash, "profile")
                for k in list(st.session_state.keys()):
                    if k.startswith("pf_"):
                        del st.session_state[k]
                st.success("基礎情報をリセットしました。")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

def menu_select_page(code_hash: str):
    render_brand_header()
    # 40代の親が迷わず押せる：大きい2列ボタン（スマホ最適）
    st.markdown('<div class="km-menu-title">やりたいことを選んでください</div>', unsafe_allow_html=True)
    st.markdown('<div class="km-menu-sub">迷ったら、いちばん気になる項目を1つ選べばOKです。</div>', unsafe_allow_html=True)

    # 2列レイアウト（スマホで縦積みになってもボタンは大きいまま）
    pairs = [p for p in list(APP_PAGES) if (p[0] != "coldflu" or is_premium(code_hash))]

    for i in range(0, len(pairs), 2):
        left = pairs[i]
        right = pairs[i + 1] if i + 1 < len(pairs) else None

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.markdown('<div class="km-bigbtn">', unsafe_allow_html=True)
            if st.button(left[1], key=f"menu_{left[0]}", use_container_width=True):
                _route_set(left[0])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if right:
                st.markdown('<div class="km-bigbtn">', unsafe_allow_html=True)
                if st.button(right[1], key=f"menu_{right[0]}", use_container_width=True):
                    _route_set(right[0])
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("")

    st.markdown('<div class="km-footer-note">※ グラフや詳細は、必要なときだけ開けばOKです。</div>', unsafe_allow_html=True)

def injury_line__box():
    st.markdown("### 🧪医師へ相談を送る")
    if not LINE_OFFICIAL_URL:
        st.info("公式LINE送信はです。環境変数 KIWI_LINE_OFFICIAL_URL を設定すると有効になります。")
        return
    st.markdown("AIの結果を踏まえて医師に相談したい場合、公式LINEを開いて送信できます。")
    ok = st.checkbox("公式LINEへ送信します（確認）", key="inj_line_confirm")
    if ok:
        # 送信はブラウザで公式LINEを開く（実際の送信はユーザー操作）
        st.link_button("公式LINEを開く", LINE_OFFICIAL_URL)

def main():
    st.set_page_config(page_title="ジュニアス", page_icon="assets/logo.png", layout="wide")
    premium_css()
    apply_css()
    init_users_db()
    init_data_db()

    user = st.session_state.get("user")
    if not user:
        user = login_panel()
        if not user:
            return

    code_hash = sha256_hex(user)

    # 最新データの自動復元（入力補助）
    try:
        auto_fill_from_la_records(code_hash)
    except Exception:
        pass

    # 基礎情報が保存済みなら、dob/体重などをセッションへ同期（他ページの初期値に使う）
    try:
        prof = _load_profile(code_hash)
        if prof:
            _sync_profile_to_session(code_hash, prof)
    except Exception:
        pass

    # 体重は個人情報を“唯一の基礎値”として全タブへ同期（ウィジェット生成前）
    _sync_weight_defaults_before_render(code_hash)

    # ルーティング初期化：基礎情報が未登録ならトップへ
    if "route" not in st.session_state or not st.session_state.get("route"):
        prof = load_snapshot(code_hash, "profile")
        st.session_state["route"] = "menu" if prof else "profile"

    r = _route_get()

    if r == "profile":
        profile_top_page(code_hash)
        return

    if r == "profile_edit":
        profile_top_page(code_hash)
        return

    if r == "menu":
        menu_select_page(code_hash)
        return

    # 3ページ目以降：必ずトップ/ボトムに「機能選択へ戻る」
    _nav_button_to_menu("top")

    if r == "exercise":
        exercise_prescription_page(code_hash)
    elif r == "meal":
        meal_page(code_hash)
    elif r == "height":
        height_page(code_hash)
    elif r == "anemia":
        anemia_page(code_hash)
    elif r == "injury":
        injury_page(code_hash)
    elif r == "coldflu":
        coldflu_page(code_hash)
        injury_line__box()
    elif r == "sleep":
        sleep_page(code_hash)
    elif r == "soccer":
        soccer_video_page(code_hash)
    else:
        menu_select_page(code_hash)
        return

    st.markdown('<div class="km-bottom">', unsafe_allow_html=True)
    _nav_button_to_menu("bottom")
    st.markdown('</div>', unsafe_allow_html=True)

    # AIコメントをDBに保存（翌日・別端末でも復元できる）
    try:
        persist_ai_cache_from_session(code_hash)
    except Exception:
        pass

if __name__ == "__main__":
    main()