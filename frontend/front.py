import sys, os, importlib
import base64
import hashlib

# Ensure project root is on sys.path so `backend.*` imports work both locally and on Streamlit Cloud.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
from streamlit.errors import StreamlitAuthError
import streamlit.components.v1 as components
import PyPDF2
from io import BytesIO
import json
from datetime import datetime
import time
import pandas as pd
import plotly.express as px

# Reload backend modules so code fixes apply without full process restart.
# Only class definitions are reloaded; heavy model weights load in __init__() only.
for _mn in ['backend.hybrid_search', 'backend.bulk_ingest', 'backend.generator']:
    if _mn in sys.modules:
        importlib.reload(sys.modules[_mn])

from backend.hybrid_search import HybridSearch
from backend.bulk_ingest import BulkIngest
from backend.generator import Generator

# Bump this when backend runtime objects need re-init after fixes.
BACKEND_RUNTIME_VERSION = "2026-02-15-runtime-fix-6"

# ---------- Auth state ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "google_intent" not in st.session_state:
    st.session_state.google_intent = None  # "login" | "signup"

def _handle_logout():
    """Callback — runs BEFORE the next rerun when Logout is clicked."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.show_signup = False
    st.session_state.auth_provider = None
    st.session_state["_just_logged_out"] = True
    try:
        _lg = globals().get("load_generator")
        if _lg is not None:
            _lg.clear()
    except Exception:
        pass
    for _k in ["messages", "conversations", "current_view",
                "last_response", "chat_draft", "docs", "doc_files", "_pending_docs_for_badge",
                "_user_db_key", "groq_api_key", "generator", "model_loaded", "_history_loaded"]:
        st.session_state.pop(_k, None)

st.set_page_config(
    page_title="ElasticNode AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

USER_DB = os.path.join(os.path.dirname(__file__), "users.json")

def _load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_users(users):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def _register_user(username, email, password):
    users = _load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {"email": email, "password": _hash_password(password), "created_at": datetime.now().isoformat()}
    _save_users(users)
    return True, "Account created successfully!"

def _login_user(username, password):
    users = _load_users()
    if username not in users:
        return False, "Username not found"
    if hashlib.sha256(password.encode()).hexdigest() != users[username]["password"]:
        return False, "Incorrect password"
    users[username]["last_login"] = datetime.now().isoformat()
    _save_users(users)
    return True, "Login successful!"


def _google_user_key(email):
    return "google:" + (email or "").strip().lower()

def _user_by_google_email(email):
    users = _load_users()
    return users.get(_google_user_key(email))

def _register_google_user(email, name, sub):
    users = _load_users()
    key = _google_user_key(email)
    if key in users:
        return True
    users[key] = {
        "provider": "google",
        "email": email,
        "name": name or email,
        "sub": sub,
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    return True


def _validate_groq_api_key(api_key: str) -> tuple:
    """Validate Groq API key by calling Groq API. Returns (True, None) or (False, error_msg)."""
    key = (api_key or "").strip()
    if not key:
        return False, "Please enter an API key."
    try:
        from groq import Groq
        client = Groq(api_key=key)
        # Lightweight check: list models (does not consume quota)
        client.models.list()
        return True, None
    except Exception as e:
        err = str(e).lower()
        if "invalid" in err or "auth" in err or "401" in err or "403" in err:
            return False, "Invalid API key. Check the key at console.groq.com and try again."
        return False, f"Could not verify key: {str(e)[:80]}"


def _groq_key_fingerprint(api_key: str) -> str:
    """Short fingerprint for display (e.g. last 6 chars). Not stored; only for UI."""
    k = (api_key or "").strip()
    if len(k) <= 6:
        return "••••" if k else ""
    return "••••" + k[-6:]


def _get_current_user_db_key():
    """Key in users.json for the current user (username or google:email or groq key hash)."""
    if st.session_state.get("auth_provider") == "groq":
        return st.session_state.get("_user_db_key") or ""
    if st.session_state.get("auth_provider") == "google":
        return st.session_state.get("_user_db_key") or ""
    return (st.session_state.get("username") or "").strip() or ""


def _get_user_groq_key():
    """Groq API key saved for the current user, or empty."""
    key = _get_current_user_db_key()
    if not key:
        return ""
    users = _load_users()
    return (users.get(key) or {}).get("groq_api_key") or ""


def _save_user_groq_key(api_key: str):
    """Save Groq API key for the current user."""
    db_key = _get_current_user_db_key()
    if not db_key:
        return False
    users = _load_users()
    if db_key not in users:
        users[db_key] = {}
    users[db_key]["groq_api_key"] = (api_key or "").strip()
    _save_users(users)
    return True


def _load_user_history() -> dict:
    """Load saved conversation history for the current user (if any)."""
    db_key = _get_current_user_db_key()
    if not db_key:
        return {}
    users = _load_users()
    return (users.get(db_key) or {}).get("history", {}) or {}


def _save_user_history():
    """Persist current user's conversations to disk so history survives refresh/restart."""
    db_key = _get_current_user_db_key()
    if not db_key:
        return
    users = _load_users()
    if db_key not in users:
        users[db_key] = {}
    users[db_key]["history"] = st.session_state.get("conversations", {}) or {}
    _save_users(users)


# ---------- Login only via Groq API key: key = auth, site detects account via key ----------
if not st.session_state.authenticated:
    st.markdown("""
    <style>
    * { font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    section[data-testid="stSidebar"] { display: none !important; }
    html, body, [data-testid="stAppViewContainer"] { margin: 0; padding: 0; min-height: 100vh; background: #fafafa !important; }
    .main { background: #fafafa !important; padding: 0 !important; }
    .block-container { padding: 2rem 1.5rem !important; max-width: 420px !important; margin: 0 auto !important; }
    .auth-card { text-align: center; padding: 0.5rem 0; }
    .auth-title { font-size: 1.25rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.25rem; }
    .auth-subtitle { color: #737373; font-size: 0.875rem; margin-bottom: 0.5rem; }
    .stTextInput > div > div > input { border: 1px solid #e5e5e5 !important; border-radius: 8px !important; font-size: 0.9375rem !important; }
    .stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-card">
        <div style="font-size: 2rem; margin-bottom: 0.25rem;">🤖</div>
        <div class="auth-title">Sign in with Groq</div>
        <div class="auth-subtitle">One key. One click. Instant access to your AI-powered docs.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.get("auth_error"):
        st.error(st.session_state.auth_error)
        st.session_state.auth_error = None
    with st.form("groq_login_form"):
        api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...", help="Create key at console.groq.com → Keys")
        submitted = st.form_submit_button("Sign in")
        if submitted and api_key:
            ok, err = _validate_groq_api_key(api_key)
            if ok:
                key_val = api_key.strip()
                st.session_state.authenticated = True
                st.session_state.groq_api_key = key_val
                st.session_state.auth_provider = "groq"
                st.session_state.username = "Groq " + _groq_key_fingerprint(key_val) if _groq_key_fingerprint(key_val) else "Groq User"
                st.session_state._user_db_key = hashlib.sha256(key_val.encode()).hexdigest()[:16]
                _save_user_groq_key(key_val)
                st.success("Signed in. Loading app...")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(err or "Invalid key.")
        elif submitted:
            st.error("Please enter your Groq API key.")
    st.markdown("[Create API key at Groq Console →](https://console.groq.com/keys)")
    st.caption("Login is only via Groq API key. No Google or password—just your key.")
    st.stop()

# (Login is Groq key only; no separate key gate.)
_user_groq = _get_user_groq_key()
if False and not _user_groq:  # dead: key-only login
    st.markdown("""
    <style>
    * { font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    section[data-testid="stSidebar"] { display: none !important; }
    html, body { margin: 0; padding: 0; min-height: 100vh; background: #fafafa !important; }
    .main { background: #fafafa !important; padding: 0 !important; }
    .block-container { padding: 2rem 1.5rem !important; max-width: 420px !important; margin: 0 auto !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("### 🔑 Groq API key required")
    st.markdown("To use ElasticNode AI, add your **Groq API key**. Create one (free) and paste it below.")
    with st.form("groq_key_form"):
        api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...", help="Your key is stored securely per account.")
        submitted = st.form_submit_button("Save & continue")
        if submitted:
            if api_key and api_key.strip():
                _save_user_groq_key(api_key.strip())
                st.session_state.groq_api_key = api_key.strip()
                st.success("Key saved. Loading app...")
                st.rerun()
            else:
                st.error("Please enter a valid API key.")
    st.markdown("[Create API key at Groq Console →](https://console.groq.com/keys)")
    st.caption("You won’t get access to the app until a key is saved for your account.")
    st.stop()

# Ensure Groq key in session (set at login; or from saved user / env)
if "groq_api_key" not in st.session_state or not (st.session_state.get("groq_api_key") or "").strip():
    st.session_state.groq_api_key = _get_user_groq_key() or os.getenv("GROQ_API_KEY", "")

# Load per-user conversation history once per session, so refresh / restart ke baad bhi
# account ke sath chats wapas aa sakein.
if st.session_state.get("authenticated") and not st.session_state.get("_history_loaded"):
    saved_history = _load_user_history()
    if saved_history:
        st.session_state.conversations = saved_history
        # Default to last conversation as active
        try:
            last_id = list(saved_history.keys())[-1]
            st.session_state.conversation_id = last_id
            conv = saved_history.get(last_id, {})
            st.session_state.messages = conv.get("messages", []).copy()
            st.session_state.docs = list(conv.get("docs", []))
            st.session_state.doc_files = dict(conv.get("doc_files", {}))
        except Exception:
            pass
    st.session_state["_history_loaded"] = True

# Groq session sync: Jab tak Groq account login hai tab tak site login. Groq se logout/revoke = site se auto logout.
# Key ko periodically validate karte hain; invalid (revoke ya Groq logout) hone par session clear → user ko dubara key dalni padegi.
if st.session_state.get("authenticated") and st.session_state.get("auth_provider") == "groq":
    _key = (st.session_state.get("groq_api_key") or "").strip()
    if _key:
        _last_check = st.session_state.get("_groq_key_last_validated")
        _now = time.time()
        if _last_check is None or (_now - _last_check) > 30:
            _ok, _ = _validate_groq_api_key(_key)
            st.session_state["_groq_key_last_validated"] = _now if _ok else 0
            if not _ok:
                _db_key = st.session_state.get("_user_db_key")
                users = _load_users()
                if _db_key and _db_key in users:
                    users[_db_key].pop("groq_api_key", None)
                    _save_users(users)
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.auth_provider = None
                st.session_state.groq_api_key = ""
                st.session_state._user_db_key = None
                st.session_state.generator = None
                st.session_state.model_loaded = False
                for _k in ["messages", "conversations", "current_view", "last_response", "chat_draft", "docs", "doc_files", "_pending_docs_for_badge", "_groq_key_last_validated"]:
                    st.session_state.pop(_k, None)
                try:
                    _lg = globals().get("load_generator")
                    if _lg is not None:
                        _lg.clear()
                except Exception:
                    pass
                st.rerun()

# Design: consistent spacing (8px base), clean colors, subtle & tasteful
st.markdown("""
<style>
:root {
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --radius: 12px;
    --neutral-800: #262626;
    --neutral-700: #404040;
    --neutral-600: #525252;
    --neutral-500: #737373;
    --neutral-400: #a3a3a3;
    --neutral-300: #d4d4d4;
    --neutral-200: #e5e5e5;
    --neutral-100: #f5f5f5;
    --neutral-50: #fafafa;
    --text: var(--neutral-800);
    --text-muted: var(--neutral-600);
    --border: var(--neutral-200);
    --surface: #ffffff;
    --bg: var(--neutral-50);
    /* ChatGPT-style typography */
    --font-ui: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    --text-base: 0.9375rem;   /* 15px - body / chat messages */
    --text-sm: 0.8125rem;    /* 13px - captions, secondary */
    --text-md: 0.875rem;     /* 14px - buttons, labels, UI */
    --text-lg: 1rem;         /* 16px - input, emphasis */
    --text-xl: 1.125rem;     /* 18px - subheadings */
    --text-2xl: 1.25rem;     /* 20px - headings */
}

* {
    font-family: var(--font-ui);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
    margin: 0;
    background: var(--bg);
    font-size: var(--text-base);
    font-family: var(--font-ui);
}

.main {
    background: var(--bg);
    padding: 0 !important;
    min-height: 100vh;
}

.block-container {
    padding: 1rem 1.5rem !important;
    max-width: 100% !important;
    min-height: calc(100vh - 2rem) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
}

/* Sidebar: keep styling minimal so native collapse works */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    font-family: var(--font-ui) !important;
    font-size: var(--text-base) !important;
}

.stButton > button {
    padding: 0.5rem 1rem !important;
    font-size: var(--text-md) !important;
    border-radius: 8px !important;
    min-height: 36px !important;
    height: 36px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
    transition: all 0.15s ease;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--neutral-800) !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--neutral-600) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--neutral-600) !important;
    border: 1px solid var(--border) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: var(--neutral-50) !important;
    border-color: var(--neutral-400) !important;
}

/* Status bar */
div[data-testid="column"]:last-child [data-testid="stMarkdown"] {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    height: 36px;
}

/* Chat container */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    background: var(--surface);
    border-radius: var(--radius);
    padding: var(--space-lg);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    border: 1px solid var(--border);
    max-height: calc(100vh - 180px) !important;
    min-height: 0;
    overflow-y: auto;
}

/* Performance Dashboard - modern metrics page */
.metrics-dashboard {
    max-width: 720px;
    margin: 0 auto;
    padding: 0.5rem 0 2rem;
}
.metrics-dashboard .dashboard-title {
    font-size: var(--text-2xl);
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--neutral-800);
    text-align: center;
    margin: 0 0 1.75rem 0;
}
.metrics-empty-state {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid rgba(14, 165, 233, 0.2);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: #0c4a6e;
    font-size: var(--text-base);
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.06);
}
.metrics-empty-state strong { color: #0369a1; }
.metrics-dashboard [data-testid="stMetric"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.metrics-dashboard [data-testid="stMetric"]:hover {
    border-color: var(--neutral-300);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.metrics-dashboard [data-testid="stMetricValue"] { font-weight: 600; }
.metrics-dashboard .js-plotly-plot { border-radius: 12px; }

/* Conversation History - same modern layout as dashboard */
.history-dashboard {
    max-width: 720px;
    margin: 0 auto;
    padding: 0.5rem 0 2rem;
}
.history-empty-state {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid rgba(14, 165, 233, 0.2);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: #0c4a6e;
    font-size: var(--text-base);
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.06);
}
.history-empty-state strong { color: #0369a1; }

/* Welcome section - takes most of the viewport so input bar sits at bottom */
.welcome-hero {
    padding: 0 !important;
    text-align: center !important;
    animation: fadeIn 0.6s ease-out;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-height: 72vh !important;
}

.welcome-hero img,
.welcome-hero .hero-logo {
    margin: 0 auto 1.5rem auto !important;
    display: block !important;
    max-width: 360px;
    width: 360px;
    height: auto;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-title {
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    color: var(--text);
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.02em;
    text-align: center !important;
}

.hero-subtitle {
    font-size: var(--text-xl) !important;
    color: var(--neutral-500) !important;
    margin-bottom: 0 !important;
    font-weight: 400 !important;
    text-align: center !important;
    max-width: 650px;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Feature cards - FIXED LAYOUT */
.feature-grid {
    margin-top: 2rem !important;
    width: 100% !important;
    max-width: 1000px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

[data-testid="column"]:has(.feature-card) {
    padding: 0 0.5rem !important;
}

.feature-card {
    background: var(--surface);
    padding: var(--space-lg);
    border-radius: var(--radius);
    text-align: center;
    border: 1px solid var(--border);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    height: 100%;
}

.feature-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04);
    transform: translateY(-2px);
}

.feature-icon { font-size: var(--text-2xl); margin-bottom: var(--space-sm); }
.feature-title { font-size: var(--text-lg); font-weight: 600; color: var(--text); margin-bottom: var(--space-xs); }
.feature-desc { font-size: var(--text-md); color: var(--text-muted); line-height: 1.5; }

/* Chat messages - ChatGPT-style body text */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: var(--space-sm) 0 !important;
    margin: var(--space-xs) 0 !important;
    font-size: var(--text-base) !important;
    font-family: var(--font-ui) !important;
}
.stChatMessage [data-testid="stMarkdown"], .stChatMessage .stMarkdown {
    font-size: var(--text-base) !important;
    font-family: var(--font-ui) !important;
}

.stChatMessage[data-testid*="user"] > div {
    background: linear-gradient(135deg, var(--neutral-800) 0%, var(--neutral-700) 100%) !important;
    color: white !important;
    padding: var(--space-sm) var(--space-md) !important;
    border-radius: var(--radius) var(--radius) var(--space-xs) var(--radius) !important;
    display: inline-block !important;
    max-width: 85% !important;
    margin-left: auto !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.stChatMessage[data-testid*="user"] * { color: white !important; }

.stChatMessage[data-testid*="assistant"] > div {
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: var(--space-sm) var(--space-md) !important;
    border-radius: var(--radius) var(--radius) var(--radius) var(--space-xs) !important;
    border: 1px solid var(--neutral-200) !important;
    display: inline-block !important;
    max-width: 85% !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Chat input - FIXED POSITIONING */
.stChatInputContainer {
    margin-top: 2rem !important;
    padding: 1rem 0 !important;
    background: transparent !important;
    border: none !important;
    position: relative;
}

/* Chat input row: one bar — input + send + paperclip aligned like a single control */
div[data-testid="stHorizontalBlock"]:has(.stChatInput) {
    gap: 0 !important;
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(.stChatInput) > div:first-child {
    flex: 1 !important;
    min-width: 0 !important;
}

/* Input bar: leave 48px on the right for paperclip so it sits next to send, not overlapping */
.stChatInput > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-right: none !important;
    border-radius: var(--radius) 0 0 var(--radius) !important;
    box-shadow: none !important;
    padding-right: 48px !important;
    font-family: var(--font-ui) !important;
    font-size: var(--text-base) !important;
}
.stChatInput input, .stChatInput textarea {
    font-family: var(--font-ui) !important;
    font-size: var(--text-base) !important;
}
.stChatInput > div:focus-within {
    border-color: var(--neutral-800) !important;
    box-shadow: 0 0 0 3px rgba(38, 38, 38, 0.08) !important;
}

/* Paperclip column: part of the same bar — no separate box, same bg as input */
div[data-testid="stHorizontalBlock"]:has(.stChatInput) > div:last-child {
    margin-left: -48px !important;
    flex: 0 0 48px !important;
    max-width: 48px !important;
    position: relative !important;
    z-index: 2 !important;
    background: var(--surface) !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove any wrapper box so paperclip is inside the bar, not a separate card */
div[data-testid="stHorizontalBlock"]:has(.stChatInput) > div:last-child > div,
[data-testid="stPopover"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    display: flex !important;
    align-items: stretch !important;
    height: 100% !important;
    min-height: 0 !important;
}

/* ── Icon buttons (📎 popover trigger & ➤ send) ── identical styling ── */
[data-testid="stPopover"] > button,
.stButton > button[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 40px !important;
    height: 40px !important;
    min-width: 44px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: var(--text-lg) !important;
    color: var(--neutral-700) !important;
    transition: background 0.15s ease, color 0.15s ease;
}

[data-testid="stPopover"] > button:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: var(--neutral-100) !important;
    color: var(--neutral-800) !important;
}

/* Upload button inside popover */
[data-testid="stPopover"] button[kind="primary"],
[data-testid="stPopover"] .stButton > button {
    background: var(--neutral-800) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: var(--space-sm) var(--space-md) !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    min-height: auto !important;
    height: auto !important;
    width: 100% !important;
}

[data-testid="stPopover"] button[kind="primary"]:hover,
[data-testid="stPopover"] .stButton > button:hover {
    background: var(--neutral-600) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--neutral-50);
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: var(--space-md);
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--neutral-400);
    background: var(--surface);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: var(--text-2xl) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: var(--text-sm);
}

/* Expander */
.streamlit-expanderHeader {
    color: var(--text) !important;
    font-weight: 600 !important;
    padding: var(--space-sm) var(--space-md) !important;
    border-radius: var(--radius) !important;
    background: var(--neutral-50) !important;
}

.streamlit-expanderHeader:hover {
    background: var(--neutral-100) !important;
}

.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    margin-top: -4px;
}

/* Messages */
.stSuccess, .stInfo, .stWarning {
    padding: var(--space-md) !important;
    border-radius: var(--radius) !important;
    border-left: 3px solid currentColor;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--neutral-100);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--neutral-300);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--neutral-400);
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--neutral-800) !important;
    border-right-color: var(--neutral-300) !important;
}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--neutral-500) !important;
    font-size: var(--text-sm) !important;
    font-weight: 500 !important;
}

.stMarkdown, .stMarkdown p {
    font-size: var(--text-base) !important;
    font-family: var(--font-ui) !important;
}
.stMarkdown h2, .stMarkdown h3 {
    margin: var(--space-sm) 0 !important;
    color: var(--text) !important;
    font-size: var(--text-xl) !important;
    font-family: var(--font-ui) !important;
}

/* ── Profile avatar circle button (top-right) ── */
[data-testid="stColumns"]:first-of-type [data-testid="column"]:last-child [data-testid="stPopover"] > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: var(--text-sm) !important;
    font-weight: 700 !important;
    border: 2px solid #e5e5e5 !important;
    float: right !important;
    margin-left: auto !important;
}
[data-testid="stColumns"]:first-of-type [data-testid="column"]:last-child [data-testid="stPopover"] > button:hover {
    opacity: 0.85 !important;
    border-color: #764ba2 !important;
}

/* Top-bar menu (☰) dropdown – shift buttons right so left/right space is even */
[data-testid="stColumns"]:first-of-type [data-testid="column"]:first-child [data-testid="stPopover"] > div {
    padding-left: 12px !important;
    padding-right: 8px !important;
}

/* Menu dropdown – boundaries fully transparent: button + its wrapper, no box visible */
[data-testid="stPopover"]:has(.menu-dropdown-spacer) [data-testid="stVerticalBlock"]:has(.stButton),
[data-testid="stPopover"]:has(.menu-dropdown-spacer) .stButton,
[data-testid="stPopover"]:has(.menu-dropdown-spacer) .stButton > button,
[data-testid="stColumns"]:first-of-type [data-testid="column"]:first-child [data-testid="stPopover"] > div .stButton,
[data-testid="stColumns"]:first-of-type [data-testid="column"]:first-child [data-testid="stPopover"] > div .stButton > button,
.menu-dropdown-spacer ~ * .stButton,
.menu-dropdown-spacer ~ * .stButton > button,
div:has(> .menu-dropdown-spacer) .stButton,
div:has(> .menu-dropdown-spacer) .stButton > button {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--neutral-700) !important;
    min-height: auto !important;
    padding: 0.35rem 0 !important;
}
[data-testid="stPopover"]:has(.menu-dropdown-spacer) [data-testid="stVerticalBlock"]:has(.stButton) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stPopover"]:has(.menu-dropdown-spacer) .stButton > button:hover,
[data-testid="stColumns"]:first-of-type [data-testid="column"]:first-child [data-testid="stPopover"] > div .stButton > button:hover,
.menu-dropdown-spacer ~ * .stButton > button:hover,
div:has(> .menu-dropdown-spacer) .stButton > button:hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: var(--neutral-800) !important;
}
</style>
""", unsafe_allow_html=True)

# Session State
@st.cache_resource
def load_generator(api_key: str = ""):
    """api_key: from session (user-entered) or pass env key; cache is per key."""
    key = (api_key or "").strip() or os.getenv("GROQ_API_KEY", "")
    gen = Generator(api_key=key)
    return gen

def init():
    defaults = {
        "messages": [], "docs": [], "doc_files": {}, "_pending_docs_for_badge": None,
        "ingestor": None, "searcher": None, "generator": None,
        "last_response": None, "metrics_history": [], "show_upload": False,
        "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "conversations": {}, "enable_history": True, "model_loaded": False,
        "_runtime_version": None, "_history_loaded": False,
        "auth_provider": None,
        "current_view": "chat", "rename_conversation": None, "sidebar_open": True
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure stale session objects are rebuilt after backend fixes.
    if st.session_state.get("_runtime_version") != BACKEND_RUNTIME_VERSION:
        st.session_state.ingestor = None
        st.session_state.searcher = None
        st.session_state.generator = None
        st.session_state.model_loaded = False
        try:
            load_generator.clear()
        except Exception:
            pass
        st.session_state["_runtime_version"] = BACKEND_RUNTIME_VERSION
    
    if not st.session_state.ingestor:
        try: st.session_state.ingestor = BulkIngest()
        except Exception as e: st.error(f"Backend error: {e}")
    
    if not st.session_state.searcher:
        try: st.session_state.searcher = HybridSearch()
        except Exception as e: st.error(f"Search error: {e}")
    
    if not st.session_state.model_loaded:
        try:
            with st.spinner("🚀 Loading AI model..."):
                _key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY") or ""
                st.session_state.generator = load_generator(_key)
                st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"❌ Model failed to load: {e}")

init()

# ============================================
# MAIN CONTENT AREA
# ============================================

# Top bar: ☰ Menu (left) + Profile Avatar (right)
_display_name = st.session_state.get("username") or "User"
_initials = "".join([w[0].upper() for w in _display_name.split()[:2]]) if _display_name else "U"

menu_col, spacer_col, avatar_col = st.columns([0.06, 0.84, 0.10], vertical_alignment="center")

with avatar_col:
    with st.popover(
        f"**{_initials}**",
        use_container_width=True,
    ):
        st.markdown(f"""
        <div style="text-align:center; padding:8px 0;">
            <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);
                        color:white;display:inline-flex;align-items:center;justify-content:center;
                        font-size:1.2rem;font-weight:700;margin-bottom:8px;">{_initials}</div>
            <div style="font-weight:600;font-size:0.9rem;">{_display_name}</div>
            <div style="font-size:0.75rem;color:#666;margin-top:2px;">
                {'✅ Model Ready' if st.session_state.get('model_loaded') else '⏳ Loading...'}
                 · {len(st.session_state.get('docs', []))} docs
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.button("🚪 Logout", key="avatar_logout_btn", use_container_width=True,
                  type="secondary", on_click=_handle_logout)

with menu_col:
    with st.popover("☰", use_container_width=True):
        # Spacer so "New Chat" sits more vertically centered in the popover section
        st.markdown('<div class="menu-dropdown-spacer" style="margin-top: 32px;"></div>', unsafe_allow_html=True)
        if st.button("➕ New Chat", key="menu_new_chat", use_container_width=True, type="secondary"):
            if st.session_state.messages:
                st.session_state.conversations[st.session_state.conversation_id] = {
                    "messages": st.session_state.messages.copy(),
                    "title": st.session_state.messages[0]["content"][:40] + "..." if st.session_state.messages else "New Chat",
                    "docs": list(st.session_state.get("docs", [])),
                    "doc_files": dict(st.session_state.get("doc_files", {})),
                }
                _save_user_history()
            st.session_state.messages = []
            st.session_state.docs = []
            st.session_state.doc_files = {}
            st.session_state._pending_docs_for_badge = None
            st.session_state.last_response = None
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_view = "chat"
            st.rerun()

        st.markdown("---")

        if st.button("💬 Chat", key="menu_nav_chat", use_container_width=True, type="secondary"):
            st.session_state.current_view = "chat"
            st.rerun()
        if st.button("📊 Metrics", key="menu_nav_metrics", use_container_width=True, type="secondary"):
            st.session_state.current_view = "metrics"
            st.rerun()
        if st.button("📚 History", key="menu_nav_history", use_container_width=True, type="secondary"):
            st.session_state.current_view = "history"
            st.rerun()
        if st.button("⚙️ Settings", key="menu_nav_settings", use_container_width=True, type="secondary"):
            st.session_state.current_view = "settings"
            st.rerun()

        st.markdown("---")

# METRICS VIEW
if st.session_state.current_view == "metrics":
    st.markdown('<div class="metrics-dashboard">', unsafe_allow_html=True)
    st.markdown('<h2 class="dashboard-title">Performance Dashboard</h2>', unsafe_allow_html=True)

    if not st.session_state.metrics_history:
        st.markdown(
            '<div class="metrics-empty-state">📊 No metrics yet. <strong>Start chatting!</strong> to see latency, cache rate and more here.</div>',
            unsafe_allow_html=True
        )
    else:
        total = len(st.session_state.metrics_history)
        avg_lat = sum(m.get('total_latency', 0) for m in st.session_state.metrics_history) / total
        cached = sum(1 for m in st.session_state.metrics_history if m.get('cached', False))

        # 🔢 Count unique documents across ALL chats (current + history),
        # so Documents metric doesn't drop to 0 on a fresh chat.
        all_docs = set(st.session_state.get("docs", []) or [])
        for conv in (st.session_state.get("conversations", {}) or {}).values():
            for d in conv.get("docs", []) or []:
                all_docs.add(d)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries", total)
        col2.metric("Avg Latency", f"{avg_lat:.2f}s")
        col3.metric("Cache Rate", f"{(cached/total*100):.0f}%")
        col4.metric("Documents", len(all_docs))

        if st.session_state.metrics_history:
            latest = st.session_state.metrics_history[-1]
            breakdown = {
                "Component": ["Search", "Rerank", "Embed"],
                "Time": [latest.get('search_time', 0), latest.get('rerank_time', 0), latest.get('embedding_time', 0)]
            }
            fig = px.pie(breakdown, values="Time", names="Component", title="Latest Query Breakdown", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# HISTORY VIEW
elif st.session_state.current_view == "history":
    st.markdown('<div class="history-dashboard">', unsafe_allow_html=True)
    st.markdown('<h2 class="dashboard-title">Conversation History</h2>', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True, type="primary"):
        if st.session_state.messages:
            st.session_state.conversations[st.session_state.conversation_id] = {
                "messages": st.session_state.messages.copy(),
                "title": st.session_state.messages[0]['content'][:40] + "...",
                "docs": list(st.session_state.get("docs", [])),
                "doc_files": dict(st.session_state.get("doc_files", {})),
            }
            _save_user_history()
        st.session_state.messages = []
        st.session_state.docs = []
        st.session_state.doc_files = {}
        st.session_state._pending_docs_for_badge = None
        st.session_state.last_response = None
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.current_view = "chat"
        st.rerun()

    st.markdown("---")

    if st.session_state.conversations:
        for conv_id, conv in reversed(list(st.session_state.conversations.items())[-10:]):
            if st.button(conv['title'], key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.messages = conv.get("messages", []).copy()
                st.session_state.docs = list(conv.get("docs", []))
                st.session_state.doc_files = dict(conv.get("doc_files", {}))
                st.session_state._pending_docs_for_badge = None
                st.session_state.last_response = None
                st.session_state.conversation_id = conv_id
                st.session_state.current_view = "chat"
                st.rerun()
    else:
        st.markdown(
            '<div class="history-empty-state">💬 No conversation history yet. <strong>Start a chat</strong> to see it here.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# SETTINGS VIEW
elif st.session_state.current_view == "settings":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("<h2 class='dashboard-title' style='text-align:center;'>Settings</h2>", unsafe_allow_html=True)
    st.markdown("**Groq API key** — used for AI answers. Change it here to use a different key.")
    with st.form("settings_groq_key"):
        current_key = st.session_state.get("groq_api_key") or ""
        new_key = st.text_input("Groq API key", value=current_key, type="password", placeholder="gsk_...", key="settings_groq_key_input")
        st.caption("Get a key: [console.groq.com/keys](https://console.groq.com/keys)")
        if st.form_submit_button("Save key"):
            if (new_key or "").strip():
                st.session_state.groq_api_key = new_key.strip()
                st.session_state.generator = None
                st.session_state.model_loaded = False
                try:
                    load_generator.clear()
                except Exception:
                    pass
                st.success("Key updated. Model will reload on next use.")
                st.rerun()
            else:
                st.error("Enter a valid API key.")
    st.markdown('</div>', unsafe_allow_html=True)

# ACCOUNTS VIEW
elif st.session_state.current_view == "accounts":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("<h2 class='dashboard-title' style='text-align:center;'>Accounts</h2>", unsafe_allow_html=True)
    st.info("Accounts page — manage your account here.")
    st.markdown('</div>', unsafe_allow_html=True)

# CHAT VIEW (DEFAULT)
else:
    user_input = None

    # Callback: fires when user presses Enter in the text_input
    def _on_enter():
        val = st.session_state.get("chat_draft", "").strip()
        if val:
            st.session_state["_pending_input"] = val

    # Clear input after message sent (must happen BEFORE widget renders)
    if st.session_state.pop("_clear_draft", False):
        st.session_state["chat_draft"] = ""

    if not st.session_state.messages:
        logo_path = os.path.join(os.path.dirname(__file__), "static", "logo.png")
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img class="hero-logo" src="data:image/png;base64,{logo_b64}" alt="ElasticNode AI" width="360" />'
        else:
            logo_html = '<div class="hero-title">✨ ElasticNode AI</div>'
        has_docs = len(st.session_state.get("docs", [])) > 0
        if has_docs:
            status_html = '<div style="margin-top:28px;padding:12px 24px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;text-align:center;color:#16a34a;font-size:0.85rem;">✅ Document loaded — ask me anything!</div>'
        else:
            status_html = '<div style="margin-top:28px;padding:16px 28px;border:1.5px dashed #ccc;border-radius:12px;text-align:center;color:#999;font-size:0.85rem;">📎 Upload a PDF using the <b>clip icon</b> below to get started</div>'

        st.markdown(
            f'<div class="welcome-hero">'
            f'{logo_html}'
            f'<div class="hero-subtitle">Your Intelligent Document Assistant powered by Hybrid Search + AI</div>'
            f'<div style="display:flex;gap:28px;justify-content:center;margin-top:32px;flex-wrap:wrap;">'
            f'<div style="text-align:center;opacity:0.65;"><div style="font-size:1.4rem;">📄</div><div style="font-size:0.75rem;color:#888;margin-top:4px;">Upload PDFs</div></div>'
            f'<div style="text-align:center;opacity:0.65;"><div style="font-size:1.4rem;">🔍</div><div style="font-size:0.75rem;color:#888;margin-top:4px;">Hybrid Search</div></div>'
            f'<div style="text-align:center;opacity:0.65;"><div style="font-size:1.4rem;">⚡</div><div style="font-size:0.75rem;color:#888;margin-top:4px;">Instant Answers</div></div>'
            f'<div style="text-align:center;opacity:0.65;"><div style="font-size:1.4rem;">📚</div><div style="font-size:0.75rem;color:#888;margin-top:4px;">Source Citations</div></div>'
            f'</div>'
            f'{status_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Input bar: text + [📎 ➤] icons (nested equal columns)
        input_col, icons_col = st.columns([0.88, 0.12], vertical_alignment="center", gap="small")
        with input_col:
            draft = st.text_input(
                "message",
                key="chat_draft",
                placeholder="Ask anything about your documents…",
                label_visibility="collapsed",
                on_change=_on_enter,
            )
        with icons_col:
            ic1, ic2 = st.columns(2, gap="small")
            with ic1:
                with st.popover("📎", use_container_width=True):
                    st.caption("Add document")
                    uploaded_file = st.file_uploader(
                        "Choose PDF", type=["pdf"], label_visibility="collapsed", key="uploader_welcome",
                    )
                    is_indexing = st.session_state.get("_indexing", False)
                    if uploaded_file:
                        if st.button("📤 Upload", key="popover_index", use_container_width=True,
                                     type="primary", disabled=is_indexing):
                            if uploaded_file.name not in st.session_state.docs:
                                st.session_state["_indexing"] = True
                                try:
                                    with st.spinner("Indexing..."):
                                        pdf_bytes = uploaded_file.getvalue()
                                        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                                        text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
                                        if text.strip():
                                            result = st.session_state.ingestor.bulk_index_document(text, uploaded_file.name)
                                            st.session_state.docs.append(uploaded_file.name)
                                            # store original bytes for download/open
                                            st.session_state.doc_files[uploaded_file.name] = pdf_bytes
                                            # show doc badge only on next user message
                                            pending = st.session_state.get("_pending_docs_for_badge") or []
                                            if uploaded_file.name not in pending:
                                                pending.append(uploaded_file.name)
                                            st.session_state["_pending_docs_for_badge"] = pending
                                            chunks = result.get("chunks", "?") if isinstance(result, dict) else "?"
                                            t = result.get("time", "?") if isinstance(result, dict) else "?"
                                            st.success(f"✅ Indexed: {chunks} chunks in {t}s")
                                except Exception as e:
                                    st.error(str(e))
                                finally:
                                    st.session_state["_indexing"] = False
                                    time.sleep(0.8)
                                    st.rerun()
                    if st.session_state.docs:
                        st.caption("Indexed")
                        for doc in st.session_state.docs:
                            st.markdown(f"· {doc}", unsafe_allow_html=True)
            with ic2:
                send_clicked = st.button("➤", key="send_btn_welcome", use_container_width=True, type="secondary")

        pending = st.session_state.pop("_pending_input", None)
        if pending:
            user_input = pending
        elif send_clicked and (draft or "").strip():
            user_input = (draft or "").strip()

    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                # Show attached document badge ONLY on the first question after upload
                if msg["role"] == "user" and msg.get("docs"):
                    doc_files = st.session_state.get("doc_files", {}) or {}
                    for doc_name in msg["docs"]:
                        with st.popover(f"📄 {doc_name}", use_container_width=False):
                            st.caption("Open or download the uploaded document")
                            data = doc_files.get(doc_name)
                            if data:
                                st.download_button(
                                    "Download PDF",
                                    data=data,
                                    file_name=doc_name,
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                            else:
                                st.info("Original file not available in this session.")
                st.markdown(msg["content"])

        if st.session_state.last_response:
            r = st.session_state.last_response

            with st.expander("⚡ Performance Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", f"{r.get('total_latency', 0):.2f}s")
                col2.metric("Search", f"{r.get('search_time', 0):.2f}s")
                rt = r.get('rerank_time', 0)
                col3.metric("Rerank", f"{rt:.2f}s" if rt > 0 else "Off")
                col4.metric("Cached", "Yes ⚡" if r.get('cached') else "No")

                method_label = "Hybrid (BM25 + Vector)"
                if rt > 0:
                    method_label += " + Cross-Encoder Rerank"

                st.caption(
                    f"Method: **{method_label}** · "
                    f"Embedding: {r.get('embedding_time', 0):.3f}s · "
                    f"Results: {r.get('num_results', len(r.get('results', [])))} docs"
                )

            results = r.get("results", [])
            if results:
                # Decide the primary document for this query from stored metadata or recompute
                primary_source = r.get("primary_source")

                if not primary_source:
                    doc_best_scores = {}
                    for res in results:
                        src_name = res.get("source", res.get("metadata", {}).get("filename", "unknown"))
                        try:
                            s_val = float(res.get("score", 0.0) or 0.0)
                        except Exception:
                            s_val = 0.0
                        if src_name not in doc_best_scores or s_val > doc_best_scores[src_name]:
                            doc_best_scores[src_name] = s_val
                    if doc_best_scores:
                        primary_source = max(doc_best_scores.items(), key=lambda x: x[1])[0]

                if primary_source:
                    # Only show chunks from the most relevant document for this query
                    display_results = [
                        res for res in results
                        if res.get("source", res.get("metadata", {}).get("filename", "unknown")) == primary_source
                    ][:8]
                else:
                    # Fallback: show top-ranked chunks if no primary source could be determined
                    display_results = results[:8]

                with st.expander(f"📚 Sources ({len(display_results)} chunks)", expanded=False):
                    for i, chunk in enumerate(display_results):
                        src = chunk.get("source", chunk.get("metadata", {}).get("filename", "unknown"))
                        cid = chunk.get("chunk_id", f"chunk-{i}")
                        # Use the hybrid ES score for display (always non-negative)
                        base_score = float(chunk.get("score", 0.0) or 0.0)
                        text_preview = chunk.get("text", "")[:250]
                        st.markdown(
                            f"**{i+1}. {src}** · `{cid}` · Score: `{base_score:.4f}`\n\n"
                            f">{text_preview}{'...' if len(chunk.get('text', '')) > 250 else ''}",
                            unsafe_allow_html=True,
                        )

        # Input bar: text + [📎 ➤] icons (nested equal columns)
        input_col, icons_col = st.columns([0.88, 0.12], vertical_alignment="center", gap="small")
        with input_col:
            draft = st.text_input(
                "message",
                key="chat_draft",
                placeholder="Ask anything about your documents…",
                label_visibility="collapsed",
                on_change=_on_enter,
            )
        with icons_col:
            ic1, ic2 = st.columns(2, gap="small")
            with ic1:
                with st.popover("📎", use_container_width=True):
                    st.caption("Add document")
                    uploaded_file = st.file_uploader(
                        "Choose PDF", type=["pdf"], label_visibility="collapsed", key="uploader_chat",
                    )
                    is_indexing = st.session_state.get("_indexing", False)
                    if uploaded_file:
                        if st.button("📤 Upload", key="popover_index_2", use_container_width=True,
                                     type="primary", disabled=is_indexing):
                            if uploaded_file.name not in st.session_state.docs:
                                st.session_state["_indexing"] = True
                                try:
                                    with st.spinner("Indexing..."):
                                        pdf_bytes = uploaded_file.getvalue()
                                        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                                        text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
                                        if text.strip():
                                            result = st.session_state.ingestor.bulk_index_document(text, uploaded_file.name)
                                            st.session_state.docs.append(uploaded_file.name)
                                            # store original bytes for download/open
                                            st.session_state.doc_files[uploaded_file.name] = pdf_bytes
                                            # show doc badge only on next user message
                                            pending = st.session_state.get("_pending_docs_for_badge") or []
                                            if uploaded_file.name not in pending:
                                                pending.append(uploaded_file.name)
                                            st.session_state["_pending_docs_for_badge"] = pending
                                            chunks = result.get("chunks", "?") if isinstance(result, dict) else "?"
                                            t = result.get("time", "?") if isinstance(result, dict) else "?"
                                            st.success(f"✅ Indexed: {chunks} chunks in {t}s")
                                except Exception as e:
                                    st.error(str(e))
                                finally:
                                    st.session_state["_indexing"] = False
                                    time.sleep(0.8)
                                    st.rerun()
                    if st.session_state.docs:
                        st.caption("Indexed")
                        for doc in st.session_state.docs:
                            st.markdown(f"· {doc}", unsafe_allow_html=True)
            with ic2:
                send_clicked = st.button("➤", key="send_btn_chat", use_container_width=True, type="secondary")

        pending = st.session_state.pop("_pending_input", None)
        if pending:
            user_input = pending
        elif send_clicked and (draft or "").strip():
            user_input = (draft or "").strip()

    if user_input:
        if not st.session_state.docs:
            st.warning("Upload a document first.")
        elif not st.session_state.model_loaded:
            st.warning("Model still loading…")
        else:
            pending_docs = st.session_state.pop("_pending_docs_for_badge", None)
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                # Attach docs ONLY to the first message after upload
                "docs": list(pending_docs) if pending_docs else [],
            })

            try:
                # 🌟 Smart follow-up handling — use last question + docs for vague queries like "anything more?"
                raw_query = user_input
                q_lower = raw_query.lower().strip()
                words = q_lower.split()
                retrieval_query = raw_query
                is_followup = False
                is_generic_followup = False  # true only for "anything more?" with no topic → reuse cache

                # "anything more about X" / "tell me more about X" → search for topic X (expand short forms)
                topic_expand = {"elastic": "elasticsearch", "oled": "OLEDs", "ai": "artificial intelligence", "jd": "job description"}
                if "about" in q_lower and any(kw in q_lower for kw in ["more", "aur", "batao", "btao", "tell me more"]):
                    parts = q_lower.split("about", 1)
                    if len(parts) == 2:
                        topic = parts[1].strip().rstrip("?").strip()
                        if topic:
                            retrieval_query = (topic + " " + topic_expand.get(topic.lower(), "")).strip()
                            is_followup = True
                            print(f"🔁 [FRONT] Topic follow-up: '{topic}' → retrieval_query='{retrieval_query}'")

                generic_followups = ["anything more", "anything else", "aur batao", "aur btao", "tell me more", "more?", "aur?"]
                if not is_followup and len(words) <= 4 and any(kw in q_lower for kw in generic_followups):
                    prev_user = None
                    for _m in reversed(st.session_state.messages[:-1]):
                        if _m.get("role") == "user":
                            prev_user = _m.get("content", "")
                            break
                    if prev_user and st.session_state.get("last_response"):
                        retrieval_query = prev_user
                        is_followup = True
                        is_generic_followup = True
                        print(f"🔁 [FRONT] Follow-up detected. Reusing previous retrieval for: '{prev_user}'")

                # Greetings (hi, hey, hy, hello): don't run query search — use broad context so we never show "No relevant documents"
                greeting_words = ("hi", "hey", "hello", "hy", "hii", "heyy", "hlw", "hlo", "namaste", "namaskar", "hey there", "hi there", "helo", "hllo")
                q_clean = q_lower.replace(" ", "").strip()
                is_greeting = (q_lower.strip() in greeting_words) or (q_clean in greeting_words) or (len(words) <= 2 and any(g in q_lower for g in greeting_words))

                query_lower = retrieval_query.lower().strip()
                _broad_keywords = [
                    # Explicit summary-style phrases only
                    'summarise', 'summarize', 'summary', 'overview',
                    'tldr', 'tl;dr', 'gist', 'smrz', 'summ',
                    'entire document', 'whole document', 'complete document',
                    'entire pdf', 'whole pdf', 'full document',
                ]
                is_summary = any(kw in query_lower for kw in _broad_keywords)
                # Very short queries (1-3 words) are likely broad/exploratory
                is_short = len(query_lower.split()) <= 3
                # For summaries: fetch 15 chunks using filename-based search_all
                # For specific questions: fetch 8 with reranker for precision
                num_docs = 15 if is_summary else 8
                use_reranker = not is_summary  # reranker narrows topics, bad for summaries

                with st.spinner("🔍 Searching documents..."):
                    if is_generic_followup and st.session_state.get("last_response"):
                        resp = st.session_state.last_response
                        print("\n🔎 [FRONT] Reusing last_response results for generic follow-up")
                    else:
                        if is_summary or is_greeting:
                            # Greeting or summary: use broad fetch so we always have context (no "No relevant documents" for hi/hey)
                            k_greet = 5 if is_greeting else num_docs
                            print(f"\n🔎 [FRONT] {'Greeting' if is_greeting else 'Summary'} mode: search_all for docs={st.session_state.docs} → k={k_greet}")
                            resp = st.session_state.searcher.search_all(
                                filenames=st.session_state.docs,
                                k=k_greet,
                            )
                        else:
                            print(f"\n🔎 [FRONT] Searching for: '{retrieval_query}' → k={num_docs}, reranker={use_reranker}")
                            resp = st.session_state.searcher.search(
                                query=retrieval_query,
                                k=num_docs,
                                use_reranker=use_reranker
                            )
                    print(f"🔎 [FRONT] Search done. rerank_time={resp.get('rerank_time', 0):.3f}s")
                    print(f"🔎 [FRONT] Search response type: {type(resp)}, keys: {list(resp.keys()) if isinstance(resp, dict) else 'N/A'}")
                    raw_results = resp.get("results", []) if isinstance(resp, dict) else []
                    print(f"🔎 [FRONT] Raw results: {len(raw_results)}")
                    results = [r for r in raw_results if r.get('text', '').strip()]
                    print(f"🔎 [FRONT] Filtered results (non-empty text): {len(results)}")

                    # If this message has specific docs attached (first message after upload), use only those
                    current_target_docs = []
                    if st.session_state.get("messages"):
                        last_msg = st.session_state.messages[-1]
                        current_target_docs = last_msg.get("docs") or []
                    if current_target_docs:
                        targeted = [
                            r for r in results
                            if r.get("source", r.get("metadata", {}).get("filename", "")) in current_target_docs
                        ]
                        results = targeted  # strict: only attached docs for this message

                    metrics = {
                        "total_latency": resp.get("total_latency", 0),
                        "search_time": resp.get("search_time", 0),
                        "rerank_time": resp.get("rerank_time", 0),
                        "embedding_time": resp.get("embedding_time", 0),
                        "cached": resp.get("cached", False),
                        "num_results": len(results),
                    } if isinstance(resp, dict) else {}

                    st.session_state.metrics_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": user_input,
                        **metrics
                    })

                    # Strict: only chunks from docs uploaded in THIS chat — no cross-contamination
                    session_docs = set(st.session_state.get("docs", []) or [])
                    if session_docs:
                        session_filtered = [
                            r for r in results
                            if r.get("source", r.get("metadata", {}).get("filename", "")) in session_docs
                        ]
                        results = session_filtered

                    # After filtering: if no results, show friendly message for greetings instead of error
                    if not results:
                        if is_greeting and session_docs:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Hey! I've read your documents. What would you like to know?"
                            })
                            st.session_state.last_response = {**metrics, "results": [], "primary_source": None}
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "⚠️ No relevant documents found. Please upload documents related to your query."
                            })
                            st.session_state.last_response = {**metrics, "results": []}
                        st.rerun()

                    # Choose the single most relevant document: prefer recent doc when scores are close
                    doc_best_scores = {}
                    for r_res in results:
                        src_name = r_res.get("source", r_res.get("metadata", {}).get("filename", "unknown"))
                        try:
                            s_val = float(r_res.get("score", 0.0) or 0.0)
                        except Exception:
                            s_val = 0.0
                        if src_name not in doc_best_scores or s_val > doc_best_scores[src_name]:
                            doc_best_scores[src_name] = s_val
                    # Boost most recently uploaded doc so generic questions answer from "recent" doc
                    recent_docs = st.session_state.get("docs", []) or []
                    if recent_docs and len(recent_docs) > 1:
                        last_doc = recent_docs[-1]
                        if last_doc in doc_best_scores:
                            doc_best_scores[last_doc] = doc_best_scores[last_doc] * 1.15

                    primary_source = None
                    if doc_best_scores:
                        primary_source = max(doc_best_scores.items(), key=lambda x: x[1])[0]
                        results_for_answer = [
                            r for r in results
                            if r.get("source", r.get("metadata", {}).get("filename", "unknown")) == primary_source
                        ]
                    else:
                        results_for_answer = results

                    # Store primary_source in metrics so UI can show consistent sources
                    metrics["primary_source"] = primary_source

                    docs = [r.get('text', '') for r in results_for_answer]
                    # Use the ES hybrid score for guardrails/top_score instead of raw reranker score
                    score = float(
                        results_for_answer[0].get('score', results_for_answer[0].get('rerank_score', 0.0))
                    ) if results_for_answer else 0.0

                    # Relevance guardrail — now VERY relaxed.
                    # Only block:
                    # - clearly low-scoring matches
                    # - long, likely-off-topic queries
                    # Skip for: summary-ish queries, short queries, cached results.
                    RELEVANCE_THRESHOLD = 0.20
                    words_for_guard = query_lower.split()
                    if (score < RELEVANCE_THRESHOLD
                            and not is_summary
                            and len(words_for_guard) >= 6
                            and "summ" not in query_lower  # allow all summary-ish / vague summaries
                            and not resp.get("cached")):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ This information does not appear to be in the uploaded documents. "
                                       "Please ask something related to the content you've uploaded."
                        })
                        st.session_state.last_response = {**metrics, "results": results}
                        st.session_state["_clear_draft"] = True
                        st.rerun()

                    history = "\n".join([
                        f"User: {m['content']}" if m['role'] == 'user'
                        else f"Assistant: {m['content']}"
                        for m in st.session_state.messages[-4:]
                    ])

                if results:
                    with st.spinner("💬 Generating answer..."):
                        # For follow-ups like "anything more?", pass both the original topic
                        # and the follow-up text so the model continues the SAME thread.
                        gen_query = user_input
                        try:
                            if "is_followup" in locals() and is_followup:
                                gen_query = f"{retrieval_query}\n(Follow-up question: {user_input})"
                        except Exception:
                            pass

                        answer = st.session_state.generator.generate(
                            query=gen_query, docs=docs, history=history, top_score=score
                        )
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_response = {**metrics, "results": results}

                        # Har successful answer ke baad current conversation ko per-user history mein save karo
                        st.session_state.conversations[st.session_state.conversation_id] = {
                            "messages": st.session_state.messages.copy(),
                            "title": st.session_state.messages[0]['content'][:40] + "..." if st.session_state.messages else "New Chat",
                            "docs": list(st.session_state.get("docs", [])),
                            "doc_files": dict(st.session_state.get("doc_files", {})),
                        }
                        _save_user_history()

            except Exception as e:
                error = f"❌ Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error})

            st.session_state["_clear_draft"] = True
            st.rerun()