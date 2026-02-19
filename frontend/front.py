import sys, os, importlib
import base64
import hashlib
import secrets
import tempfile

# Ensure project root is on sys.path so `backend.*` imports work both locally and on Streamlit Cloud.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load .env (project root + cwd) so local run pe REDIRECT_URI mil jaye
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv()  # cwd ka .env bhi
except Exception:
    pass

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

# Streamlit Cloud: secrets TOML is in st.secrets; backend uses os.getenv(). Copy so backend gets ELASTIC_* and USE_GROQ_API.
def _env_from_secrets():
    for key in ("ELASTIC_URL", "ELASTIC_API_KEY", "USE_GROQ_API", "GROQ_API_KEY"):
        if os.environ.get(key):
            continue
        try:
            val = st.secrets.get(key) or getattr(st.secrets, key, None)
            if val:
                os.environ[key] = str(val)
        except Exception:
            pass
_env_from_secrets()

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
    try:
        _sid = st.query_params.get("sid") if hasattr(st, "query_params") and hasattr(st.query_params, "get") else None
        if _sid:
            store = _load_session_store()
            store.pop(_sid, None)
            _save_session_store(store)
        try:
            if hasattr(st, "query_params") and hasattr(st.query_params, "clear"):
                st.query_params.clear()
        except Exception:
            pass
    except Exception:
        pass
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
                "_user_db_key", "groq_api_key", "hf_api_key", "generator", "model_loaded", "_history_loaded", "_sid_cookie_set"]:
        st.session_state.pop(_k, None)

st.set_page_config(
    page_title="ElasticNode AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

USER_DB = os.path.join(os.path.dirname(__file__), "users.json")
# Session store: sid -> session data. Streamlit Cloud pe repo read-only hota hai, isliye writable temp dir use karte hain.
SESSION_STORE_PATH = os.path.join(tempfile.gettempdir(), "elastic_session_store.json")

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


def _encrypt_api_key(key: str) -> str:
    """Lightweight reversible 'encryption' for API keys (base64 with prefix)."""
    k = (key or "").strip()
    if not k:
        return ""
    try:
        b64 = base64.b64encode(k.encode("utf-8")).decode("utf-8")
        return "enc:" + b64
    except Exception:
        return k


def _decrypt_api_key(stored: str) -> str:
    """Reverse of _encrypt_api_key; returns original value if not encrypted."""
    s = (stored or "").strip()
    if not s:
        return ""
    if s.startswith("enc:"):
        try:
            raw = s[4:]
            decoded = base64.b64decode(raw.encode("utf-8")).decode("utf-8")
            return decoded
        except Exception:
            return ""
    return s

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


def _email_user_key(email: str) -> str:
    """Key in users.json for email+Groq based signup users."""
    return (email or "").strip().lower()


def _email_user_exists(email: str) -> bool:
    users = _load_users()
    key = _email_user_key(email)
    return key in users


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
    stored = (users.get(key) or {}).get("groq_api_key") or ""
    return _decrypt_api_key(stored)


def _save_user_groq_key(api_key: str):
    """Save Groq API key for the current user."""
    db_key = _get_current_user_db_key()
    if not db_key:
        return False
    users = _load_users()
    if db_key not in users:
        users[db_key] = {}
    users[db_key]["groq_api_key"] = _encrypt_api_key(api_key or "")
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


def _load_session_store():
    """Load sid -> {groq_api_key, _user_db_key, username} from file + users.json (backup for Streamlit Cloud restart)."""
    store = {}
    try:
        if os.path.exists(SESSION_STORE_PATH):
            with open(SESSION_STORE_PATH, "r", encoding="utf-8") as f:
                store = json.load(f)
    except Exception:
        pass
    if not store:
        try:
            users = _load_users()
            store = users.get("sid_sessions") or {}
        except Exception:
            pass
    return store if isinstance(store, dict) else {}


def _save_session_store(store: dict):
    """Save session store to file and users.json so Streamlit Cloud restart pe bhi restore ho sake."""
    try:
        with open(SESSION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except Exception:
        pass
    try:
        users = _load_users()
        users["sid_sessions"] = store
        _save_users(users)
    except Exception:
        pass


def _restore_session_from_sid(sid: str) -> bool:
    """If sid is valid, restore session state and return True (Google or legacy Groq session)."""
    if not (sid or "").strip():
        return False
    store = _load_session_store()
    data = store.get((sid or "").strip())
    if not data:
        return False
    provider = data.get("auth_provider") or ("groq" if data.get("groq_api_key") else "google")
    st.session_state.authenticated = True
    st.session_state.auth_provider = provider
    st.session_state._user_db_key = data.get("_user_db_key", "")
    st.session_state.username = data.get("username", "User")
    if data.get("groq_api_key"):
        st.session_state.groq_api_key = data.get("groq_api_key", "")
    return True


def _get_google_auth_config():
    """Return auth config from secrets.toml (or Streamlit Cloud Secrets) or .env, or None if not configured."""
    def _get(obj, key, default=""):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default) or default
        val = getattr(obj, key, None)
        if val is None:
            try:
                val = obj[key]
            except (KeyError, TypeError, AttributeError):
                pass
        return (val or default) if val is not None else default

    # 1) Streamlit Cloud / secrets.toml — st.secrets.auth.client_id style (most reliable on Cloud)
    try:
        auth = getattr(st.secrets, "auth", None)
        if auth is None:
            auth = st.secrets.get("auth") if hasattr(st.secrets, "get") else None
        if auth is None:
            auth = st.secrets["auth"]
        if auth:
            cid = _get(auth, "client_id")
            csec = _get(auth, "client_secret")
            if cid and csec:
                redirect = _get(auth, "redirect_uri") or "http://localhost:8501/oauth2callback"
                # Local run: .env mein REDIRECT_URI=http://localhost:8501 set karo taaki Google yahi redirect kare
                env_redirect = os.environ.get("REDIRECT_URI", "").strip()
                if env_redirect:
                    if "/oauth2callback" not in env_redirect:
                        env_redirect = env_redirect.rstrip("/") + "/oauth2callback"
                    redirect = env_redirect
                meta = _get(auth, "server_metadata_url") or "https://accounts.google.com/.well-known/openid-configuration"
                return {
                    "client_id": str(cid).strip(),
                    "client_secret": str(csec).strip(),
                    "redirect_uri": str(redirect).strip(),
                    "server_metadata_url": str(meta).strip(),
                }
    except Exception:
        pass

    # 2) Fallback: .env (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI) — local only
    cid = os.environ.get("GOOGLE_CLIENT_ID")
    csec = os.environ.get("GOOGLE_CLIENT_SECRET")
    if cid and csec:
        redirect = os.environ.get("REDIRECT_URI", "http://localhost:8501/oauth2callback")
        if redirect and "/oauth2callback" not in redirect:
            redirect = redirect.rstrip("/") + "/oauth2callback"
        return {
            "client_id": cid,
            "client_secret": csec,
            "redirect_uri": redirect,
            "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
        }
    return None


def _google_oauth_login_url():
    """Return (authorization_url, state) for Google OAuth. State is stored in session for callback."""
    import requests
    config = _get_google_auth_config()
    if not config or not config.get("redirect_uri"):
        return None, None
    try:
        r = requests.get(config["server_metadata_url"], timeout=10)
        meta = r.json()
        auth_endpoint = meta.get("authorization_endpoint")
        if not auth_endpoint:
            return None, None
        state = secrets.token_urlsafe(24)
        st.session_state["_oauth_state"] = state
        scope = "openid email profile"
        url = (
            auth_endpoint
            + "?client_id=" + requests.utils.quote(config["client_id"])
            + "&redirect_uri=" + requests.utils.quote(config["redirect_uri"])
            + "&response_type=code"
            + "&scope=" + requests.utils.quote(scope)
            + "&state=" + state
            + "&access_type=offline&prompt=consent"
        )
        return url, state
    except Exception:
        return None, None


def _google_oauth_callback(code: str, state: str) -> dict:
    """Exchange code for token and return userinfo {email, name, sub}. On failure return {}."""
    import requests
    config = _get_google_auth_config()
    if not config:
        return {}
    saved_state = st.session_state.get("_oauth_state")
    if not saved_state or saved_state != state:
        return {}
    try:
        r = requests.get(config["server_metadata_url"], timeout=10)
        meta = r.json()
        token_endpoint = meta.get("token_endpoint")
        userinfo_endpoint = meta.get("userinfo_endpoint") or meta.get("userinfo_endpoint")
        if not token_endpoint:
            return {}
        # Exchange code for token
        tr = requests.post(
            token_endpoint,
            data={
                "code": code,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "redirect_uri": config["redirect_uri"],
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if tr.status_code != 200:
            return {}
        token = tr.json()
        access_token = token.get("access_token")
        if not access_token:
            return {}
        # Get userinfo
        ur = requests.get(
            userinfo_endpoint or "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": "Bearer " + access_token},
            timeout=10,
        )
        if ur.status_code != 200:
            return {}
        return ur.json()
    except Exception:
        return {}
    finally:
        st.session_state.pop("_oauth_state", None)


# ---------- Query params: OAuth callback, sid restore, then API keys from localStorage (restore_groq/restore_hf) ----------
try:
    _qp = st.query_params if hasattr(st, "query_params") and hasattr(st.query_params, "get") else None
    _sid = _qp.get("sid") if _qp else None
    _code = _qp.get("code") if _qp else None
    _state = _qp.get("state") if _qp else None
    _restore_groq = _qp.get("restore_groq") if _qp else None
    _restore_hf = _qp.get("restore_hf") if _qp else None
except Exception:
    _sid = _code = _state = _restore_groq = _restore_hf = None

# 1) OAuth callback: Google redirected back with code & state
if _code and _state and not st.session_state.authenticated:
    userinfo = _google_oauth_callback(_code, _state)
    if userinfo:
        email = (userinfo.get("email") or "").strip().lower()
        name = (userinfo.get("name") or userinfo.get("email") or "User").strip()
        sub = userinfo.get("sub", "")
        _register_google_user(email, name, sub)
        db_key = _google_user_key(email)
        st.session_state.authenticated = True
        st.session_state.auth_provider = "google"
        st.session_state.username = name or email or "User"
        st.session_state._user_db_key = db_key
        sid = secrets.token_urlsafe(16)
        store = _load_session_store()
        store[sid] = {
            "auth_provider": "google",
            "_user_db_key": db_key,
            "username": st.session_state.username,
        }
        _save_session_store(store)
        # Google se wapas aate waqt naya request hota hai, session state nahi milti. Isliye sid URL mein bhej ke next load pe store se restore karte hain.
        try:
            _sid_js = json.dumps(sid)
            _redirect_html = (
                '<script>(function(){'
                'var sid = ' + _sid_js + ';'
                'var origin = window.location.origin || "";'
                'var base = origin + "/";'
                'window.top.location.replace(base + "?sid=" + encodeURIComponent(sid));'
                '})();</script>'
            )
            components.html(_redirect_html, height=0)
            st.stop()
        except Exception:
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()
    else:
        st.session_state.auth_error = "Google sign-in failed. Try again."
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

# 2) Restore session from sid (refresh or OAuth redirect)
if _sid and not st.session_state.authenticated:
    if _restore_session_from_sid(_sid):
        try:
            st.query_params.pop("sid", None)
        except Exception:
            pass
        st.rerun()

# 3) Authenticated: restore API keys from localStorage (restore_groq / restore_hf in URL)
if st.session_state.authenticated and (_restore_groq or _restore_hf):
    try:
        if _restore_groq:
            _b64 = _restore_groq.replace(" ", "+")
            _pad = (4 - len(_b64) % 4) % 4
            _key_bytes = base64.urlsafe_b64decode(_b64 + ("=" * _pad))
            _key_str = _key_bytes.decode("utf-8", errors="replace").strip()
            if _key_str:
                st.session_state.groq_api_key = _key_str
                _save_user_groq_key(_key_str)
        if _restore_hf:
            _b64 = _restore_hf.replace(" ", "+")
            _pad = (4 - len(_b64) % 4) % 4
            _key_bytes = base64.urlsafe_b64decode(_b64 + ("=" * _pad))
            _key_str = _key_bytes.decode("utf-8", errors="replace").strip()
            if _key_str:
                st.session_state.hf_api_key = _key_str
    except Exception:
        pass
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

# 4) Not authenticated: cookie se sid restore (redirect to ?sid=...)
if not st.session_state.authenticated and not _sid and not _code:
    if st.session_state.get("_just_logged_out"):
        _clear_cookie = """<script>(function(){
            var c = "elastic_sid=; path=/; max-age=0; Secure";
            try { if (window.top.document) window.top.document.cookie = c; } catch(e) {}
            try { document.cookie = c; } catch(e) {}
            try { localStorage.removeItem('elastic_groq_key'); localStorage.removeItem('elastic_hf_key'); } catch(e) {}
        })();</script>"""
        components.html(_clear_cookie, height=0)
        st.session_state["_just_logged_out"] = False
    _cookie_restore_html = """
    <script>
    (function() {
        var doc = (window.top && window.top.document) ? window.top.document : document;
        var c = doc.cookie || "";
        if (window.location.search.includes("sid=") || window.location.search.includes("code=")) return;
        var m = c.match(/elastic_sid=([^;]+)/);
        if (m) {
            var q = window.location.search ? window.location.search + "&" : "?";
            window.top.location.replace(window.location.pathname + q + "sid=" + encodeURIComponent(m[1].trim()));
        }
    })();
    </script>
    """
    components.html(_cookie_restore_html, height=0)

# ---------- Login only via Groq API key: key = auth, site detects account via key ----------
if not st.session_state.authenticated:
    auth_config = _get_google_auth_config()
    if not auth_config:
        st.warning(
            "Google OAuth is not configured. "
            "**Local:** Add `[auth]` in `.streamlit/secrets.toml` or `.env`. "
            "**Streamlit Cloud:** App → Settings → Secrets."
        )
        st.caption("Google Cloud Console → APIs & Services → Credentials.")
        st.stop()
    auth_url, _ = _google_oauth_login_url()
    if not auth_url:
        st.error("Could not generate Google login URL. Check redirect_uri in secrets.")
        st.stop()

    # redirect_uri_mismatch fix: Google Console mein yehi exact URI add karo (Copy karke)
    _auth_cfg = _get_google_auth_config()
    _ru = (_auth_cfg or {}).get("redirect_uri", "")
    if _ru:
        with st.expander("🔧 redirect_uri_mismatch? Is URI ko Google Console → Credentials → Authorized redirect URIs mein add karo"):
            st.code(_ru, language=None)
            st.caption("Copy the above URL and add it in Google Cloud Console if you see Error 400: redirect_uri_mismatch")

    _auth_error = st.session_state.get("auth_error")
    if _auth_error:
        st.session_state.auth_error = None

    _auth_static = os.path.join(os.path.dirname(__file__), "static")
    _logo_paths = [
        os.path.join(_auth_static, "logo.png"),
        os.path.join(_auth_static, "elasticnode-logo.png"),
    ]
    _logo_html = ""
    for _lp in _logo_paths:
        if os.path.exists(_lp):
            try:
                with open(_lp, "rb") as _f:
                    _b64 = base64.b64encode(_f.read()).decode()
                _logo_html = '<img src="data:image/png;base64,' + _b64 + '" alt="Logo" class="auth-split-logo" />'
                break
            except Exception:
                pass
    if not _logo_html:
        _logo_html = '<div class="auth-split-logo-placeholder"><span>Logo</span></div>'

    _illustration_src = ""
    _illus_candidates = [
        (os.path.join(PROJECT_ROOT, "assets", "chat_bot.gif"), "image/gif"),
        (os.path.join(_auth_static, "auth-illustration.png"), "image/png"),
        (os.path.join(_auth_static, "auth-right-bg.png"), "image/png"),
    ]
    for _ip, _mime in _illus_candidates:
        if os.path.exists(_ip):
            try:
                with open(_ip, "rb") as _f:
                    _b64 = base64.b64encode(_f.read()).decode()
                _illustration_src = "data:" + _mime + ";base64," + _b64
                break
            except Exception:
                pass

    _GOOGLE_LOGO_URL = "https://www.gstatic.com/images/branding/googlelogo/svg/googlelogo_clr_74x24px.svg"
    _auth_url_safe = auth_url.replace("'", "&#39;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    _hash = "#"

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    #MainMenu, footer, header { visibility: hidden !important; }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    .stDeployButton { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    html, body, [data-testid="stAppViewContainer"], .main { background: #FFFFFF !important; }
    html, body, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main { padding: 0 !important; height: 100vh !important; overflow: hidden !important; }
    /* Full page: pure white */
    .block-container { padding: 1.5rem !important; width: 100% !important; min-height: 100vh !important; display: flex !important; align-items: center !important; justify-content: center !important; background: #FFFFFF !important; }
    /* Outer container: white, no tint, very subtle structure only */
    .main [data-testid="stHorizontalBlock"] { width: 88% !important; max-width: 1200px !important; overflow: hidden !important; display: flex !important; flex-direction: row !important; flex-shrink: 0 !important; background: #FFFFFF !important; box-shadow: none !important; border: none !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child { flex: 0 0 320px !important; width: 320px !important; max-width: 320px !important; min-width: 0 !important; background: #FFFFFF !important; overflow: hidden !important; border: none !important; box-shadow: none !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child { flex: 1 1 auto !important; min-width: 0 !important; background: #FFFFFF !important; }
    /* Left panel: wrapper for vertical centering */
    .left-panel { display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; width: 100% !important; min-height: 100% !important; padding: 1rem !important; box-sizing: border-box !important; margin-top: 3rem !important; }
    .left-form-wrapper { width: 320px !important; max-width: 320px !important; margin: 0 auto 12px !important; display: flex !important; flex-direction: column !important; gap: 8px !important; box-sizing: border-box !important; }
    /* Left column: force 320px so Streamlit overrides nahi aaye — .main se specificity high */
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child > div { width: 320px !important; max-width: 320px !important; min-width: 0 !important; margin-left: auto !important; margin-right: auto !important; margin-bottom: 16px !important; box-sizing: border-box !important; overflow: hidden !important; border: none !important; box-shadow: none !important; background: transparent !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child > div > div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child > * { width: 320px !important; max-width: 320px !important; min-width: 0 !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stVerticalBlock"] { width: 320px !important; max-width: 320px !important; min-width: 0 !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] { width: 320px !important; max-width: 320px !important; min-width: 0 !important; padding: 0 !important; overflow: hidden !important; box-sizing: border-box !important; border: none !important; box-shadow: none !important; background: transparent !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] > div { width: 320px !important; max-width: 320px !important; min-width: 0 !important; padding: 0 !important; margin: 8px 0 0 0 !important; box-sizing: border-box !important; border: none !important; box-shadow: none !important; background: transparent !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] [data-testid="stHorizontalBlock"] { width: 320px !important; max-width: 320px !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stTextInput"], .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stTextInput"] > div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child input { width: 100% !important; max-width: 100% !important; min-width: 0 !important; height: 44px !important; border-radius: 10px !important; box-sizing: border-box !important; border: 1px solid #d1d5db !important; background: #f8fafc !important; transition: border-color 0.2s, box-shadow 0.2s !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] input { width: 100% !important; max-width: 100% !important; min-width: 0 !important; height: 44px !important; border-radius: 10px !important; box-sizing: border-box !important; border: 1px solid #d1d5db !important; background: #f8fafc !important; transition: border-color 0.2s, box-shadow 0.2s !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] .stButton { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button { width: 100% !important; max-width: 100% !important; min-width: 0 !important; height: 44px !important; border-radius: 10px !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child [data-testid="stForm"] .stButton > button { width: 100% !important; max-width: 100% !important; min-width: 0 !important; height: 44px !important; border-radius: 10px !important; box-sizing: border-box !important; background: #1f4ed8 !important; color: #ffffff !important; border: none !important; box-shadow: 0 2px 8px rgba(31,78,216,0.25) !important; }
    .left-form-wrapper input, .left-form-wrapper button { width: 100% !important; max-width: 100% !important; height: 44px !important; border-radius: 10px !important; box-sizing: border-box !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child { padding: 0 !important; display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; height: 100% !important; min-height: 400px !important; }
    .auth-split-brand { display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; gap: 0.5rem; margin: 0 0 4px 0; }
    .auth-split-logo { max-height: 56px; width: auto; }
    .auth-split-logo-placeholder { width: 32px; height: 28px; background: #f8fafc; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; font-weight: 600; color: #94a3b8; }
    .auth-split-brand-name { font-size: 1.05rem; font-weight: 700; color: #1a1a1a; letter-spacing: -0.02em; }
    .auth-split-title { font-size: 1.25rem; font-weight: 700; color: #1a1a1a; margin: 0; letter-spacing: -0.02em; }
    .auth-split-btn { display: inline-flex !important; align-items: center !important; justify-content: center !important; gap: 0.5rem !important; width: 100% !important; max-width: 100% !important; height: 44px !important; padding: 0 1rem !important; font-size: 14px !important; font-weight: 500 !important; border-radius: 10px !important; text-decoration: none !important; border: 1px solid #e2e8f0 !important; font-family: 'Inter', sans-serif !important; cursor: pointer !important; transition: box-shadow 0.2s !important; background: #fff !important; color: #334155 !important; margin: 0 !important; box-sizing: border-box !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
    .auth-split-btn:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .auth-split-btn img { width: 18px; height: 18px; flex-shrink: 0; order: -1; }
    .auth-split-divider { display: flex; align-items: center; margin: 0; color: #94a3b8; font-size: 14px; }
    .auth-split-divider::before, .auth-split-divider::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }
    .auth-split-divider span { padding: 0 0.5rem; }
    /* Right panel: pure white, image blends in — no box or tint (unchanged) */
    [data-testid="column"]:last-child { padding: 2rem 2.5rem !important; display: flex !important; align-items: center !important; justify-content: center !important; min-height: 400px !important; }
    .auth-split-right-panel { width: 100%; height: 100%; display: flex !important; align-items: center !important; justify-content: center !important; background: transparent !important; }
    .auth-split-illus-wrap { width: 90% !important; max-width: 100%; display: flex; align-items: center; justify-content: center; padding: 1rem; }
    .auth-split-illus-wrap img { width: 100%; height: auto; object-fit: contain; display: block; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child input { padding: 0 1rem !important; font-size: 14px !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child input:focus { border-color: #1f4ed8 !important; box-shadow: 0 0 0 3px rgba(31,78,216,0.15) !important; outline: none !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child svg { color: #111827 !important; opacity: 0.9 !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button { padding: 0 1rem !important; font-size: 14px !important; font-weight: 600 !important; border: none !important; border-radius: 10px !important; }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button:hover { transform: translateY(-1px); }
    .auth-split-forgot { color: #64748b; text-decoration: none; font-size: 13px; transition: color 0.2s; }
    .auth-split-forgot:hover { color: #111827; }
    /* Auth form primary button: robot blue, rounded, no stroke */
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child form .stButton > button {
        background: #1f4ed8 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(31,78,216,0.25) !important;
        transform: translateY(0);
    }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child form .stButton > button:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(31,78,216,0.35) !important;
        transform: translateY(-1px);
    }
    .main [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child form svg {
        color: #111827 !important;
        opacity: 0.9 !important;
    }
    .auth-split-signup { margin: 0; font-size: 13px; color: #64748b; text-align: center; }
    .auth-split-signup a { color: #1f4ed8; text-decoration: none; font-weight: 500; }
    .auth-split-signup a:hover { text-decoration: underline; }
    .auth-split-error { background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; font-size: 0.8rem; padding: 0.5rem 0.75rem; border-radius: 10px; margin-top: 0.5rem; }
    /* Custom email/password card (login container) */
    .auth-email-card { background: #f3f4f6; border-radius: 16px; padding: 1.1rem 1.25rem 1.25rem; border: 1px solid rgba(148,163,184,0.35); box-shadow: 0 4px 14px rgba(15,23,42,0.06); display: flex; flex-direction: column; gap: 10px; }
    .auth-email-input { width: 100%; height: 40px; border-radius: 10px; border: 1px solid #d1d5db; background: #f9fafb; padding: 0 1rem 0 2.25rem; font-size: 14px; box-sizing: border-box; transition: border-color 0.2s, box-shadow 0.2s, background 0.2s; }
    .auth-email-input-email { background-image: url("data:image/svg+xml,%3Csvg width='16' height='16' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='3' y='5' width='18' height='14' rx='2' ry='2' stroke='%2394A3B8' stroke-width='1.5'/%3E%3Cpath d='M4 7L11.2 11.6C11.7 11.9 12.3 11.9 12.8 11.6L20 7' stroke='%2394A3B8' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 0.75rem 50%; background-size: 16px; }
    .auth-email-input-password { background-image: url("data:image/svg+xml,%3Csvg width='16' height='16' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='10' width='14' height='9' rx='2' ry='2' stroke='%2394A3B8' stroke-width='1.5'/%3E%3Cpath d='M9 10V8C9 5.8 10.3 4 12.5 4C14.7 4 16 5.8 16 8V10' stroke='%2394A3B8' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 0.75rem 50%; background-size: 16px; }
    .auth-email-row { display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: #4b5563; margin-top: 4px; }
    .auth-email-keep { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #4b5563; }
    .auth-email-keep input[type="checkbox"] {
        -webkit-appearance: none;
        appearance: none;
        width: 12px !important;
        height: 12px !important;
        border-radius: 3px;
        border: 1px solid #111827;
        background: #ffffff;
        display: inline-block;
        position: relative;
        cursor: pointer;
        margin: 0;
        padding: 0;
    }
    .auth-email-keep input[type="checkbox"]:checked::after {
        content: "";
        position: absolute;
        left: 2px;
        top: 1px;
        width: 7px;
        height: 4px;
        border-left: 1.5px solid #111827;
        border-bottom: 1.5px solid #111827;
        transform: rotate(-45deg);
    }
    .auth-email-login-btn { width: 100%; margin-top: 10px; height: 40px; border-radius: 999px; border: none; background: #111827; color: #ffffff; font-size: 14px; font-weight: 600; cursor: pointer; box-shadow: 0 2px 8px rgba(15,23,42,0.25); transition: box-shadow 0.2s, transform 0.2s, background 0.2s; }
    .auth-email-login-btn:hover { background: #020617; box-shadow: 0 4px 12px rgba(15,23,42,0.35); transform: translateY(-1px); }
    .auth-email-or { text-align: center; font-size: 13px; color: #94a3b8; margin: 4px 0 4px 0; }
    .auth-loading-overlay { display: none; position: fixed; inset: 0; background: rgba(255,255,255,0.95); align-items: center; justify-content: center; z-index: 9999; flex-direction: column; gap: 0.75rem; }
    .auth-loading-overlay.show { display: flex !important; }
    .auth-spinner { width: 38px; height: 38px; border: 3px solid #e2e8f0; border-top-color: #1f4ed8; border-radius: 50%; animation: auth-spin 0.7s linear infinite; }
    @keyframes auth-spin { to { transform: rotate(360deg); } }
    .auth-loading-text { font-size: 0.85rem; color: #64748b; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

    _error_html = ('<div class="auth-split-error">' + _auth_error.replace("<", "&lt;").replace(">", "&gt;") + '</div>') if _auth_error else ""

    # Decide whether to show Login or Signup view
    _view_param = None
    try:
        if hasattr(st, "query_params") and hasattr(st.query_params, "get"):
            _view_param = st.query_params.get("view")
    except Exception:
        _view_param = None
    _auth_view = _view_param or ("signup" if st.session_state.get("show_signup") else "login")
    if isinstance(_auth_view, (list, tuple)):
        _auth_view = _auth_view[0] if _auth_view else "login"
    _auth_view = "signup" if str(_auth_view).lower() == "signup" else "login"
    st.session_state.show_signup = _auth_view == "signup"

    _login_clicked = False
    _user = _pass = ""

    col_left, col_right = st.columns(2)

    with col_left:
        if _auth_view == "login":
            # Login view: logo + custom email/password card + Google login
            st.markdown(
                '<div class="left-panel">'
                '<div id="auth-form-ref" class="left-form-wrapper">'
                '<div class="auth-split-brand">' + _logo_html + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="left-form-wrapper">'
                '<div class="auth-email-card">'
                '<input type="email" class="auth-email-input auth-email-input-email" name="elastic_login_email" placeholder="Email address" autocomplete="off" />'
                '<input type="password" class="auth-email-input auth-email-input-password" name="elastic_login_password" placeholder="Password" autocomplete="off" />'
                '<div class="auth-email-row">'
                '<label class="auth-email-keep"><input type="checkbox" /> Remember me</label>'
                '<a href="' + _hash + '" class="auth-split-forgot" style="font-size:12px;color:#6b7280;">Forgot password?</a>'
                '</div>'
                '<button type="button" class="auth-email-login-btn">Login</button>'
                '</div>'
                '<div class="auth-email-or">or</div>'
                '<a href="' + _auth_url_safe + '" id="auth-google-link" class="auth-split-btn">'
                '<img src="' + _GOOGLE_LOGO_URL + '" alt="" />Continue with Google</a>'
                '</div>'
                + _error_html
                + '<p class="auth-split-signup">Don\'t have an account? <a href="?view=signup">Sign Up</a></p>'
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            # New signup page: email + Groq API key + Terms
            st.markdown(
                '<div class="left-panel">'
                '<div id="auth-form-ref" class="left-form-wrapper">'
                '<div class="auth-split-brand">' + _logo_html + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.form("auth_signup_form", clear_on_submit=False):
                _signup_email = st.text_input(
                    "Email Address",
                    placeholder="you@example.com",
                    key="signup_email",
                    label_visibility="collapsed",
                )
                _signup_key = st.text_input(
                    "Groq API Key",
                    placeholder="gsk_...",
                    type="password",
                    key="signup_groq_key",
                    label_visibility="collapsed",
                    help="Get your free API key at console.groq.com",
                )
                _signup_terms = st.checkbox(
                    "I agree to Terms of Service and Privacy Policy",
                    key="signup_terms",
                )
                _signup_clicked = st.form_submit_button("Create Account")

            # Real-time inline validation hints (simple)
            _se = (_signup_email or "").strip()
            _sk = (_signup_key or "").strip()
            _email_ok = bool(_se) and "@" in _se and "." in _se.split("@")[-1]
            _key_format_ok = _sk.startswith("gsk_") and len(_sk) > 30

            if _se:
                st.caption("✅ Valid email" if _email_ok else "❌ Please enter a valid email address")
            if _sk:
                if _key_format_ok:
                    st.caption("✅ Key format looks correct. Will verify with Groq on submit.")
                else:
                    st.caption("❌ Invalid format. Groq keys start with 'gsk_' and should be long.")

            _signup_error_msg = ""

            if _signup_clicked:
                # Step 1: email checks
                if not _se:
                    _signup_error_msg = "Email required"
                elif not _email_ok:
                    _signup_error_msg = "Please enter a valid email address"
                # Step 2: API key checks
                elif not _sk:
                    _signup_error_msg = "API key required"
                elif not _sk.startswith("gsk_"):
                    _signup_error_msg = "Invalid format. Groq keys start with 'gsk_'."
                elif len(_sk) <= 30:
                    _signup_error_msg = "API key too short"
                # Step 3: Terms
                elif not _signup_terms:
                    _signup_error_msg = "You must agree to Terms of Service"
                # Step 4: Email already exists?
                elif _email_user_exists(_se):
                    _signup_error_msg = "Account exists. Sign in instead?"
                else:
                    # Test key with Groq
                    with st.spinner("Verifying Groq API key…"):
                        _ok, _err = _validate_groq_api_key(_sk)
                    if not _ok:
                        _signup_error_msg = _err or "Invalid API key. Please check and try again."
                    else:
                        # Save user to users.json
                        users = _load_users()
                        db_key = _email_user_key(_se)
                        username = (_se.split("@")[0] or "user").strip()
                        users[db_key] = {
                            "provider": "groq_email",
                            "email": _se,
                            "username": username,
                            "groq_api_key": _encrypt_api_key(_sk),
                            "created_at": datetime.now().isoformat(),
                        }
                        _save_users(users)

                        # Auto-login user
                        st.session_state.authenticated = True
                        st.session_state.auth_provider = "groq"
                        st.session_state.username = username
                        st.session_state._user_db_key = db_key
                        st.session_state.groq_api_key = _sk

                        # Save Groq key for auto-restore (localStorage)
                        _b64 = base64.urlsafe_b64encode(_sk.encode()).decode().rstrip("=")
                        _safe = _b64.replace("\\", "\\\\").replace('"', '\\"')[:256]
                        components.html(
                            f'<script>try{{var b="{_safe}".replace(/-/g,"+").replace(/_/g,"/"); while(b.length%4) b+="="; var k=decodeURIComponent(escape(atob(b))); localStorage.setItem("elastic_groq_key",k);}}catch(e){{}}</script>',
                            height=0,
                        )

                        st.success("Account created successfully! Redirecting to your dashboard…")
                        st.balloons()
                        time.sleep(1.8)
                        st.rerun()

            if _signup_error_msg:
                st.error(_signup_error_msg)
            else:
                st.markdown(
                    '<p class="auth-split-signup">Already have an account? <a href="?view=login">Sign In</a></p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with col_right:
        _right_content = (
            '<div class="auth-split-right-panel"><div class="auth-split-illus-wrap">'
            '<img src="' + _illustration_src + '" alt="Illustration" class="auth-split-illus" />'
            '</div></div>'
        ) if _illustration_src else (
            '<div class="auth-split-right-panel"><div class="auth-split-illus-wrap">'
            '<div class="auth-split-logo-placeholder" style="width:200px;height:200px;font-size:1rem;">Illustration</div>'
            '</div></div>'
        )
        st.markdown(_right_content, unsafe_allow_html=True)

    if _auth_view == "login" and _login_clicked:
        if not (_user or "").strip() or not (_pass or "").strip():
            st.session_state.auth_error = "Please enter email and password."
            st.rerun()
        else:
            st.session_state.auth_error = None
            st.session_state._auth_login_loading = True

    st.markdown(
        '<div id="auth-loading-overlay" class="auth-loading-overlay">'
        '<div class="auth-spinner"></div>'
        '<span class="auth-loading-text">Signing in…</span>'
        '</div>'
        '<script>(function(){'
        'var g=document.getElementById("auth-google-link"); if(g) g.addEventListener("click",function(){ document.getElementById("auth-loading-overlay").classList.add("show"); });'
        'var f=document.querySelector("form"); if(f) f.addEventListener("submit",function(){ document.getElementById("auth-loading-overlay").classList.add("show"); });'
        'function authResize(){ var ref=document.getElementById("auth-form-ref"); if(!ref) return; var w=320; ref.style.setProperty("width",w+"px","important"); ref.style.setProperty("max-width",w+"px","important"); var col=ref.closest("[data-testid=column]"); if(!col) return; col.style.setProperty("width",w+"px","important"); col.style.setProperty("max-width",w+"px","important"); for(var i=0;i<col.children.length;i++){ var ch=col.children[i]; ch.style.setProperty("width",w+"px","important"); ch.style.setProperty("max-width",w+"px","important"); } var form=col.querySelector("form")||col.querySelector("[data-testid=stForm]"); if(form){ form.style.setProperty("width",w+"px","important"); form.style.setProperty("max-width",w+"px","important"); var p=form.parentElement; while(p&&p!==col){ p.style.setProperty("width",w+"px","important"); p.style.setProperty("max-width",w+"px","important"); p=p.parentElement; } } col.querySelectorAll("input").forEach(function(inp){ inp.style.setProperty("width","100%","important"); inp.style.setProperty("max-width","100%","important"); inp.style.setProperty("box-sizing","border-box","important"); }); col.querySelectorAll(".stButton").forEach(function(btn){ btn.style.setProperty("width",w+"px","important"); btn.style.setProperty("max-width",w+"px","important"); if(btn.firstElementChild){ btn.firstElementChild.style.setProperty("width","100%","important"); btn.firstElementChild.style.setProperty("max-width","100%","important"); } }); }'
        'authResize(); setTimeout(authResize,100); setTimeout(authResize,300); setTimeout(authResize,600); setTimeout(authResize,1200); if(document.readyState!=="complete") window.addEventListener("load",function(){ authResize(); setTimeout(authResize,200); }); var col=document.querySelector("[data-testid=stHorizontalBlock] [data-testid=column]:first-child"); if(col){ var obs=new MutationObserver(function(){ authResize(); }); obs.observe(col,{childList:true,subtree:true}); }'
        '})();</script>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("_auth_login_loading"):
        st.session_state._auth_login_loading = False
        st.markdown('<script>document.getElementById("auth-loading-overlay").classList.add("show");</script>', unsafe_allow_html=True)

    st.stop()

# Login ke baad sid cookie set (top window + Secure): site band/reopen pe bhi session restore
try:
    _qp = st.query_params if hasattr(st, "query_params") and hasattr(st.query_params, "get") else None
    _qsid = _qp.get("sid") if _qp else None
    if st.session_state.get("authenticated") and _qsid and not st.session_state.get("_sid_cookie_set"):
        _safe_sid = _qsid.replace("\\", "\\\\").replace('"', '\\"').replace(";", "")[:64]
        _set_cookie = f'''<script>(function(){{
            var c = "elastic_sid={_safe_sid}; path=/; max-age=7776000; SameSite=Lax; Secure";
            try {{ if (window.top.document) window.top.document.cookie = c; }} catch(e) {{}}
            try {{ document.cookie = c; }} catch(e) {{}}
        }})();</script>'''
        components.html(_set_cookie, height=0)
        st.session_state["_sid_cookie_set"] = True
except Exception:
    pass

# Ensure Groq key in session (from saved user, env, or restore_groq URL). If missing, try restore from localStorage.
if "groq_api_key" not in st.session_state or not (st.session_state.get("groq_api_key") or "").strip():
    st.session_state.groq_api_key = _get_user_groq_key() or os.getenv("GROQ_API_KEY", "")
# Authenticated but no keys yet: inject script to load from localStorage and redirect (restore_groq/restore_hf)
if st.session_state.get("authenticated") and not (_restore_groq or _restore_hf):
    _need_groq = not (st.session_state.get("groq_api_key") or "").strip()
    if _need_groq:
        _ls_restore = """
        <script>
        (function() {
            if (window.location.search.includes('restore_groq=')) return;
            var g = localStorage.getItem('elastic_groq_key');
            var h = localStorage.getItem('elastic_hf_key');
            if (!g && !h) return;
            var q = window.location.search ? window.location.search + '&' : '?';
            if (g) q += 'restore_groq=' + encodeURIComponent(g);
            if (h) { if (g) q += '&'; q += 'restore_hf=' + encodeURIComponent(h); }
            window.top.location.replace(window.location.pathname + q);
        })();
        </script>
        """
        components.html(_ls_restore, height=0)

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

# SETTINGS VIEW — API keys (saved to session + localStorage; ✏️ edit, clear on Logout)
elif st.session_state.current_view == "settings":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("<h2 class='dashboard-title' style='text-align:center;'>Settings</h2>", unsafe_allow_html=True)
    st.markdown("Enter your API keys below. They are saved in the browser (localStorage) and auto-load on next visit until you Logout or clear them.")
    with st.form("settings_api_keys"):
        current_groq = st.session_state.get("groq_api_key") or ""
        current_hf = st.session_state.get("hf_api_key") or ""
        new_groq = st.text_input("✏️ Groq API key", value=current_groq, type="password", placeholder="gsk_...", key="settings_groq_key_input", help="Used for AI answers. Get key at console.groq.com/keys")
        new_hf = st.text_input("✏️ HuggingFace API key (optional)", value=current_hf, type="password", placeholder="hf_...", key="settings_hf_key_input", help="Optional, for some models")
        st.caption("Get Groq key: [console.groq.com/keys](https://console.groq.com/keys)")
        submitted = st.form_submit_button("Save keys")
        if submitted:
            groq_ok = (new_groq or "").strip()
            hf_ok = (new_hf or "").strip()
            groq_valid = False
            if groq_ok:
                _ok, _err = _validate_groq_api_key(groq_ok)
                if not _ok:
                    st.error(_err or "Invalid Groq key.")
                else:
                    groq_valid = True
                    st.session_state.groq_api_key = groq_ok
                    _save_user_groq_key(groq_ok)
                    _b64 = base64.urlsafe_b64encode(groq_ok.encode()).decode().rstrip("=")
                    _safe = _b64.replace("\\", "\\\\").replace('"', '\\"')[:256]
                    components.html(f'<script>try{{var b="{_safe}".replace(/-/g,"+").replace(/_/g,"/"); while(b.length%4) b+="="; var k=decodeURIComponent(escape(atob(b))); localStorage.setItem("elastic_groq_key",k);}}catch(e){{}}</script>', height=0)
                    st.session_state.generator = None
                    st.session_state.model_loaded = False
                    try: load_generator.clear()
                    except Exception: pass
            if hf_ok:
                st.session_state.hf_api_key = hf_ok
                _b64 = base64.urlsafe_b64encode(hf_ok.encode()).decode().rstrip("=")
                _safe = _b64.replace("\\", "\\\\").replace('"', '\\"')[:256]
                components.html(f'<script>try{{var b="{_safe}".replace(/-/g,"+").replace(/_/g,"/"); while(b.length%4) b+="="; var k=decodeURIComponent(escape(atob(b))); localStorage.setItem("elastic_hf_key",k);}}catch(e){{}}</script>', height=0)
            if groq_valid or hf_ok:
                st.success("Keys saved. Stored in browser (localStorage); will auto-load on next visit.")
            st.rerun()
    if st.button("Clear API keys (session + localStorage)", key="settings_clear_keys", type="secondary"):
        st.session_state.groq_api_key = ""
        st.session_state.hf_api_key = ""
        st.session_state.generator = None
        st.session_state.model_loaded = False
        try: load_generator.clear()
        except Exception: pass
        components.html('<script>try{localStorage.removeItem("elastic_groq_key");localStorage.removeItem("elastic_hf_key");}catch(e){}</script>', height=0)
        st.success("Keys cleared. Add new keys above or they will be cleared on Logout.")
        st.rerun()
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