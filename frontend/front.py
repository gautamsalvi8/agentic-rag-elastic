import streamlit as st
import time
import sys
import os
import json
import base64
import PyPDF2
from io import BytesIO
from streamlit_oauth import OAuth2Component

# =========================================================
# PAGE CONFIG (MUST BE FIRST)
# =========================================================
st.set_page_config(
    page_title="RAG Brain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 BLACK + PINK GLASS THEME + ANIMATIONS
# =========================================================
st.markdown("""
<style>

/* background */
html, body, [data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0b0b0b,#151515);
    color:white;
}

/* glass cards */
section[data-testid="stSidebar"]{
    background: rgba(20,20,20,0.7);
    backdrop-filter: blur(12px);
}

/* buttons */
.stButton>button{
    background: linear-gradient(90deg,#ff0080,#ff4da6);
    border:none;
    border-radius:12px;
    color:white;
}

/* chat input floating */
.stChatFloatingInputContainer{
    bottom: 25px;
}

/* typing dots */
.typing span{
    height:8px;
    width:8px;
    margin:2px;
    background:#ff4da6;
    border-radius:50%;
    display:inline-block;
    animation:bounce 1.4s infinite ease-in-out both;
}
.typing span:nth-child(1){animation-delay:-.32s}
.typing span:nth-child(2){animation-delay:-.16s}

@keyframes bounce{
    0%,80%,100%{transform:scale(0)}
    40%{transform:scale(1)}
}

/* avatar */
.avatar{
    width:36px;
    height:36px;
    border-radius:50%;
    background:#ff4da6;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:bold;
}

/* hide footer */
footer{visibility:hidden}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 🧠 GOOGLE OAUTH (REAL)
# =========================================================
CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID"
CLIENT_SECRET = "YOUR_GOOGLE_CLIENT_SECRET"

oauth = OAuth2Component(
    CLIENT_ID,
    CLIENT_SECRET,
    "https://accounts.google.com/o/oauth2/auth",
    "https://oauth2.googleapis.com/token",
)

REDIRECT_URI = "http://localhost:8501"


# =========================================================
# USERS STORAGE
# =========================================================
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    return {}

def save_users(u):
    json.dump(u, open(USERS_FILE, "w"))


# =========================================================
# LOGIN PAGE
# =========================================================
def login_page():
    users = load_users()

    left, right = st.columns([1.3, 1])

    # ---------- LEFT (Mascot area) ----------
    with left:
        st.markdown("""
        <h1 style='color:#ff4da6;font-size:56px'>🧠 RAG Brain</h1>
        <h3 style='color:#aaa'>Chat with your PDFs like magic</h3>
        <br><br>
        <p style='color:#888;font-size:18px'>
        Upload → Ask → Understand instantly
        </p>
        """, unsafe_allow_html=True)

    # ---------- RIGHT (Auth card) ----------
    with right:

        tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign up"])

        # LOGIN
        with tab1:
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")

            if st.button("Login"):
                if email in users and users[email] == pwd:
                    st.session_state.user = email
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            # GOOGLE
            result = oauth.authorize_button(
                name="Continue with Google",
                redirect_uri=REDIRECT_URI,
                scope="openid email profile",
            )

            if result:
                st.session_state.user = "google_user"
                st.session_state.authenticated = True
                st.rerun()

        # SIGNUP
        with tab2:
            e = st.text_input("Email ", key="new")
            p = st.text_input("Password ", type="password", key="newp")

            if st.button("Create account"):
                users[e] = p
                save_users(users)
                st.success("Account created. Login now!")


# =========================================================
# AUTH CHECK
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
    st.stop()


# =========================================================
# BACKEND IMPORTS
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, "..", "backend")
sys.path.append(backend_path)

from bulk_ingest import BulkIngest
from hybrid_search import HybridSearch
from generator import generate_answer


# =========================================================
# SESSION STATE
# =========================================================
if "ingestor" not in st.session_state:
    st.session_state.ingestor = BulkIngest()
    st.session_state.searcher = HybridSearch()
    st.session_state.chats = {"New Chat": []}
    st.session_state.current_chat = "New Chat"
    st.session_state.docs_uploaded = False


history = st.session_state.chats[st.session_state.current_chat]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("### 👤 Profile")
    st.markdown(f"<div class='avatar'>{st.session_state.user[0].upper()}</div>", unsafe_allow_html=True)
    st.caption(st.session_state.user)

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()

    # Chats
    for chat in list(st.session_state.chats.keys()):
        c1, c2 = st.columns([4,1])

        if c1.button(chat):
            st.session_state.current_chat = chat
            st.rerun()

        if chat != "New Chat" and c2.button("❌", key=chat):
            del st.session_state.chats[chat]
            st.rerun()


# =========================================================
# HEADER
# =========================================================
st.title("🧠 RAG Brain")


# =========================================================
# FILE UPLOAD
# =========================================================
file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    with st.spinner("Indexing..."):
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = "".join([p.extract_text() or "" for p in reader.pages])
        st.session_state.ingestor.bulk_index_document(text, file.name)
        st.session_state.docs_uploaded = True


# =========================================================
# CHAT HISTORY
# =========================================================
for role, msg in history:
    with st.chat_message(role):
        st.write(msg)


# =========================================================
# CHAT INPUT
# =========================================================
query = st.chat_input("Ask about your document...", disabled=not st.session_state.docs_uploaded)

if query:
    history.append(("user", query))

    with st.chat_message("assistant"):
        dots = st.empty()
        dots.markdown("<div class='typing'><span></span><span></span><span></span></div>", unsafe_allow_html=True)

    results = st.session_state.searcher.search(query, k=5)

    answer = generate_answer(query, results, "")

    dots.empty()

    history.append(("assistant", answer))
    st.rerun()


# =========================================================
# PDF VIEWER
# =========================================================
if file:
    with st.expander("📄 Preview"):
        b64 = base64.b64encode(file.getvalue()).decode()
        st.markdown(
            f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='600'></iframe>",
            unsafe_allow_html=True,
        )
