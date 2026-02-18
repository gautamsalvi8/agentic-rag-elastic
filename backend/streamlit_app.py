import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Chat", layout="wide")

st.title("🤖 Agentic RAG Chat")

# session memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------- FILE UPLOAD ----------
st.sidebar.header("📂 Upload Knowledge")
uploaded = st.sidebar.file_uploader("Upload text file", type=["txt"])

if uploaded:
    res = requests.post(
        f"{API_URL}/upload",
        files={"file": uploaded}
    )
    st.sidebar.success("File uploaded successfully")


# -------- CHAT UI ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/chat",
                json={"query": query}
            ).json()

            answer = res["answer"]
            request_id = res["request_id"]

            st.markdown(answer)
            st.caption(f"🆔 Request ID: `{request_id}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
