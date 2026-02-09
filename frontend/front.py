import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Agentic RAG Chat", layout="wide")

st.title("🧠 Agentic RAG Chat")
st.caption("Elastic + Hybrid Search + Memory")


# =========================
# Input
# =========================
query = st.text_input("Ask a question:")


# =========================
# Send Button
# =========================
if st.button("Send") and query:

    with st.spinner("Thinking..."):

        try:
            response = requests.post(
                API_URL,
                json={"query": query}
            )

            data = response.json()

            answer = data.get("answer", "No answer returned")

            st.success("Response:")
            st.write(answer)

        except Exception as e:
            st.error(f"API Error: {e}")
