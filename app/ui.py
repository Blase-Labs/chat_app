# ui.py

# streamlit

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
    st.session_state.rows = 0
    st.session_state.file_sig = ""

st.set_page_config(page_title="Know your data.", page_icon="🗣️")
st.title("Load your dataset and ask questions.")

uploaded = st.file_uploader("Choose a CSV", type=["csv"])


def sig(file):
    return f"{file.name}:{file.size if hasattr(file, 'size') else len(file.getvalue())}"


if uploaded:
    cur_sig = sig(uploaded)
    if cur_sig != st.session_state.file_sig:
        st.session_state.indexed = False
        st.session_state.rows = 0
        st.session_state.file_sig = cur_sig

    if not st.session_state.indexed:
        if st.button("Build index"):
            with st.spinner("Indexing on server..."):
                r = requests.post(
                    f"{API_BASE}/ingest",
                    files={"csv": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=120,
                )
                if r.ok:
                    st.session_state.indexed = True
                    st.session_state.rows = r.json().get("rows", 0)
                    st.success(f"Indexed {st.session_state.rows} rows.")
                else:
                    st.error(f"Ingest failed: {r.status_code} {r.text}")
else:
    st.info("Upload a csv to get started.")

st.divider()
q = st.text_input("Ask a question about the data", disabled=not st.session_state.indexed)
ask_clicked = st.button("Ask", disabled=not st.session_state.indexed)

if ask_clicked and q.strip():
    with st.spinner("Thinking..."):
        r = requests.post(f"{API_BASE}/ask", json={"question": q}, timeout=45)
    if r.ok:
        data = r.json()
        st.markdown("### Answer")
        st.write(data["answer"])
        st.markdown("### Source")
        st.code("\n".join([f"row {i}" for i in data.get("sources", [])]) or "None")
    else:
        st.error(f"Ask failed: {r.status_code} {r.text}")
