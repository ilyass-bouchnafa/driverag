"""
═══════════════════════════════════════════════════════════════════════════════
  DriveRAG — Professional Chat Interface
  Simple, accessible AI chat with document search and RAG
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import streamlit as st

root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))

from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.ingestion.sync_manager import start_auto_sync, get_auto_sync_status
from src.retrieval.vectorstore import add_chunks_to_store, get_indexed_files
from src.generation.llm_chain import ask, clear_memory
from src.config import GOOGLE_DRIVE_FOLDER_ID

import uuid
import time
from datetime import datetime
from typing import Optional, Dict, List


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "langsmith_thread_id": str(uuid.uuid4()),
        "conversation_id": f"DriveRAG_{int(time.time())}",
        "messages": [],
        "mode": "rag",
        "auto_sync_started": False,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    if not st.session_state.auto_sync_started:
        start_auto_sync(interval_seconds=300)
        st.session_state.auto_sync_started = True


init_session_state()

st.set_page_config(
    page_title="DriveRAG",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


# ══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL CLEAN CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #ffffff;
    --bg-light: #f8f9fa;
    --bg-lighter: #e9ecef;
    --border: #dee2e6;
    --text: #212529;
    --text-light: #6c757d;
    --text-lighter: #adb5bd;
    --primary: #0066cc;
    --primary-dark: #0052a3;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.main { background: var(--bg) !important; }

.main .block-container {
    max-width: 900px !important;
    padding: 24px 20px 550px 20px !important;
    margin: 0 auto !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* Chat Container */
.chat-wrapper {
    margin-bottom: 28px;
}

.messages-area {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.message-row {
    display: flex;
    gap: 12px;
    animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.message-row.user {
    justify-content: flex-end;
}

.msg-bubble {
    max-width: 72%;
    padding: 12px 16px;
    word-wrap: break-word;
    line-height: 1.5;
    font-size: 15px;
    border-radius: 12px;
}

.message-row.user .msg-bubble {
    background: var(--primary);
    color: white;
    border-radius: 12px 2px 12px 12px;
}

.message-row.assistant .msg-bubble {
    background: var(--bg-light);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 2px 12px 12px 12px;
}

.msg-time {
    font-size: 11px;
    color: var(--text-lighter);
    margin-top: 6px;
}

.msg-meta {
    font-size: 12px;
    color: var(--text-light);
    margin-bottom: 6px;
}

/* Welcome Screen */
.welcome-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 450px;
    text-align: center;
    gap: 12px;
}

.welcome-title {
    font-size: 32px;
    font-weight: 600;
    color: var(--text);
    margin: 0;
}

.welcome-desc {
    font-size: 15px;
    color: var(--text-light);
    margin: 0;
    max-width: 500px;
}

/* Files Display */
.files-bar {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 13px;
    color: var(--text-light);
}

.files-count {
    font-weight: 600;
    color: var(--text);
}

/* Input Section */
.input-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, rgba(255,255,255,1) 0%, rgba(255,255,255,0.95) 90%);
    border-top: 1px solid var(--border);
    padding: 20px 0;
    z-index: 1000;
}

.input-section {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 20px;
}

.input-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

.add-btn, .sync-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.2s;
}

.add-btn:hover, .sync-btn:hover {
    background: var(--primary-dark);
}

.input-box {
    flex: 1;
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.2s;
}

.input-box:focus-within {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(0,102,204,0.08);
}

.input-box input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 15px;
    font-family: 'Inter', sans-serif;
    outline: none;
}

.input-box input::placeholder {
    color: var(--text-lighter);
}

.mode-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-lighter);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 4px 8px;
    background: var(--bg-lighter);
    border-radius: 4px;
    white-space: nowrap;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}

[data-testid="stButton"] > button:hover {
    background: var(--primary-dark) !important;
}

[data-testid="stPopover"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-lighter); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def do_manual_sync() -> int:
    """Manual sync with Google Drive"""
    with st.spinner("Syncing with Google Drive..."):
        try:
            files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
        except Exception as e:
            st.error(f"Sync error: {e}")
            return 0

    if not files:
        st.warning("No files found.")
        return 0

    progress = st.progress(0, text="Indexing...")
    count = 0
    
    for i, f in enumerate(files):
        try:
            file_bytes = download_file(f["id"], f["name"], f["mimeType"])
            pages = extract_text_from_bytes(file_bytes, f["name"], f["format"])
            for page in pages:
                page["drive_path"] = f["path"]
                page["file_format"] = f["format"]
                page["drive_modified_time"] = f.get("modifiedTime", "")
            
            chunks = chunk_pages(pages)
            add_chunks_to_store(chunks)
            count += 1
        except Exception as e:
            st.warning(f"{f['name']}: {str(e)[:40]}")
        
        progress.progress((i + 1) / len(files), text=f"Processing {f['name'][:25]}...")

    progress.empty()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 32px; padding-top: 12px;">
    <h1 style="font-size: 26px; font-weight: 600; margin: 0; color: #212529;">DriveRAG</h1>
    <p style="font-size: 13px; color: #6c757d; margin: 6px 0 0 0;">Ask questions about your Google Drive documents</p>
</div>
""", unsafe_allow_html=True)

# Chat area
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <h2 class="welcome-title">Welcome to DriveRAG</h2>
        <p class="welcome-desc">
            Powered by RAG (Retrieval-Augmented Generation)<br>
            Search and analyze your Google Drive documents instantly
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="messages-area">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("timestamp", "")
        msg_class = "user" if role == "user" else "assistant"
        
        if role == "user":
            st.markdown(f"""
            <div class="message-row user">
                <div class="msg-bubble">{content}</div>
            </div>
            <div style="text-align: right; margin-right: 10px;">
                <div class="msg-time">{ts}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            mode_text = "RAG" if msg.get("mode") == "rag" else "Direct"
            st.markdown(f"""
            <div class="message-row assistant">
                <div>
                    <div class="msg-meta">{mode_text} Response · {ts}</div>
                    <div class="msg-bubble">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if msg.get("sources"):
                with st.expander("View sources"):
                    for src in msg["sources"]:
                        st.text(f"From: {src.get('file', 'Document')} (Page {src.get('page', '?')})")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT AREA (Fixed Bottom)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="input-wrapper"><div class="input-section">', unsafe_allow_html=True)

# Display indexed files
indexed = get_indexed_files()
if indexed:
    file_list = ", ".join([f.get("name", "?")[:15] for f in indexed[:3]])
    if len(indexed) > 3:
        file_list += f", +{len(indexed)-3} more"
    st.markdown(f"""
    <div class="files-bar">
        <span class="files-count">{len(indexed)}</span> documents indexed: {file_list}
    </div>
    """, unsafe_allow_html=True)

# Main input area
st.markdown('<div class="input-row">', unsafe_allow_html=True)

# Add button
col_add, col_input, col_sync = st.columns([0.4, 1, 0.4], gap="small")

with col_add:
    with st.popover("+ Add", help="Add or configure"):
        st.markdown("**Options**")
        if st.button("Import Files", use_container_width=True, key="add_files"):
            st.info("File upload feature coming soon")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("RAG", use_container_width=True, key="mode_rag"):
                st.session_state.mode = "rag"
                st.rerun()
        with col2:
            if st.button("Direct", use_container_width=True, key="mode_direct"):
                st.session_state.mode = "direct"
                st.rerun()

# Main input
with col_input:
    placeholder_text = "Ask about your documents..." if st.session_state.mode == "rag" else "Ask anything..."
    user_input = st.text_input(
        "Message",
        placeholder=placeholder_text,
        label_visibility="collapsed",
        key="main_input"
    )

# Sync button
with col_sync:
    if st.button("Sync", use_container_width=True, key="sync_btn", help="Sync with Google Drive"):
        synced = do_manual_sync()
        if synced > 0:
            st.success(f"Synced {synced} file(s)")
            st.rerun()
        else:
            st.info("No new files to sync")

st.markdown('</div></div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

if user_input:
    ts_now = datetime.now().strftime("%H:%M")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": ts_now
    })
    
    # Build history
    previous = st.session_state.messages[:-1]
    history_direct = [{"role": m["role"], "content": m["content"]} for m in previous]
    
    from langchain.schema import HumanMessage, AIMessage
    history_rag = []
    for m in previous:
        if m["role"] == "user":
            history_rag.append(HumanMessage(content=m["content"]))
        else:
            history_rag.append(AIMessage(content=m["content"]))
    
    langsmith_meta = {
        "metadata": {
            "conversation_id": st.session_state.conversation_id,
            "thread_id": st.session_state.langsmith_thread_id,
            "mode": st.session_state.mode,
        }
    }
    
    # Generate response
    with st.spinner("Generating response..."):
        if st.session_state.mode == "rag":
            result = ask(
                user_input,
                external_history=history_rag,
                thread_id=st.session_state.langsmith_thread_id,
                conversation_id=st.session_state.conversation_id,
                langsmith_extra=langsmith_meta
            )
        else:
            from src.generation.llm_direct import ask_direct
            result = ask_direct(
                user_input,
                history=history_direct,
                thread_id=st.session_state.langsmith_thread_id,
                conversation_id=st.session_state.conversation_id,
                langsmith_extra=langsmith_meta
            )
    
    # Add response
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "mode": st.session_state.mode,
        "timestamp": ts_now
    })
    
    st.rerun()
