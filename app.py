import sys
from pathlib import Path
import streamlit as st

root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))

from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.retrieval.vectorstore import add_chunks_to_store, get_indexed_files
from src.generation.llm_chain import ask, clear_memory
from src.config import GOOGLE_DRIVE_FOLDER_ID

st.set_page_config(
    page_title="DriveRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.source-box {
    background: #f0f2f6;
    border-left: 3px solid #6c63ff;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85em;
}
.badge {
    background: #6c63ff;
    color: white;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 0.75em;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Google Drive")

    # Synchronisation
    if st.button("🔄 Synchroniser depuis Drive", type="primary", use_container_width=True):
        with st.spinner("Connexion à Google Drive..."):
            try:
                files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
            except Exception as e:
                st.error(f"Erreur Drive : {e}")
                files = []

        if not files:
            st.warning("Aucun fichier supporté trouvé.")
        else:
            progress = st.progress(0)
            status = st.empty()

            for i, f in enumerate(files):
                status.text(f"📄 {f['name']}...")
                try:
                    file_bytes = download_file(f['id'], f['name'], f['mimeType'])
                    pages = extract_text_from_bytes(file_bytes, f['name'], f['format'])

                    # Ajouter le chemin Drive dans les métadonnées
                    for page in pages:
                        page['drive_path'] = f['path']
                        page['file_format'] = f['format']

                    chunks = chunk_pages(pages)
                    add_chunks_to_store(chunks)
                except Exception as e:
                    st.warning(f"Erreur avec {f['name']}: {e}")

                progress.progress((i + 1) / len(files))

            status.empty()
            st.success(f"✅ {len(files)} fichier(s) indexé(s) !")
            st.rerun()

    st.divider()

    # Fichiers indexés
    st.subheader("📚 Documents indexés")
    indexed = get_indexed_files()

    if indexed:
        # Grouper par format
        by_format = {}
        for f in indexed:
            fmt = f['format']
            by_format.setdefault(fmt, []).append(f)

        for fmt, files_of_fmt in by_format.items():
            icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "pptx": "📊"}.get(fmt, "📁")
            with st.expander(f"{icon} {fmt.upper()} ({len(files_of_fmt)})"):
                for f in files_of_fmt:
                    st.caption(f['path'])
    else:
        st.caption("Aucun document indexé.")

    st.divider()

    # Paramètres avancés
    with st.expander("⚙️ Paramètres"):
        st.slider("Chunks récupérés", 5, 20, 10, key="top_k")
        st.slider("Alpha Dense/BM25", 0.0, 1.0, 0.5, 0.1, key="alpha",
                  help="0=BM25 pur, 1=Dense pur")
        if st.button("🗑️ Effacer la mémoire"):
            clear_memory()
            st.success("Mémoire effacée")

    st.divider()
    st.caption("RAG Avancé · LangChain · ChromaDB · Gemini · Docker")

# ─── Zone principale ───────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🤖 DriveRAG")
    st.caption("Hybrid Search · Multi-query · HyDE · CrossEncoder Reranking · Source Attribution")
with col2:
    if st.button("🗑️ Effacer chat"):
        st.session_state.messages = []
        clear_memory()
        st.rerun()

# Initialiser la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Message de bienvenue
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Bonjour ! Je suis **DriveRAG**, ton assistant académique avancé.\n\n"
            "Commence par **synchroniser tes documents** depuis la sidebar. "
            "Je peux lire : PDF, Word, TXT, Markdown, PowerPoint.\n\n"
            "Je cite toujours mes sources (fichier + page) dans mes réponses."
        )

# Historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "pptx": "📊"}.get(src['format'], "📁")
                    st.markdown(
                        f'<div class="source-box">{icon} <strong>{src["file"]}</strong>'
                        f' — Page {src["page"]}/{src["total_pages"]}'
                        f'<span class="badge">{src["score"]}</span></div>',
                        unsafe_allow_html=True
                    )

# Input
if prompt := st.chat_input("Pose ta question sur tes documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche avancée en cours..."):
            result = ask(prompt)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📎 {len(result['sources'])} source(s) utilisée(s)"):
                for src in result["sources"]:
                    icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "pptx": "📊"}.get(src['format'], "📁")
                    st.markdown(
                        f'<div class="source-box">{icon} <strong>{src["file"]}</strong>'
                        f' — Page {src["page"]}/{src["total_pages"]}'
                        f'<span class="badge">{src["score"]}</span></div>',
                        unsafe_allow_html=True
                    )

        # Feedback
        col_a, col_b, col_c = st.columns([1, 1, 8])
        with col_a:
            st.button("👍", key=f"up_{len(st.session_state.messages)}")
        with col_b:
            st.button("👎", key=f"down_{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })