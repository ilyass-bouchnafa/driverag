"""
src/generation/conversation_store.py

Corrige la fuite de memoire entre sessions.

Probleme initial dans llm_chain.py :
---------------------------------------
    _history = []   # variable globale au niveau du MODULE

Toutes les requetes, quel que soit le thread_id/conversation_id envoye
par le frontend, lisaient/ecrivaient dans CETTE MEME liste partagee.
Meme en mono-utilisateur (un seul etudiant), c'est un bug reel des qu'il
ouvre deux conversations en parallele (deux onglets, ou "nouvelle
conversation" sans recharger le serveur) : les messages de la
conversation A se retrouvent injectes comme historique dans la
conversation B. De plus, un redemarrage du serveur (tres frequent en
dev local) efface tout instantanement.

Pourquoi SQLite et pas juste un dict en memoire scope par thread_id ?
-------------------------------------------------------------------------
Un dict en memoire scope par thread_id aurait deja corrige le melange
entre conversations, mais pas la perte au redemarrage. Avec SQLite (un
fichier .db local), l'etudiant peut fermer son PC, rouvrir l'app le
lendemain, et retrouver ses conversations -- comportement attendu d'un
outil de travail academique sur plusieurs semaines (cf. NotebookLM, qui
garde l'historique des "notebooks").
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager

from langchain.schema import HumanMessage, AIMessage, BaseMessage

DB_PATH = Path(__file__).parent.parent.parent / "data" / "conversations.db"

# Nombre de tours (paires question/reponse) gardes comme contexte envoye
# au LLM. Une limite est necessaire : au-dela, le prompt devient trop
# long (cout, latence, dilution de l'attention du LLM).
MAX_TURNS_IN_CONTEXT = 5


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Cree les tables si elles n'existent pas. A appeler au demarrage de l'app."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('human', 'ai')),
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON messages(thread_id)")
        conn.commit()


def append_message(thread_id: str, role: str, content: str):
    """Ajoute un message a l'historique d'un thread donne."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, time.time()),
        )
        conn.commit()


def get_history(thread_id: str, max_turns: int = MAX_TURNS_IN_CONTEXT) -> List[BaseMessage]:
    """
    Recupere l'historique d'un thread, converti en objets langchain
    (HumanMessage/AIMessage), limite aux N derniers tours.

    Le scoping par thread_id est la correction principale : deux threads
    differents ne peuvent jamais se melanger, contrairement a l'ancienne
    liste globale.
    """
    limit = max_turns * 2  # un tour = 1 message humain + 1 message IA
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at FROM messages
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ) ORDER BY created_at ASC
            """,
            (thread_id, limit),
        ).fetchall()

    messages: List[BaseMessage] = []
    for role, content in rows:
        if role == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def clear_thread(thread_id: str):
    """Efface l'historique d'UN thread specifique (pas tous les threads)."""
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.commit()


def list_threads() -> List[Dict]:
    """
    Liste les threads existants avec un apercu (premiere question, date),
    utile pour un futur historique de conversations dans l'UI (a la
    NotebookLM : liste des conversations passees).
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, MIN(content) as first_message, MIN(created_at) as started_at
            FROM messages
            WHERE role = 'human'
            GROUP BY thread_id
            ORDER BY started_at DESC
            """
        ).fetchall()
    return [
        {"thread_id": r[0], "preview": r[1][:80], "started_at": r[2]}
        for r in rows
    ]