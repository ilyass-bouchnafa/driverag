# evaluation/ragas_eval.py
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY manquant dans .env")
    sys.exit(1)

print("=" * 60)
print("RAGAS — Évaluation DriveRAG")
print("=" * 60)

# ── Imports RAGAS ────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# ── Groq via SDK direct (pas LangChain) ──────────────────
from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Wrapper RAGAS-compatible pour Groq ───────────────────
# RAGAS a besoin d'un objet avec une méthode .invoke()
# On crée un wrapper minimal qui respecte cette interface

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional

class GroqChatWrapper(BaseChatModel):
    """
    Wrapper minimal pour utiliser Groq avec RAGAS.
    Implémente uniquement ce que RAGAS utilise.
    """
    model: str = "llama-3.1-8b-instant"
    api_key: str = ""

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convertir les messages LangChain → format Groq
        groq_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "system":
                role = "system"
            elif msg.type == "ai":
                role = "assistant"
            groq_messages.append({"role": role, "content": msg.content})

        # Appel Groq SDK
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=groq_messages,
            temperature=0,
            max_tokens=1024
        )

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


# ── Embeddings via sentence-transformers ──────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("1. Chargement du LLM Groq...")
groq_llm = GroqChatWrapper(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)
ragas_llm = LangchainLLMWrapper(groq_llm)

print("2. Chargement des embeddings...")
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Assigner aux métriques
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings

# ── Import du pipeline RAG depuis src/ ───────────────────
print("3. Import du pipeline RAG...")
try:
    from src.retrieval.query_processor import advanced_retrieve
    from src.retrieval.reranker import rerank
    from src.generation.llm_chain import ask
    from src.config import TOP_K_RETRIEVAL, TOP_K_RERANKED
except ImportError as e:
    print(f"❌ Erreur import src/ : {e}")
    print("Lance depuis la racine du projet : python evaluation/ragas_eval.py")
    sys.exit(1)

# ── Questions de test ─────────────────────────────────────
try:
    from evaluation.test_questions import TEST_QUESTIONS
except ImportError:
    print("⚠️  test_questions.py non trouvé — utilisation des questions par défaut")
    TEST_QUESTIONS = [
        "Qu'est-ce que la gestion de la mémoire ?",
        "Comment fonctionne la pagination ?",
        "Qu'est-ce que PL/SQL ?",
    ]


def run_evaluation():
    print(f"\n📋 {len(TEST_QUESTIONS)} questions de test")
    print("─" * 60)

    answers, contexts = [], []
    errors = []

    # Générer les réponses
    print("\n4. Génération des réponses...")
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"   [{i+1}/{len(TEST_QUESTIONS)}] {question[:55]}...")
        try:
            result = ask(question)
            answers.append(result["answer"])

            candidates = advanced_retrieve(question, k=TOP_K_RETRIEVAL)
            final_chunks = rerank(question, candidates, top_k=TOP_K_RERANKED)
            contexts.append([c["text"] for c in final_chunks])

            print(f"      ✅ OK")
        except Exception as e:
            print(f"      ❌ {e}")
            errors.append(str(e))
            answers.append("Erreur")
            contexts.append(["Aucun contexte"])

    # Dataset RAGAS
    print("\n5. Calcul des métriques RAGAS...")
    dataset = Dataset.from_dict({
        "question": TEST_QUESTIONS,
        "answer": answers,
        "contexts": contexts,
    })

    try:
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            raise_exceptions=False
        )
    except Exception as e:
        print(f"❌ Erreur RAGAS : {e}")
        return {}

    faith = float(results["faithfulness"])
    rel = float(results["answer_relevancy"])

    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"  Faithfulness     : {faith:.3f} / 1.0  {'✅' if faith >= 0.8 else '⚠️'}")
    print(f"  Answer Relevancy : {rel:.3f} / 1.0  {'✅' if rel >= 0.8 else '⚠️'}")

    # Sauvegarder
    output = {"faithfulness": faith, "answer_relevancy": rel, "errors": errors}
    with open(ROOT / "evaluation" / "last_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n💾 Sauvegardé dans evaluation/last_results.json")

    return output


if __name__ == "__main__":
    # Vérifier ChromaDB
    try:
        from src.retrieval.vectorstore import get_collection
        count = get_collection().count()
        if count == 0:
            print("⚠️  ChromaDB vide — lance test_week1.py d'abord")
            sys.exit(1)
        print(f"✅ ChromaDB : {count} chunks")
    except Exception as e:
        print(f"❌ ChromaDB : {e}")
        sys.exit(1)

    run_evaluation()