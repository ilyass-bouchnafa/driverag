"""
DriveRAG — Évaluation RAGAS
Compatible : ragas==0.1.21, langchain==0.2.16
"""

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

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from groq import Groq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional

class GroqChatWrapper(BaseChatModel):
    model: str = "llama-3.1-8b-instant"
    api_key: str = ""

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        groq_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "system":
                role = "system"
            elif msg.type == "ai":
                role = "assistant"
            groq_messages.append({"role": role, "content": msg.content})

        # Retry 3 fois avec timeout augmenté
        for attempt in range(3):
            try:
                client = Groq(api_key=self.api_key, timeout=120.0)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=0,
                    max_tokens=512
                )
                content = response.choices[0].message.content
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception as e:
                if attempt == 2:
                    raise
                import time
                time.sleep(5)

from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("1. Chargement du LLM Groq...")
groq_llm = GroqChatWrapper(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
ragas_llm = LangchainLLMWrapper(groq_llm)

print("2. Chargement des embeddings...")
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Assigner aux métriques
faithfulness.llm        = ragas_llm
answer_relevancy.llm    = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm   = ragas_llm
context_recall.llm      = ragas_llm

print("3. Import du pipeline RAG...")
try:
    from src.retrieval.query_processor import advanced_retrieve
    from src.retrieval.reranker import rerank
    from src.generation.llm_chain import ask
    from src.config import TOP_K_RETRIEVAL, TOP_K_RERANKED
except ImportError as e:
    print(f"❌ Erreur import src/ : {e}")
    sys.exit(1)

try:
    from evaluation.test_questions import TEST_QUESTIONS
except ImportError:
    TEST_QUESTIONS = [
        "Quels sont les différents formats d'images étudiés ?",
        "Qu'est-ce que l'égalisation d'histogramme en traitement d'images ?",
        "Qu'est-ce qu'un shell et quelles sont ses différentes formes ?",
        "Qu'est-ce qu'une adresse logique et une adresse physique en système informatique ?",
        "Quelle est la structure physique d'un serveur Oracle ?",
        "Qu'est-ce qu'un trigger en base de données ?",
    ]

# Vérité terrain minimale pour context_precision et context_recall
GROUND_TRUTHS = [
    "Les formats d'images incluent JPEG, PNG, BMP, GIF, TIFF et RAW.",
    "L'égalisation d'histogramme améliore le contraste en redistribuant les niveaux de gris.",
    "Un shell est une interface entre l'utilisateur et le système, sous forme textuelle ou graphique.",
    "L'adresse logique est générée par le CPU, l'adresse physique est l'emplacement réel en RAM.",
    "La structure physique d'Oracle comprend les datafiles, redo log files et control files.",
    "Un trigger est un programme stocké qui s'exécute automatiquement sur un événement en base de données.",
]


def run_evaluation():
    print(f"\n📋 {len(TEST_QUESTIONS)} questions de test")
    print("─" * 60)

    answers, contexts, ground_truths = [], [], []
    errors = []

    print("\n4. Génération des réponses...")
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"   [{i+1}/{len(TEST_QUESTIONS)}] {question[:55]}...")
        try:
            result     = ask(question)
            answers.append(result["answer"])

            candidates = advanced_retrieve(question, k=TOP_K_RETRIEVAL)
            chunks     = rerank(question, candidates, top_k=TOP_K_RERANKED)
            contexts.append([c["text"] for c in chunks])
            ground_truths.append(GROUND_TRUTHS[i])

            print(f"      ✅ OK")
        except Exception as e:
            print(f"      ❌ {e}")
            errors.append(str(e))
            answers.append("Erreur")
            contexts.append(["Aucun contexte"])
            ground_truths.append(GROUND_TRUTHS[i])

    print("\n5. Calcul des métriques RAGAS...")
    dataset = Dataset.from_dict({
        "question":     TEST_QUESTIONS,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    try:
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,
            run_config=RunConfig(
                max_workers=1,        # séquentiel — pas de parallélisme
                timeout=180,          # 3 minutes par appel
                max_retries=3,        # retry automatique
            )
        )
    except Exception as e:
        print(f"❌ Erreur RAGAS : {e}")
        return {}

    faith   = float(results["faithfulness"])
    rel     = float(results["answer_relevancy"])
    prec    = float(results["context_precision"])
    rec     = float(results["context_recall"])

    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"  Faithfulness       : {faith:.3f} / 1.0  {'✅' if faith >= 0.8 else '⚠️'}")
    print(f"  Answer Relevancy   : {rel:.3f}   / 1.0  {'✅' if rel   >= 0.8 else '⚠️'}")
    print(f"  Context Precision  : {prec:.3f}  / 1.0  {'✅' if prec  >= 0.8 else '⚠️'}")
    print(f"  Context Recall     : {rec:.3f}   / 1.0  {'✅' if rec   >= 0.8 else '⚠️'}")

    output = {
        "faithfulness":      faith,
        "answer_relevancy":  rel,
        "context_precision": prec,
        "context_recall":    rec,
        "errors":            errors,
    }
    out_path = ROOT / "evaluation" / "last_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Sauvegardé dans {out_path}")
    return output


if __name__ == "__main__":
    try:
        from src.retrieval.vectorstore import get_collection
        count = get_collection().count()
        if count == 0:
            print("⚠️  ChromaDB vide — synchronise d'abord")
            sys.exit(1)
        print(f"✅ ChromaDB : {count} chunks")
    except Exception as e:
        print(f"❌ ChromaDB : {e}")
        sys.exit(1)

    run_evaluation()