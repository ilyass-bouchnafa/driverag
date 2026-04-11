import sys
from pathlib import Path

# === IMPORTANT : Ajouter la racine du projet pour les imports src.* ===
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Wrappers Ragas
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ton projet
from src.generation.llm_chain import ask
from src.retrieval.query_processor import advanced_retrieve
from src.config import GROQ_API_KEY, RERANKER_MODEL   # optionnel


def run_evaluation():
    """
    Évaluation RAGAS sans OpenAI → Groq (LLM Judge) + HuggingFace Embeddings (local)
    """

    # ── Questions de test (à adapter avec de vraies questions précises sur TES documents) ──
    test_questions = [
        "Quelle est la définition principale de la gestion de mémoire ?",
        "Quels sont les avantages de SGBD ?",
        "C'est quoi une base de données ?",
        "C'est quoi le traitement de l'image ?",
        "Quel est le filtre adapté au bruit ?",
        # Ajoute ici 5 à 10 questions plus spécifiques issues de tes PDFs
    ]

    print("=" * 70)
    print("🚀 ÉVALUATION RAGAS - Groq + Embeddings locaux")
    print(f"{len(test_questions)} questions de test")
    print("=" * 70)

    answers = []
    contexts = []

    for i, q in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] → {q}")

        # Récupérer la réponse via ton pipeline RAG
        result = ask(q)
        answers.append(result.get("answer") or result.get("content", "No answer"))

        # Récupérer les contexts (top 5 chunks)
        chunks = advanced_retrieve(q, k=5)
        contexts.append([c.get("text", str(c)) for c in chunks])

    # Créer le dataset Ragas
    dataset = Dataset.from_dict({
        "question": test_questions,
        "answer": answers,
        "contexts": contexts
    })

    # ===================== CONFIGURATION SANS OPENAI =====================
    print("\n🔧 Configuration Groq (LLM Judge) + Embeddings locaux...")

    # LLM Judge avec Groq (température 0 pour des évaluations stables)
    judge_llm = ChatGroq(
        model="llama-3.1-8b-instant",   # rapide et suffisant
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    wrapped_llm = LangchainLLMWrapper(judge_llm)

    # Embeddings légers et multilingues (compatible français)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ~120 Mo
        # Alternative encore plus légère : "sentence-transformers/all-MiniLM-L6-v2"
    )
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Attacher les modèles aux métriques
    faithfulness.llm = wrapped_llm
    answer_relevancy.llm = wrapped_llm
    answer_relevancy.embeddings = wrapped_embeddings

    # ===================== LANCEMENT DE L'ÉVALUATION =====================
    print("\n📊 Calcul des métriques en cours (Faithfulness + Answer Relevancy)...")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )

    # ===================== AFFICHAGE DES RÉSULTATS =====================
    print("\n" + "=" * 70)
    print("📈 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)
    print(f"Faithfulness     : {results['faithfulness']:.3f} / 1.0")
    print(f"Answer Relevancy : {results['answer_relevancy']:.3f} / 1.0")
    print("=" * 70)

    faith = results['faithfulness']
    if faith >= 0.80:
        print("✅ Excellent ! Ton RAG est très fidèle et pertinent.")
    elif faith >= 0.65:
        print("⚠️  Correct mais perfectible (améliore le prompt système, le chunking ou le reranking).")
    else:
        print("❌ Faithfulness faible → vérifie la qualité du retrieval et du prompt.")

    return results


if __name__ == "__main__":
    run_evaluation()