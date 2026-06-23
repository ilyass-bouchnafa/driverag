"""
DriveRAG — RAGAS Evaluation
Compatible: ragas==0.1.21, langchain==0.2.16

This script runs the RAGAS evaluation pipeline using Groq LLM
(llama-3.3-70b-versatile as evaluator) and local sentence-transformers
embeddings. Uses the current Qdrant-based retrieval pipeline.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY missing in .env")
    sys.exit(1)

print("=" * 60)
print("RAGAS — DriveRAG Evaluation")
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
    model: str = "llama-3.3-70b-versatile"
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

        # Retry 3 times with increased timeout
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

print("1. Loading Groq LLM (llama-3.3-70b-versatile for evaluation)...")
groq_llm = GroqChatWrapper(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
ragas_llm = LangchainLLMWrapper(groq_llm)

print("2. Loading embeddings...")
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Assigner aux métriques
faithfulness.llm        = ragas_llm
answer_relevancy.llm    = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm   = ragas_llm
context_recall.llm      = ragas_llm

print("3. Importing RAG pipeline...")
try:
    from src.retrieval.query_processor import advanced_retrieve_async
    from src.retrieval.reranker import rerank
    from src.generation.llm_chain import ask_async
    from src.config import TOP_K_RETRIEVAL, TOP_K_RERANKED
    from src.ingestion.sync_manager import get_current_bm25_stats
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from evaluation.test_questions import TEST_QUESTIONS
except ImportError:
    TEST_QUESTIONS = [
        "À quoi sert le privilège RESOURCE accordé à un utilisateur Oracle ?",
        "Quel est l'avantage d'utiliser NUMBER(10,2) pour le champ prix de la table Produit en PL/SQL ?",
        "Que fait l'instruction EXIT WHEN c_prod%NOTFOUND dans une boucle de curseur PL/SQL ?",
        "Qu'est-ce qu'un BLOB en base de données ?",
        "Selon l'article d'Alan Turing en 1950, quel est le but principal de l'intelligence artificielle ?",
        "Quelle est la principale différence de capacité entre une IA faible et une IA forte ?",
        "Quel est le principe de l'algorithme K Plus Proches Voisins (KNN) pour classer une nouvelle donnée ?",
        "Quel est le rôle de la fonction noyau dans les Machines à Vecteur de Support (SVM) ?",
        "En traitement d'image, à quoi correspond l'échantillonnage numérique ?",
        "Quelle est la différence fondamentale de structure entre une image vectorielle et une image matricielle ?",
        "Quel est le but et le résultat principal d'une égalisation d'histogramme sur une image ?",
        "Quelles sont les caractéristiques principales du filtre médian en analyse d'image ?",
        "Quelles solutions existent pour traiter les effets de bord lors de l'application d'un filtre de convolution ?",
        "De quelles couleurs de base est composé le système de représentation RGB ?",
        "Qu'est-ce que le système de gestion de fichiers (SGF) et quel est son rôle global ?",
        "Que contient le super bloc d'une partition de disque sous Unix ?",
        "Dans l'en-tête d'un fichier binaire exécutable, à quoi sert le 'Nombre magique' ?",
        "Quelles sont les informations contenues dans un i-nœud ?",
        "Que se passe-t-il pour un lien symbolique lorsque le fichier original est supprimé ?",
        "Comment fonctionne l'allocation physique de fichiers par liste chaînée de blocs ?"
    ]

# Vérité terrain minimale pour context_precision et context_recall
GROUND_TRUTHS = [
    "Le privilège RESOURCE permet à l'utilisateur de créer des tables, des séquences, des procédures et d'autres objets dans la base de données.",
    "Utiliser NUMBER(10,2) permet de stocker un prix avec un haut niveau de précision, en incluant deux décimales.",
    "Cette instruction permet d'arrêter la boucle de lecture lorsque toutes les lignes retournées par le curseur ont été parcourues.",
    "Un BLOB (Binary Large Object) est un type de données utilisé pour stocker des fichiers binaires de grande taille, tels que des images, des vidéos ou des documents PDF.",
    "Le but de l'IA est de concevoir des systèmes capables de reproduire le comportement humain dans ses activités de raisonnement.",
    "Une IA faible résout des problèmes en détectant des motifs répétitifs, tandis qu'une IA forte (qui n'existe pas encore) serait capable de penser, raisonner et manipuler des concepts abstraits au même niveau qu'un être humain.",
    "L'algorithme KNN attribue un nouveau point à la catégorie la plus présente parmi ses K voisins les plus proches, après avoir calculé la distance qui les sépare.",
    "La fonction noyau permet de projeter les données d'entrée dans un espace de très grande dimension lorsque celles-ci ne sont pas linéairement séparables.",
    "L'échantillonnage correspond à la discrétisation de l'espace 2D de l'image, ce qui définit le nombre exact de points (pixels) qui pourront être coloriés.",
    "Une image vectorielle est représentée par des formes géométriques simples permettant un redimensionnement sans perte, contrairement à une image matricielle qui est représentée par une matrice de pixels (Bitmap).",
    "L'égalisation d'histogramme équilibre la distribution des pixels, ce qui a pour résultat d'augmenter globalement le contraste de l'image.",
    "Le filtre médian est un filtre non linéaire qui remplace la valeur d'un pixel par la médiane de ses voisins, ce qui supprime le bruit de type 'Poivre et Sel' tout en préservant l'information de contour.",
    "Pour traiter les effets de bord, on peut utiliser le 'zero padding' (considérer le voisinage hors image comme des valeurs nulles) ou la 'duplication' (attribuer la valeur du pixel de l'image le plus proche).",
    "Le système RGB est composé de trois couleurs primaires : le Rouge, le Vert et le Bleu.",
    "Le SGF est la partie du système d'exploitation chargée de gérer les fichiers, incluant leur création, suppression, les accès en lecture/écriture, le partage, la protection et l'allocation de l'espace disque.",
    "Le super bloc contient les paramètres clés du système de fichiers, notamment la taille des blocs, le nombre de blocs libres et la taille de la table des i-nœuds.",
    "Le 'Nombre magique' est une valeur particulière qui indique au système d'exploitation que le fichier respecte un format l'identifiant comme exécutable.",
    "Un i-nœud contient des informations de localisation (comme 13 adresses disques directes et indirectes) ainsi que des attributs généraux tels que le type de fichier, le propriétaire, la taille, les dates et les droits d'accès.",
    "Lors de la suppression du fichier original, le lien symbolique devient inutilisable car il pointe vers un chemin d'accès qui n'existe plus.",
    "Le fichier est conservé sous forme d'une liste de blocs dispersés sur le disque ; il suffit de garder l'adresse du premier bloc, car chaque bloc de la liste contient un pointeur vers le bloc suivant."
]


async def run_evaluation():
    print(f"\n📋 {len(TEST_QUESTIONS)} test questions")
    print("─" * 60)

    # Preload BM25 stats once (shared across all questions)
    print("\n   Loading BM25 stats...")
    vocab, df, n_docs = get_current_bm25_stats()
    print(f"   BM25: vocab={len(vocab.token_to_id)} terms, n_docs={n_docs}")


    answers, contexts, ground_truths = [], [], []
    errors = []

    print("\n4. Generating answers...")
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"   [{i+1}/{len(TEST_QUESTIONS)}] {question[:55]}...")
        eval_thread_id = f"eval_ragas_{i}"
        try:
            result = await ask_async(question, vocab=vocab, df=df, n_docs=n_docs, thread_id=eval_thread_id)
            answers.append(result["answer_for_eval"])

            raw_ctx = result["raw_contexts"]
            if len(raw_ctx) < TOP_K_RERANKED:
                from src.retrieval.query_processor import advanced_retrieve_async
                from src.retrieval.reranker import rerank
                candidates = await advanced_retrieve_async(question, vocab=vocab, df=df, n_docs=n_docs, k=TOP_K_RETRIEVAL)
                full_chunks = rerank(question, candidates, top_k=TOP_K_RERANKED)
                raw_ctx = [c["text"] for c in full_chunks]

            contexts.append(raw_ctx)
            ground_truths.append(GROUND_TRUTHS[i])

            print(f"      ✅ OK")
        except Exception as e:
            print(f"      ❌ {e}")
            errors.append(str(e))
            answers.append("Error")
            contexts.append(["No context"])
            ground_truths.append(GROUND_TRUTHS[i])

    print("\n5. Computing RAGAS metrics...")
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
                max_workers=1,        # sequential — no parallelism
                timeout=180,          # 3 minutes par appel
                max_retries=3,        # retry automatique
            )
        )
    except Exception as e:
        print(f"❌ RAGAS error: {e}")
        return {}

    faith   = float(results["faithfulness"])
    rel     = float(results["answer_relevancy"])
    prec    = float(results["context_precision"])
    rec     = float(results["context_recall"])

    print("\n" + "=" * 60)
    print("RESULTS")
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
    print(f"\n💾 Saved to {out_path}")
    return output


if __name__ == "__main__":
    try:
        from src.retrieval.qdrant_store import get_client, QDRANT_COLLECTION
        from qdrant_client.http.exceptions import UnexpectedResponse
        client = get_client()
        info = client.get_collection(QDRANT_COLLECTION)
        count = info.points_count
        if count == 0:
            print("⚠️  Qdrant collection empty — please sync first")
            sys.exit(1)
        print(f"✅ Qdrant: {count} chunks indexed")
    except UnexpectedResponse:
        print(f"❌ Qdrant collection '{QDRANT_COLLECTION}' not found — sync first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Qdrant: {e}")
        sys.exit(1)

    asyncio.run(run_evaluation())