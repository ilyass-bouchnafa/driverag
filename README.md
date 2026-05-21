# DriveRAG

DriveRAG est un assistant RAG (Retrieval-Augmented Generation) conçu pour interroger des documents stockés dans Google Drive.

## Vue d’ensemble

Le projet utilise une architecture en couches :
- ingestion Google Drive
- extraction et chunking des documents
- indexation ChromaDB (+ Redis en cache optionnel)
- récupération hybride BM25 + dense
- reranking CrossEncoder
- génération de réponse via Groq avec citations
- interface utilisateur via API FastAPI + React (Streamlit est présent comme test/demo optionnel)

## Qui connecte tout ?

- `langchain` est la couche d’orchestration principale.
- Elle relie les composants de récupération, les prompts, les embeddings et les appels LLM.
- `langsmith` est utilisé pour le monitoring/tracing du pipeline via le décorateur `traceable` dans `src/generation/llm_chain.py`.

## Architecture complète

### 1. Ingestion / synchronisation

Fichiers principaux :
- `src/ingestion/gdrive_loader.py`
- `src/ingestion/file_router.py`
- `src/ingestion/chunker.py`
- `src/ingestion/sync_manager.py`

Fonctions clés :
- `list_files_recursive()` parcourt Google Drive et récupère les métadonnées des fichiers.
- `download_file()` télécharge chaque fichier Drive.
- `extract_text_from_bytes()` convertit les fichiers supportés en texte brut.
- `chunk_pages()` segmente le texte en morceaux de 1000 caractères avec 200 de recouvrement.
- `add_chunks_to_store()` indexe chaque chunk dans ChromaDB et synchronise Redis si disponible.

Formats supportés : PDF, DOCX, TXT, MD, PPTX, Google Docs.

### 2. Stockage

Fichier principal : `src/retrieval/vectorstore.py`

Composants :
- `ChromaDB` persistant : stocke embeddings, documents et métadonnées.
- `_doc_store` : cache en mémoire des textes originaux.
- `get_all_chunks()` : renvoie tous les chunks pour BM25.
- `get_indexed_files()` : liste les fichiers indexés pour l’interface.
- `delete_chunks_by_source()` : supprime les anciennes versions d’un document.

Redis optionnel :
- `src/retrieval/redis_corpus.py` gère la cache Redis des chunks pour accélérer BM25.
- Si Redis est disponible, `hybrid_search()` l’utilise d’abord, sinon il retombe sur ChromaDB.

### 3. Recherche hybride

Fichiers principaux :
- `src/retrieval/query_processor.py`
- `src/retrieval/hybrid_search.py`
- `src/retrieval/reranker.py`
- `src/retrieval/rrf.py`

Pipeline de récupération :
1. `generate_multi_queries()` reformule la requête en plusieurs variantes avec Groq.
2. `generate_hyde_document()` génère un document hypothétique (HyDE) à partir de la question.
3. `hybrid_search()` combine BM25 lexical et recherche dense ChromaDB.
4. `advanced_retrieve()` fusionne les résultats de toutes les variantes et trie par score.
5. `rerank()` applique un cross-encoder pour garder les `TOP_K_RERANKED` meilleurs chunks.

#### Calcul hybride

- BM25 est construit sur tous les chunks textuels.
- Dense search utilise `embed_query()` et `ChromaDB.query()`.
- Score final = `HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * bm25`.

### 4. Génération de réponse

Fichier principal : `src/generation/llm_chain.py`

Flux :
- Récupération avancée → reranking → formatage du contexte → appel LLM Groq.
- Le contexte est constitué de blocs `[{source}, Page {page}]` suivis du texte extrait.
- La requête finale envoyée au LLM contient :
  - un prompt système strict
  - l’historique de conversation (les 10 derniers messages maximum)
  - les documents récupérés + la question

#### Prompt principal

Le système demande à l’IA de :
- répondre uniquement à partir du contexte documentaire.
- ne pas inventer.
- citer chaque affirmation avec `[File Name, Page X]`.
- être concis.

Ce prompt est essentiel à la qualité des réponses et au respect du contenu des documents.

### 5. Mode direct

Fichier : `src/generation/llm_direct.py`

Ce mode ne fait pas de récupération documentaire.
- Utilise aussi `ChatGroq`.
- Se base uniquement sur la conversation simultanée et les messages précédents.
- Prompt : assistant académique expert, clair et structuré.

### 6. Orchestration / interface

#### FastAPI + React
- `backend/main.py` expose l’API :
  - `/chat`
  - `/sync`
  - `/files`
  - `/upload`
  - `/clear`
  - `/health`
- `frontend/` contient une application React qui consomme cette API.
- Le backend démarre aussi une synchronisation automatique (`start_auto_sync`).

#### Streamlit (usage test/demo)
- `app.py` est une interface de développement expérimental utilisée pour tester le chat.
- Ce n’est pas le cœur de l’architecture de production.

### 7. Évaluation RAGAS

Dossier : `evaluation/`

But : mesurer la performance du pipeline grâce à des métriques RAGAS.
- Questions de test (`evaluation/test_questions.py`)
- Script d’évaluation (`evaluation/ragas_eval.py`)
- Résultats sauvegardés dans `evaluation/last_results.json`

Métriques utilisées :
- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Modèle d’évaluation :
- RAGAS utilise un wrapper Groq dédié basé sur `llama-3.1-8b-instant`.
- Ce modèle est spécifique à l’évaluation et distinct du modèle utilisé pour la production du chat.

### 8. Configuration globale

Fichier : `src/config.py`

Paramètres importants :
- `GROQ_API_KEY`
- `GOOGLE_DRIVE_FOLDER_ID`
- `CHROMA_PERSIST_DIR`
- `REDIS_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `TOP_K_RETRIEVAL`, `TOP_K_RERANKED`
- `HYBRID_ALPHA`
- `MULTI_QUERY_COUNT`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`

Modèles utilisés :
- LLM Groq : `llama-3.1-8b-instant`
- Embeddings : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker CrossEncoder : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Monitoring et tracing :
- `langsmith` est utilisé pour tracer le pipeline et monitorer les exécutions via `langchain`.
- `src/generation/llm_chain.py` utilise le décorateur `traceable`.

L’environnement est chargé via `python-dotenv`.

### 9. Liste des composants clés

- `backend/main.py` : API FastAPI
- `frontend/` : application React
- `app.py` : interface Streamlit de test/demo
- `src/config.py` : configuration centrale
- `src/ingestion/` : ingestion Drive et chunking
- `src/retrieval/` : recherche hybride et stockage
- `src/generation/` : génération LLM
- `evaluation/` : métriques et validation

### 10. Orchestration de bout en bout

1. L’utilisateur lance le service Streamlit ou la stack backend+frontend.
2. Le système se connecte à Google Drive et détecte les fichiers valides.
3. Chaque document est téléchargé, converti en texte et découpé en chunks.
4. Les chunks sont indexés dans ChromaDB ; Redis sert de cache BM25.
5. Quand l’utilisateur pose une question :
   - Multi-query + HyDE génèrent plusieurs requêtes.
   - Chaque variante est recherchée en hybride.
   - Les résultats sont fusionnés, dédupliqués et rerankés.
   - Le contexte final est formaté pour le LLM.
   - Groq répond avec citations explicites.
6. La réponse est renvoyée à l’UI avec la liste des sources.

### 11. Prompts détaillés

#### Prompt RAG principal

- système : précision, académique, citation stricte, pas d’invention.
- question : incluse après `DOCUMENTS :` + contexte.
- format attendu : citations `[File Name, Page X]`.

#### Prompt Multi-Query

- objectif : générer exactement `MULTI_QUERY_COUNT` reformulations.
- règles : même langue, pas de numérotation, pas d’explication, pas de répétition.
- résultat : plusieurs variantes qui améliorent le rappel documentaire.

#### Prompt HyDE

- objectif : générer un passage académique de 150-200 mots qui répond à la question.
- règles : même langue, ton académique, ne pas mentionner que c’est hypothétique.
- usage : cet extrait est embarqué comme document supplémentaire pour la recherche.

#### Prompt Direct

- système : assistant académique expert.
- usage : mode sans contexte documentaire pour répondre librement.

### 12. Exécution et démarrage

1. Créer `.env` avec :
   - `GROQ_API_KEY`
   - `GOOGLE_DRIVE_FOLDER_ID`
   - éventuellement `REDIS_URL`
2. Installer les dépendances.
3. Lancer FastAPI + React :
   - backend : `uvicorn backend.main:app --reload`
   - frontend : `npm install && npm start`
4. Le backend lance une synchronisation automatique toutes les **30 minutes** (1800 secondes).
5. Cliquer sur `/sync` pour forcer une synchronisation manuelle.

### 13. Smart Sync et détection de changements

Le système utilise une synchronisation intelligente basée sur `modifiedTime` de Google Drive :
- **Nouveau fichier** → indexé directement
- **Fichier modifié** → supprimé + ré-indexé
- **Fichier inchangé** → ignoré

**Important** : Pour que cette détection fonctionne, ChromaDB doit contenir les métadonnées `drive_modified_time` pour chaque chunk. Si cette métadonnée est manquante ou incohérente, tous les fichiers seront considérés comme "modifiés".

Diagnostique en cas de re-chunking systématique :
- Vérifier les logs : chercher les messages `Unchanged` ou `Updated`.
- Si tous les fichiers sont marqués `Updated` à chaque sync, vérifier :
  1. Que `get_indexed_file_timestamps()` retourne bien les timestamps stockés.
  2. Que Google Drive API retourne des timestamps stables (`modifiedTime`).

### 14. Points d'extension possibles

- ajouter OCR / support audio
- supporter plus de formats dans `file_router.py`
- améliorer `hybrid_search.py` avec plus de signaux
- ajouter monitoring et métriques d’usage

---

## Conclusion

Ce README décrit l’architecture actuelle de DriveRAG, chaque composant et l’orchestration complète, ainsi que tous les prompts utilisés dans le pipeline.

