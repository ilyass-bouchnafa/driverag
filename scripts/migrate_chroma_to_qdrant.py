"""
scripts/migrate_chroma_to_qdrant.py

Migration ponctuelle : relit tout le corpus existant dans ChromaDB et
le re-indexe dans Qdrant avec les nouveaux chunk_uid (bases sur drive_id).

A lancer UNE FOIS apres avoir deploye le nouveau code, avant la premiere
utilisation. Necessite que l'ancien vectorstore.py (Chroma) soit encore
present sur le disque (ne pas le supprimer avant d'avoir lance ce script).

IMPORTANT : si tes chunks existants dans Chroma n'ont pas de champ
drive_id (cas de TOUT le corpus actuel, indexe avant ce patch), ce
script ne peut pas deviner le drive_id retroactivement -- il faut
relancer un sync complet (force_all=True) depuis Google Drive plutot
que de migrer les chunks Chroma existants tels quels. C'est l'option
recommandee ci-dessous (option B), plus sure que la migration directe.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def option_a_migrate_existing_chunks():
    """
    Option A : migre les chunks Chroma existants tels quels.
    NE FONCTIONNE QUE SI drive_id est deja present dans leurs metadata
    (improbable pour un corpus indexe avant ce patch -- l'ancien
    chunker.py ne l'ecrivait pas).
    """
    from src.retrieval.vectorstore import get_all_chunks as get_all_chroma_chunks
    from src.retrieval.qdrant_store import index_chunks, rebuild_bm25_stats
    from src.ingestion.chunk_identity import build_chunk_uid

    old_chunks = get_all_chroma_chunks()
    print(f"{len(old_chunks)} chunks trouves dans Chroma")

    migratable = []
    skipped = 0
    for chunk in old_chunks:
        meta = chunk["metadata"]
        if not meta.get("drive_id"):
            skipped += 1
            continue
        meta["chunk_uid"] = build_chunk_uid(meta["drive_id"], meta["page"], meta["chunk_index"])
        migratable.append(chunk)

    print(f"{len(migratable)} migrables, {skipped} ignores (drive_id manquant)")

    if skipped > 0:
        print("ATTENTION: des chunks ont ete ignores. Utilise l'option B "
              "(resync complet depuis Drive) pour un corpus complet et coherent.")

    if migratable:
        from src.retrieval.sparse_encoder import Vocabulary, tokenize, compute_corpus_stats
        tokenized = [tokenize(c["text"]) for c in migratable]
        df, avgdl = compute_corpus_stats(tokenized)
        vocab = Vocabulary()
        for tokens in tokenized:
            for t in set(tokens):
                vocab.get_or_add(t)
        index_chunks(migratable, vocab, df, avgdl)
        print("Migration terminee.")


def option_b_full_resync_recommended():
    """
    Option B (RECOMMANDEE) : ignore les chunks Chroma existants, relance
    un sync complet depuis Google Drive avec le nouveau pipeline
    (chunker avec drive_id + sections, indexation Qdrant directe).

    Plus sur car ca garantit que TOUS les chunks ont un drive_id correct
    et un decoupage par section a jour, plutot que de migrer un melange
    d'anciens chunks (sans drive_id, sans section_title) avec les
    nouveaux.
    """
    from src.ingestion.sync_manager import smart_sync

    print("Lancement d'un sync complet (force_all=True) vers Qdrant...")
    print("Cela re-telecharge et re-indexe TOUS les fichiers Drive.")
    stats = smart_sync(force_all=True)
    print(f"Resultat: {stats}")


if __name__ == "__main__":
    print("=" * 60)
    print("Migration Chroma -> Qdrant")
    print("=" * 60)
    print("\nOption recommandee : B (resync complet, plus sur)")
    print("Option A n'est utile que si tes chunks Chroma ont deja drive_id\n")

    choice = input("Lancer l'option B (resync complet) ? [O/n] ").strip().lower()
    if choice in ("", "o", "oui", "y", "yes"):
        option_b_full_resync_recommended()
    else:
        option_a_migrate_existing_chunks()