"""
src/ingestion/chunk_identity.py

Corrige la collision d'identifiant de chunk.

Probleme initial :
-------------------
    chunk_id = f"{source}_p{page}_c{chunk_index}"

`source` est le NOM de fichier (ex: "TD1.pdf"). Si deux fichiers s'appellent
"TD1.pdf" dans deux dossiers Drive differents (tres frequent : un etudiant
range "TD1.pdf" dans "Semestre 5/" ET dans "Semestre 6/Rappels/"), les deux
fichiers generent EXACTEMENT le meme chunk_id pour leur page 1 / chunk 0.

Consequence concrete : suppression ou mise a jour de l'un des deux fichiers
ecrase/efface les chunks de l'AUTRE par collision de cle -> le contexte
recupere peut melanger ou perdre des passages d'un document qui n'a jamais
ete touche. C'est le bug "contexte recupere contredit le document reel".

Fix :
-----
On utilise le `drive_id` (identifiant unique fourni par Google Drive,
jamais reutilise meme si le fichier est renomme ou deplace) comme racine
de l'identifiant, jamais le nom de fichier.

    chunk_uid = f"{drive_id}_p{page}_c{chunk_index}"

`drive_id` doit etre propage depuis gdrive_loader.py -> file_router.py ->
chunker.py -> vectorstore.py. Le nom de fichier ("source") reste affiche
a l'utilisateur (pour les citations), mais n'est plus jamais utilise comme
cle d'identite.
"""


def build_chunk_uid(drive_id: str, page: int, chunk_index: int) -> str:
    """
    Construit l'identifiant unique et stable d'un chunk.

    drive_id est l'identifiant Google Drive du fichier (champ "id" retourne
    par l'API Drive), PAS le nom de fichier. Il ne change jamais, meme si
    l'utilisateur renomme ou deplace le fichier dans son Drive.
    """
    if not drive_id:
        raise ValueError(
            "drive_id manquant : un chunk ne peut pas etre identifie de "
            "facon stable sans l'id Drive du fichier source."
        )
    return f"{drive_id}_p{page}_c{chunk_index}"


def build_local_upload_uid(upload_session_id: str, page: int, chunk_index: int) -> str:
    """
    Variante pour les fichiers uploades directement depuis l'interface
    (avant qu'ils n'aient necessairement un drive_id, ex: pendant le
    traitement avant confirmation d'upload reussi vers Drive).

    upload_session_id doit etre un uuid4 genere a la reception du fichier.
    """
    return f"upload_{upload_session_id}_p{page}_c{chunk_index}"