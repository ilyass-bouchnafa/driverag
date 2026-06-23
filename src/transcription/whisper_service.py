"""
src/transcription/whisper_service.py

Corrige deux problemes de l'ancien /transcribe :
  1. `whisper.load_model("base")` etait rappele a CHAQUE requete (cout
     CPU/RAM inutile, charge le modele depuis le disque chaque fois).
  2. `language="fr"` etait force en dur, alors que l'objectif du projet
     est multilingue -> un etudiant parlant anglais ou arabe au micro
     se faisait "corriger" vers du francais par Whisper.

Strategie retenue pour la detection de langue :
-------------------------------------------------
Whisper sait detecter la langue automatiquement (`model.detect_language`)
a partir des 30 premieres secondes d'audio, avec un vecteur de
probabilites par langue. On NE force PAS la langue, sauf dans le cas
ambigu suivant :

    Si les deux langues les plus probables sont proches en probabilite
    (ecart < AMBIGUITY_MARGIN) ET que l'une des deux est FR ou EN,
    on privilegie FR/EN plutot qu'une langue improbable.

Pourquoi ce choix et pas "toujours auto-detecter sans biais" ?
------------------------------------------------------------------
Whisper confond frequemment des langues proches phonetiquement quand le
segment audio est court (ex: une question de 3 secondes peut etre
classee "ca" catalan ou "gl" galicien au lieu de "fr" a cause d'un
accent ou d'un bruit de fond) -- ce sont des erreurs frequentes connues
sur les clips courts. Comme l'utilisateur cible est un etudiant
ENSA/IMT Mines Ales dont le corpus de cours est tres majoritairement
FR/EN, biaiser vers FR/EN en cas d'ambiguite reduit fortement ces
erreurs de classification sans empecher la detection d'autres langues
(arabe, espagnol, etc.) quand le signal est suffisamment clair.
"""

import os
import time
import tempfile
import threading
from typing import Optional

import whisper

# Modele charge une seule fois (singleton), comme le reranker.
_model = None
_lock = threading.Lock()

# Marge de probabilite sous laquelle on considere la detection "ambigue"
AMBIGUITY_MARGIN = 0.15

# Langues privilegiees en cas d'ambiguite, par ordre de preference
PREFERRED_LANGUAGES = ["fr", "en"]


def get_whisper_model(model_name: str = "base"):
    """Charge le modele Whisper une seule fois, le garde en memoire."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print(f"whisper: chargement du modele '{model_name}'...")
                _model = whisper.load_model(model_name)
    return _model


def _resolve_language(probs: dict) -> str:
    """
    Decide la langue finale a partir des probabilites de detection
    Whisper, en appliquant le biais FR/EN documente ci-dessus.
    """
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_lang, top_prob = ranked[0]

    if len(ranked) < 2:
        return top_lang

    second_lang, second_prob = ranked[1]
    is_ambiguous = (top_prob - second_prob) < AMBIGUITY_MARGIN

    if not is_ambiguous:
        return top_lang  # detection nette, on ne touche a rien

    # Ambigu : si une langue preferee est dans le top des candidats
    # plausibles, on la choisit plutot que la 1ere place incertaine.
    candidates = {lang: p for lang, p in ranked if (top_prob - p) < AMBIGUITY_MARGIN}
    for preferred in PREFERRED_LANGUAGES:
        if preferred in candidates:
            return preferred

    return top_lang


def transcribe_audio(file_bytes: bytes, suffix: str = ".webm") -> dict:
    """
    Transcrit un fichier audio en detectant automatiquement la langue
    (avec biais FR/EN en cas d'ambiguite, voir _resolve_language).

    Returns
    -------
    dict: {"text": str, "language": str, "language_probability": float}
    """
    model = get_whisper_model()

    tmp_path = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time() * 1000)}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        _, probs = model.detect_language(mel)
        language = _resolve_language(probs)

        result = model.transcribe(tmp_path, language=language, fp16=False)

        return {
            "text": result["text"].strip(),
            "language": language,
            "language_probability": round(float(probs.get(language, 0.0)), 3),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)