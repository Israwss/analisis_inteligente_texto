"""
Preprocesamiento de texto en español.

Decisiones de diseño (justificadas en el paper, sección Marco teórico / Setup):
  - Minúsculas y normalización de espacios.
  - Eliminación de URLs, menciones y caracteres no alfabéticos (se conservan
    tildes y la 'ñ', que son informativas en español).
  - Lista de stopwords en español embebida (no requiere descargas en runtime).
  - La lematización/stemming se ofrece opcional: por defecto NO se aplica porque
    para TF-IDF con n-gramas de carácter suele perjudicar; se evalúa como variable
    del experimento.
"""
from __future__ import annotations

import re
import unicodedata

# Lista compacta de stopwords en español (subconjunto del estándar de NLTK).
SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
    "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
    "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías",
    "tuyo", "tuya", "suyo", "suya", "nuestro", "nuestra", "vuestro", "vuestra",
    "esos", "esas", "ha", "he", "has", "han", "fue", "ser", "es", "son", "era",
    "está", "están", "este", "tiene", "tienen", "había", "sería", "puede",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@#]\w+")
# Conserva letras (incluida ñ y vocales acentuadas) y espacios.
_NON_ALPHA_RE = re.compile(r"[^a-záéíóúüñ\s]")
_SPACES_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Quita tildes (útil como variable del experimento)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(
    text: str,
    remove_stopwords: bool = True,
    remove_accents: bool = False,
    min_token_len: int = 2,
) -> str:
    """Limpia un texto en español y devuelve la cadena normalizada."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    if remove_accents:
        text = strip_accents(text)
        # Tras quitar acentos, la regex permite el alfabeto sin tildes.
        text = re.sub(r"[^a-zñ\s]", " ", text)
    else:
        text = _NON_ALPHA_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()

    tokens = [t for t in text.split() if len(t) >= min_token_len]
    if remove_stopwords:
        stop = SPANISH_STOPWORDS
        if remove_accents:
            stop = {strip_accents(w) for w in stop}
        tokens = [t for t in tokens if t not in stop]
    return " ".join(tokens)


def clean_series(series, **kwargs):
    """Aplica clean_text a una pandas.Series."""
    return series.apply(lambda t: clean_text(t, **kwargs))
