"""
Descarga y carga del *Spanish Fake News Corpus* (Posadas-Durán et al., 2019).

Repositorio original: https://github.com/jpposadas/FakeNewsCorpusSpanish
Licencia del corpus: CC-BY-4.0

Columnas del corpus:
    Id, Category (true/fake), Topic, Source, Headline, Text, Link
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

# Raíz del proyecto (carpeta que contiene este archivo -> sube un nivel)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# El repositorio usa la rama `master`; dejamos `main` como respaldo.
_BRANCHES = ("master", "main")
_FILES = {
    "train": "train.xlsx",
    "development": "development.xlsx",
    "test": "test.xlsx",
}
_BASE = "https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/{branch}/{fname}"
_ZIP = "https://codeload.github.com/jpposadas/FakeNewsCorpusSpanish/zip/refs/heads/{branch}"
# raw.githubusercontent exige User-Agent; sin él responde 404.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsCorpus/1.0)"}


def _download_one(fname: str, dest: Path) -> None:
    """Descarga un archivo probando las ramas conocidas (raw, con respaldo ZIP)."""
    last_err: Exception | None = None
    for branch in _BRANCHES:
        url = _BASE.format(branch=branch, fname=fname)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=60)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                print(f"  [ok] {fname}  ({len(resp.content)/1024:.0f} KB)  [{branch}]")
                return
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"No se pudo descargar {fname}: {last_err}")


def _download_via_zip() -> bool:
    """Respaldo: descarga el repo completo como ZIP y extrae los .xlsx."""
    import zipfile
    for branch in _BRANCHES:
        try:
            resp = requests.get(_ZIP.format(branch=branch), headers=_HEADERS, timeout=120)
            if resp.status_code != 200 or not resp.content:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if base in _FILES.values():
                        (RAW_DIR / base).write_bytes(zf.read(member))
                        print(f"  [ok] {base}  [zip:{branch}]")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  ! respaldo ZIP [{branch}] falló: {exc}")
            continue
    return False


def download(force: bool = False) -> None:
    """Descarga los tres splits del corpus a data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando corpus a {RAW_DIR} ...")
    pending = []
    for fname in _FILES.values():
        dest = RAW_DIR / fname
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"  - {fname} ya existe (usa --force para re-descargar)")
            continue
        pending.append((fname, dest))

    if not pending:
        print("Listo.\n")
        return

    # Estrategia: primero el ZIP completo (más robusto ante 404 del CDN raw);
    # si falla, se intenta archivo por archivo desde raw.githubusercontent.
    if _download_via_zip():
        print("Listo.\n")
        return
    print("  ! Respaldo ZIP falló; intentando archivo por archivo...")
    for fname, dest in pending:
        _download_one(fname, dest)
    print("Listo.\n")


def _normalize_label(series: pd.Series) -> pd.Series:
    """Mapea la columna Category a 0=true / 1=fake (clase positiva = noticia falsa)."""
    s = series.astype(str).str.strip().str.lower()
    mapping = {"true": 0, "real": 0, "verdadero": 0, "fake": 1, "false": 1, "falso": 1}
    return s.map(mapping)


def load_split(split: str) -> pd.DataFrame:
    """
    Carga un split ('train' | 'development' | 'test') como DataFrame con columnas:
        text  -> Headline + Text concatenados
        label -> 0 (true) / 1 (fake)
    además de las columnas originales.
    """
    if split not in _FILES:
        raise ValueError(f"split debe ser uno de {list(_FILES)}; recibí {split!r}")
    path = RAW_DIR / _FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro {path}. Ejecuta primero:  python main.py download"
        )

    df = pd.read_excel(path)
    # Homogeneizar nombres de columnas (algunos splits varían en mayúsculas).
    df.columns = [c.strip().capitalize() for c in df.columns]

    headline = df.get("Headline", pd.Series([""] * len(df))).fillna("")
    body = df.get("Text", pd.Series([""] * len(df))).fillna("")
    df["text"] = (headline.astype(str) + ". " + body.astype(str)).str.strip()
    df["label"] = _normalize_label(df["Category"])

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Devuelve {'train':..., 'development':..., 'test':...}."""
    return {k: load_split(k) for k in _FILES}


if __name__ == "__main__":
    download()
    for name, frame in load_all().items():
        n_fake = int(frame["label"].sum())
        print(f"{name:12s}: {len(frame):4d} notas  |  fake={n_fake}  true={len(frame)-n_fake}")
