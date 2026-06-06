"""
Pipeline de Machine Learning clásico para detección de noticias falsas en español.

Compara 5 algoritmos sobre representaciones TF-IDF y realiza un estudio de
ablación sobre las variables del modelo (tipo de n-grama y uso de stopwords).

Métricas (justificadas en el paper):
  - Accuracy (referencia, dataset balanceado)
  - Precision / Recall / F1 (clase positiva = noticia FALSA)
  - F1-macro (métrica principal del shared task FakeDeS/IberLEF)
  - ROC-AUC (capacidad de ranking, independiente del umbral)

Salidas:
  results/metrics_classic.csv     -> tabla de resultados
  results/ablation.csv            -> estudio de ablación
  results/figures/*.png           -> matrices de confusión y curvas ROC
  models/<modelo>.joblib          -> mejor modelo serializado
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sin ventana (servidores / CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import data as data_mod
from .preprocess import clean_series

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
SEED = 42


# --------------------------------------------------------------------------- #
# Definición de modelos                                                        #
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    """Devuelve los clasificadores a comparar (hiperparámetros razonables)."""
    return {
        "NaiveBayes": MultinomialNB(alpha=0.1),
        "LogReg": LogisticRegression(C=10, max_iter=2000, random_state=SEED),
        "LinearSVM": LinearSVC(C=1.0, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=SEED
        ),
        "GradBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=SEED
        ),
    }


def make_vectorizer(analyzer: str = "word", use_stopwords: bool = True) -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF según la variable del experimento."""
    if analyzer == "word":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9,
            sublinear_tf=True, max_features=50000,
        )
    # n-gramas de carácter: robustos ante erratas/morfología del español.
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        sublinear_tf=True, max_features=50000,
    )


# --------------------------------------------------------------------------- #
# Evaluación                                                                   #
# --------------------------------------------------------------------------- #
def _scores(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            out["roc_auc"] = np.nan
    return out


def _decision_scores(model, X):
    """Obtiene scores continuos para ROC-AUC (prob o función de decisión)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["true", "fake"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y_true, y_score) in curves.items():
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC — conjunto de prueba")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experimento principal                                                        #
# --------------------------------------------------------------------------- #
def run(
    analyzer: str = "word",
    use_stopwords: bool = True,
    remove_accents: bool = False,
    cv_folds: int = 5,
    save_models: bool = True,
) -> pd.DataFrame:
    """Entrena y evalúa todos los modelos. Devuelve la tabla de métricas."""
    for d in (RESULTS, FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)

    print("== Cargando datos ==")
    splits = data_mod.load_all()
    train, dev, test = splits["train"], splits["development"], splits["test"]

    # Para entrenamiento usamos train+development (más datos); test queda intacto.
    df_tr = pd.concat([train, dev], ignore_index=True)
    print(f"Entrenamiento: {len(df_tr)} | Prueba: {len(test)}")

    print("== Preprocesando ==")
    pre = dict(remove_stopwords=use_stopwords, remove_accents=remove_accents)
    X_tr = clean_series(df_tr["text"], **pre)
    X_te = clean_series(test["text"], **pre)
    y_tr = df_tr["label"].values
    y_te = test["label"].values

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    rows = []
    roc_curves = {}
    best = {"name": None, "f1_macro": -1.0, "pipeline": None}

    for name, clf in get_models().items():
        pipe = Pipeline([
            ("tfidf", make_vectorizer(analyzer, use_stopwords)),
            ("clf", clf),
        ])
        print(f"\n-- {name} --")
        # Validación cruzada (k iteraciones) sobre el set de entrenamiento.
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"  CV F1-macro: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # Ajuste final y evaluación en prueba (held-out).
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_score = _decision_scores(pipe, X_te)

        m = _scores(y_te, y_pred, y_score)
        m.update({
            "model": name,
            "cv_f1_macro_mean": cv_f1.mean(),
            "cv_f1_macro_std": cv_f1.std(),
        })
        rows.append(m)
        roc_curves[name] = (y_te, y_score)
        plot_confusion(y_te, y_pred, f"{name} — prueba",
                       FIGURES / f"cm_{name}.png")
        print("  " + classification_report(y_te, y_pred,
              target_names=["true", "fake"]).replace("\n", "\n  "))

        if m["f1_macro"] > best["f1_macro"]:
            best.update(name=name, f1_macro=m["f1_macro"], pipeline=pipe)

    # Tabla de métricas ordenada.
    cols = ["model", "accuracy", "precision_fake", "recall_fake", "f1_fake",
            "f1_macro", "roc_auc", "cv_f1_macro_mean", "cv_f1_macro_std"]
    df_res = pd.DataFrame(rows)[cols].sort_values("f1_macro", ascending=False)
    df_res.to_csv(RESULTS / "metrics_classic.csv", index=False)
    plot_roc(roc_curves, FIGURES / "roc_all.png")

    print("\n================  RESULTADOS (ordenados por F1-macro)  ================")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if save_models and best["pipeline"] is not None:
        out = MODELS / f"best_classic_{best['name']}.joblib"
        joblib.dump(best["pipeline"], out)
        (RESULTS / "best_classic.json").write_text(
            json.dumps({"model": best["name"], "f1_macro": best["f1_macro"]}, indent=2)
        )
        print(f"\nMejor modelo: {best['name']} (F1-macro={best['f1_macro']:.4f}) -> {out}")

    return df_res


def run_ablation(cv_folds: int = 5) -> pd.DataFrame:
    """
    Estudio de ablación sobre las variables del modelo:
      - analyzer: word vs char_wb
      - stopwords: con vs sin
    => 4 configuraciones × 5 modelos = 20 experimentos.
    """
    print("\n########## ESTUDIO DE ABLACIÓN ##########")
    configs = [
        ("word", True), ("word", False),
        ("char", True), ("char", False),
    ]
    all_rows = []
    for analyzer, use_sw in configs:
        tag = f"{analyzer}_{'sw' if use_sw else 'nosw'}"
        print(f"\n### Configuración: {tag} ###")
        df = run(analyzer=analyzer, use_stopwords=use_sw,
                 cv_folds=cv_folds, save_models=False)
        df.insert(0, "config", tag)
        all_rows.append(df)
    abl = pd.concat(all_rows, ignore_index=True)
    abl.to_csv(RESULTS / "ablation.csv", index=False)
    print("\n== Ablación guardada en results/ablation.csv ==")
    best = abl.loc[abl["f1_macro"].idxmax()]
    print(f"Mejor combinación global: {best['config']} + {best['model']} "
          f"(F1-macro={best['f1_macro']:.4f})")
    return abl


if __name__ == "__main__":
    run()
