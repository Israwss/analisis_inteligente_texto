# Detección Inteligente de Noticias Falsas en Español

Proyecto de **análisis inteligente de textos** que clasifica notas periodísticas
en español como **verdaderas** o **falsas**, comparando modelos de *Machine
Learning* clásico (TF-IDF + 5 algoritmos) contra un *transformer* en español
(**BETO**).

> Asignatura: Análisis Inteligente de Textos — Facultad de Ingeniería, UNAM.

---

## Problema y motivación
La desinformación en redes y medios digitales tiene impacto social directo
(salud pública, procesos electorales, polarización). Detectarla automáticamente
en **español** —idioma con menos recursos que el inglés— es un problema abierto
y socialmente relevante.

**Hipótesis:** es posible discriminar noticias falsas de verdaderas en español
con F1-macro > 0.70 usando únicamente el contenido textual, y los modelos
basados en *transformers* (BETO) superan a los modelos clásicos TF-IDF.

## Dataset
**Spanish Fake News Corpus** (Posadas-Durán et al., 2019), base del shared task
**FakeDeS / IberLEF 2020**. Licencia CC-BY-4.0.
Repositorio: <https://github.com/jpposadas/FakeNewsCorpusSpanish>

| Split        | Notas | Clases            |
|--------------|-------|-------------------|
| train        | ~676  | true / fake       |
| development  | ~295  | true / fake       |
| test         | ~572  | balanceado        |

La descarga es **automática** (`python main.py download`).

## Estructura del repositorio
```
.
├── main.py                 # CLI orquestador
├── requirements.txt
├── src/
│   ├── data.py             # descarga + carga del corpus
│   ├── preprocess.py       # limpieza/normalización en español
│   ├── train_classic.py    # TF-IDF + 5 modelos + CV + ablación + figuras
│   └── train_beto.py       # fine-tuning de BETO (GPU)
├── paper/paper.md          # documento científico (6 secciones)
├── results/                # métricas y figuras (generadas)
└── models/                 # modelos serializados (generados)
```

## Instalación
```bash
# 1) Entorno
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Dependencias
pip install -r requirements.txt
```
> Si solo se usará el pipeline clásico, pueden omitirse `torch/transformers/datasets`.

## Reproducir los experimentos
```bash
python main.py download              # descarga el corpus
python main.py stats                 # estadísticas del dataset

# ML clásico: 5 modelos, validación cruzada 5-fold, métricas + figuras
python main.py classic

# Estudio de ablación: word vs char-ngram x con/sin stopwords (20 experimentos)
python main.py ablation

# Transformer en español (requiere GPU)
python main.py beto --epochs 4 --batch-size 16

# Todo el pipeline clásico de una vez
python main.py all
```

### Salidas
- `results/metrics_classic.csv` — tabla comparativa de los 5 modelos.
- `results/ablation.csv` — 20 experimentos del estudio de ablación.
- `results/metrics_beto.json` — métricas del transformer.
- `results/figures/` — matrices de confusión y curvas ROC.
- `models/` — mejor modelo serializado (`.joblib`) y BETO fine-tuneado.

## Métricas de evaluación
- **Accuracy** (referencia; el test está balanceado).
- **Precision / Recall / F1** de la clase *fake* (la clase de interés).
- **F1-macro** — métrica principal (criterio del shared task FakeDeS; no premia
  el sesgo hacia la clase mayoritaria).
- **ROC-AUC** — capacidad de *ranking* independiente del umbral.

## Modelos comparados
| Familia            | Modelos |
|--------------------|---------|
| Probabilístico     | Multinomial Naïve Bayes |
| Lineales           | Regresión Logística, SVM lineal |
| Ensambles          | Random Forest, Gradient Boosting |
| Transformer (es)   | BETO (`dccuchile/bert-base-spanish-wwm-cased`) |

## Reproducibilidad
`random_state = 42` en todos los componentes estocásticos; validación cruzada
estratificada de 5 *folds*; versiones fijadas en `requirements.txt`.

## Documento científico
Ver [`paper/paper.pdf`](paper/paper.md) — incluye Introducción, Estado de la
técnica, Marco teórico, Setup experimental, Resultados, Discusión/Trabajo futuro
y Bibliografía (21 referencias).

## Equipo
| Integrante              |  GitHub  |    Rol    |
|-------------------------|----------|-----------|
| Israel Martinez Jimenez | @israwss | developer |


## Licencia
Código bajo licencia MIT. El corpus se distribuye bajo CC-BY-4.0 (ver repositorio
original y cita correspondiente).
