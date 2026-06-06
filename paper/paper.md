# Detección Inteligente de Noticias Falsas en Español: Comparación de Modelos Clásicos de Aprendizaje Automático y *Transformers*

**Autores:** _[Nombre 1], [Nombre 2], [Nombre 3]_
**Afiliación:** Facultad de Ingeniería, Universidad Nacional Autónoma de México (UNAM)
**Asignatura:** Análisis Inteligente de Textos
**Repositorio de código:** _[liga a GitHub]_
**Fecha:** junio de 2026

---

## Resumen
La proliferación de noticias falsas en plataformas digitales representa un riesgo
social y democrático documentado. Este trabajo aborda la detección automática de
desinformación en **español**, idioma con menos recursos computacionales que el
inglés. Empleamos el *Spanish Fake News Corpus* [1] y comparamos cinco
algoritmos clásicos de aprendizaje automático sobre representaciones TF-IDF
frente a un modelo *transformer* preentrenado en español, **BETO** [2]. La
métrica principal es F1-macro, consistente con el shared task FakeDeS de
IberLEF 2020 [3]. _[Resumir aquí el mejor F1-macro obtenido tras correr los
experimentos.]_ Los resultados respaldan la hipótesis de que el contenido
textual contiene señal suficiente para discriminar notas falsas con desempeño
competitivo.

**Palabras clave:** desinformación, procesamiento de lenguaje natural, español,
clasificación de texto, TF-IDF, BERT, BETO.

---

## 1. Introducción

### 1.1 Planteamiento del problema
Una *noticia falsa* (*fake news*) es información verificablemente falsa difundida
con apariencia de contenido periodístico legítimo [4]. Vosoughi, Roy y Aral [5]
demostraron empíricamente en *Science* que las noticias falsas se difunden en
Twitter más rápido, más lejos y más profundo que las verdaderas, principalmente
por su novedad y carga emocional. Allcott y Gentzkow [6] cuantificaron su
influencia en el proceso electoral estadounidense de 2016.

### 1.2 Justificación social y técnica
**Social:** la desinformación afecta la salud pública (p. ej., durante la
pandemia de COVID-19), los procesos electorales y la cohesión social. Verificar
manualmente el enorme volumen de contenido es inviable, por lo que se requieren
herramientas automáticas de apoyo a *fact-checkers*.
**Técnica:** la mayoría de los recursos y modelos de PLN se desarrollan para el
inglés; el español está **subrepresentado** pese a ser la segunda lengua materna
más hablada del mundo. Construir y evaluar detectores en español es, por tanto,
una contribución pertinente.

### 1.3 Objetivo e hipótesis
**Objetivo:** desarrollar y evaluar un sistema que clasifique notas en español
como *verdaderas* o *falsas* a partir de su contenido textual.
**Hipótesis (H1):** es posible alcanzar F1-macro > 0.70 usando solo el texto.
**Hipótesis (H2):** un *transformer* en español (BETO) supera a los modelos
clásicos TF-IDF.

### 1.4 Organización del documento
La Sección 2 revisa el estado de la técnica; la Sección 3 presenta el marco
teórico; la Sección 4 detalla el *setup* experimental reproducible; la Sección 5
expone los experimentos y resultados; la Sección 6 discute hallazgos, valida la
hipótesis y plantea trabajo futuro. Cierra la bibliografía.

---

## 2. Estado de la cuestión / Estado de la técnica
La detección automática de desinformación ha sido revisada de forma sistemática
por Shu et al. [4] y por Zhou y Zafarani [7], quienes distinguen enfoques
basados en **contenido** (estilo, lenguaje) y en **contexto social**
(propagación, usuarios). Conroy, Rubin y Chen [8] propusieron taxonomías
tempranas de detección lingüística del engaño.

Para el **español**, Posadas-Durán et al. [1] introdujeron el primer corpus
balanceado y reportaron resultados base con bolsa de palabras y clasificadores
clásicos. El shared task **FakeDeS** de IberLEF 2020 [3] consolidó la tarea: los
sistemas ganadores combinaron rasgos léxicos con modelos basados en
*transformers*. Aragón et al. y otros participantes mostraron que BETO [2] y
modelos multilingües (mBERT, XLM-R) [9] mejoran sobre TF-IDF, aunque el margen
depende del tamaño —reducido— del corpus.

Nuestro trabajo se sitúa en el enfoque **basado en contenido** y aporta una
comparación controlada y reproducible entre cinco familias de modelos clásicos
y un *transformer*, incluyendo un estudio de ablación sobre la representación.

---

## 3. Marco teórico
Conceptos necesarios para comprender la propuesta:

- **Representación vectorial de texto.** Para que un algoritmo procese texto, se
  transforma en vectores numéricos. El modelo **bolsa de palabras** y su
  ponderación **TF-IDF** (*Term Frequency–Inverse Document Frequency*) [10]
  asignan mayor peso a términos frecuentes en un documento pero raros en el
  corpus. Usamos n-gramas de palabra y de **carácter**, estos últimos robustos
  ante la morfología flexiva y las erratas del español.
- **Clasificadores clásicos.** *Naïve Bayes multinomial* (probabilístico),
  *Regresión Logística* y *SVM lineal* [11] (modelos lineales de margen),
  *Random Forest* [12] y *Gradient Boosting* (ensambles de árboles).
- **Word embeddings.** Representaciones densas distribuidas —word2vec [13],
  GloVe [14]— que capturan similitud semántica, antecedente conceptual de los
  *transformers*.
- **Transformers y BERT.** La arquitectura *Transformer* [15] basada en
  auto-atención permite modelar dependencias de largo alcance. **BERT** [16] la
  preentrena con modelado de lenguaje enmascarado y se ajusta (*fine-tuning*) a
  tareas específicas. **BETO** [2] es BERT preentrenado sobre un gran corpus en
  español, lo que aporta conocimiento lingüístico del idioma.
- **Validación y métricas.** Validación cruzada estratificada para estimar
  desempeño con varianza controlada; **F1-macro** como métrica robusta al
  desbalance (promedia el F1 de cada clase sin ponderar por frecuencia) y
  **ROC-AUC** como medida de *ranking* independiente del umbral.

---

## 4. Setup experimental (reproducibilidad)

### 4.1 Datos
*Spanish Fake News Corpus* v1.0/v2.0 [1] (CC-BY-4.0). Se concatenan los splits
*train* + *development* para entrenamiento y se evalúa sobre *test* (held-out).
Etiquetas: `true=0`, `fake=1`. Texto = `Headline` + `Text`.

### 4.2 Software
- Python 3.10+
- scikit-learn ≥ 1.3 [17], pandas, numpy, matplotlib/seaborn
- nltk (stopwords) [18]
- PyTorch ≥ 2.1, HuggingFace Transformers ≥ 4.38 [19], datasets, accelerate
- Versiones exactas fijadas en `requirements.txt`.

### 4.3 Hardware
- _CPU: [modelo], RAM: [GB]_ para los modelos clásicos.
- _GPU: [modelo, p. ej. NVIDIA RTX 3060 12 GB], CUDA [versión]_ para BETO.

### 4.4 Preprocesamiento
Minúsculas; eliminación de URLs, menciones (`@`/`#`) y caracteres no
alfabéticos (se conservan tildes y `ñ`); tokenización por espacios; remoción
opcional de stopwords y de acentos (variables del experimento). Implementado en
`src/preprocess.py`.

### 4.5 Modelos e hiperparámetros
| Modelo | Hiperparámetros clave |
|--------|-----------------------|
| Naïve Bayes (multinomial) | `alpha=0.1` |
| Regresión Logística | `C=10`, `max_iter=2000` |
| SVM lineal | `C=1.0` |
| Random Forest | `n_estimators=300` |
| Gradient Boosting | `n_estimators=200`, `max_depth=3` |
| **BETO** | `lr=2e-5`, `epochs=4`, `batch=16`, `max_len=256`, `weight_decay=0.01`, `warmup_ratio=0.1`, `fp16` |

**TF-IDF:** palabra (1–2)-gramas, `min_df=2`, `max_df=0.9`, `sublinear_tf`,
`max_features=50000`; carácter (`char_wb`) 2–5-gramas.

### 4.6 Protocolo experimental
- **Semilla fija** `random_state=42` en todos los componentes.
- **Validación cruzada** estratificada de **5 folds** (5 iteraciones por modelo)
  para estimar F1-macro en entrenamiento.
- **Ajuste final** sobre todo el entrenamiento y evaluación única sobre *test*.
- **Estudio de ablación:** 2 analizadores (palabra/carácter) × 2 (con/sin
  stopwords) × 5 modelos = **20 configuraciones** (`src/train_classic.py::run_ablation`).

---

## 5. Ejecución y resultados

> _Esta sección se completa automáticamente con las salidas de
> `results/metrics_classic.csv`, `results/ablation.csv`,
> `results/metrics_beto.json` y las figuras de `results/figures/`._

### 5.1 Comparación de modelos clásicos
_Insertar tabla `results/metrics_classic.csv`._

| Modelo | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | **F1-macro** | ROC-AUC |
|--------|---------:|-----------------:|--------------:|----------:|-------------:|--------:|
| ... | | | | | | |

### 5.2 Estudio de ablación
_Insertar tabla `results/ablation.csv` y discutir el efecto de la representación
(palabra vs carácter) y de las stopwords._

### 5.3 Transformer (BETO)
_Insertar métricas de `results/metrics_beto.json`._

### 5.4 Figuras
- Matrices de confusión: `results/figures/cm_*.png`.
- Curvas ROC comparadas: `results/figures/roc_all.png`.

### 5.5 Justificación de métricas
Se eligió **F1-macro** como métrica principal porque equilibra precisión y
exhaustividad de **ambas** clases sin premiar el sesgo hacia la mayoritaria
—criterio del shared task FakeDeS [3]—. *Accuracy* se reporta por ser el test
balanceado, pero no basta cuando los costos de error son asimétricos.
**ROC-AUC** mide la calidad del *ranking* con independencia del umbral, útil si
se desea ajustar la sensibilidad del detector. Se prioriza el *recall* de la
clase *fake* porque omitir una noticia falsa (falso negativo) suele ser más
costoso socialmente que revisar de más una verdadera.

---

## 6. Discusión y trabajo futuro

### 6.1 Interpretación de resultados
_¿Por qué se obtuvieron estos resultados? Discutir: tamaño reducido del corpus,
señales léxicas/estilísticas explotadas por TF-IDF, ventaja (o no) de BETO según
la cantidad de datos, posibles fugas de información por la fuente/medio._

### 6.2 Validación de la hipótesis
_H1 (F1-macro > 0.70): aceptar/rechazar según resultados. H2 (BETO > clásicos):
aceptar/rechazar. Indicar la diferencia observada y, de ser posible, su
significancia (p. ej., desviación estándar entre folds o prueba estadística)._

### 6.3 Viabilidad y potencial comercial
El sistema es una **prueba de concepto** reproducible. Para un producto viable
faltaría: mayor volumen y diversidad de datos, robustez ante dominios nuevos
(COVID-19, política), explicabilidad para *fact-checkers* y monitoreo de
*data drift*. Caso de uso comercial: módulo de apoyo a redacciones y
plataformas de verificación.

### 6.4 Limitaciones y trabajo futuro
- Corpus pequeño → explorar *data augmentation* y modelos multilingües (XLM-R) [9].
- Solo contenido → incorporar señales de **propagación** y de **fuente** [4,7].
- Evaluar explicabilidad (SHAP/LIME) y sesgos por tema/medio.
- Validación cruzada de dominio (train en un tema, test en otro).

---

## Bibliografía

[1] Posadas-Durán, J. P., Gómez-Adorno, H., Sidorov, G., & Moreno, J. J. M. (2019). *Detection of fake news in a new corpus for the Spanish language*. Journal of Intelligent & Fuzzy Systems, 36(5), 4869–4876.

[2] Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data*. PML4DC Workshop, ICLR 2020.

[3] Aragón, M. E., Jarquín-Vásquez, H., Montes-y-Gómez, M., et al. (2020). *Overview of FakeDeS at IberLEF 2020: Fake News Detection in Spanish*. Procesamiento del Lenguaje Natural, 65, 223–231.

[4] Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective*. ACM SIGKDD Explorations, 19(1), 22–36.

[5] Vosoughi, S., Roy, D., & Aral, S. (2018). *The spread of true and false news online*. Science, 359(6380), 1146–1151.

[6] Allcott, H., & Gentzkow, M. (2017). *Social Media and Fake News in the 2016 Election*. Journal of Economic Perspectives, 31(2), 211–236.

[7] Zhou, X., & Zafarani, R. (2020). *A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities*. ACM Computing Surveys, 53(5), 1–40.

[8] Conroy, N. J., Rubin, V. L., & Chen, Y. (2015). *Automatic deception detection: Methods for finding fake news*. ASIST, 52(1), 1–4.

[9] Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*. ACL 2020.

[10] Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513–523.

[11] Joachims, T. (1998). *Text categorization with Support Vector Machines*. ECML 1998, 137–142.

[12] Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

[13] Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS 2013.

[14] Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014, 1532–1543.

[15] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

[16] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019.

[17] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.

[18] Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

[19] Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020 (System Demos), 38–45.
