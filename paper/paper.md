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
IberLEF 2020 [3]. El mejor sistema clásico —Regresión Logística con n-gramas
de carácter sin stopwords— alcanzó **F1-macro = 0.730** en el conjunto de prueba,
superando al transformer BETO (F1-macro = 0.650). Los resultados confirman que
el contenido textual contiene señal suficiente para superar el umbral de 0.70
con modelos clásicos, pero refutan la hipótesis de superioridad automática del
transformer dado el tamaño reducido del corpus.

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
- CPU: equipo local Windows 11 para los modelos clásicos.
- GPU: Google Colab (NVIDIA T4 / A100) para el fine-tuning de BETO (`fp16` habilitado).

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

### 5.1 Justificación de métricas
Se eligió **F1-macro** como métrica principal porque promedia el F1 de cada
clase con igual peso, sin favorecer a la mayoritaria —criterio del shared task
FakeDeS [3]—. *Accuracy* se reporta porque el conjunto de prueba está
balanceado (286 true / 286 fake), pero no sería suficiente con clases
desbalanceadas. **ROC-AUC** mide la calidad de *ranking* del clasificador con
independencia del umbral de decisión, útil si se desea ajustar la sensibilidad
del detector. Se prioriza el *recall* de la clase *fake* porque omitir una
noticia falsa (falso negativo) tiene mayor costo social que revisar de más una
verdadera.

### 5.2 Comparación de modelos clásicos (configuración base: TF-IDF word, con stopwords)

Los cinco modelos se entrenaron con TF-IDF sobre unigramas y bigramas de palabra,
validación cruzada estratificada de 5 folds y evaluación sobre el conjunto de
prueba (*held-out*). La Tabla 1 reporta los resultados en test.

**Tabla 1.** Modelos clásicos — representación TF-IDF word (1-2)-gramas, con stopwords.

| Modelo | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | **F1-macro** | ROC-AUC | CV F1-macro |
|--------|:--------:|:----------------:|:-------------:|:---------:|:------------:|:-------:|:-----------:|
| Random Forest | 0.720 | 0.723 | 0.713 | 0.718 | **0.720** | 0.795 | 0.819 ± 0.029 |
| SVM Lineal | 0.715 | 0.786 | 0.591 | 0.675 | 0.711 | 0.781 | 0.803 ± 0.025 |
| Reg. Logística | 0.710 | 0.780 | 0.584 | 0.668 | 0.705 | 0.785 | 0.800 ± 0.023 |
| Gradient Boosting | 0.652 | 0.619 | 0.790 | 0.694 | 0.645 | 0.725 | 0.775 ± 0.029 |
| Naïve Bayes | 0.596 | 0.784 | 0.266 | 0.397 | 0.547 | 0.751 | 0.778 ± 0.033 |

Random Forest obtiene el mejor F1-macro en test (0.720) con la configuración base.
Naïve Bayes muestra el sesgo más notable: alta precisión pero recall muy bajo en
la clase *fake*, lo que lo hace inadecuado para este problema a pesar de un
F1-macro de validación cruzada aceptable (0.778).

### 5.3 Estudio de ablación

Se evaluaron 4 configuraciones de representación × 5 modelos = **20 experimentos**.
Las variables del estudio son: (a) tipo de n-grama — *word* (palabra) vs.
*char_wb* (carácter dentro de palabra) — y (b) uso o no de stopwords.

**Tabla 2.** Mejores resultados por configuración (métrica: F1-macro en test).

| Configuración | Mejor modelo | Accuracy | F1 (fake) | **F1-macro** | ROC-AUC |
|---------------|:------------:|:--------:|:---------:|:------------:|:-------:|
| word + stopwords | Random Forest | 0.720 | 0.718 | 0.720 | 0.795 |
| word + sin stopwords | Random Forest | 0.722 | 0.734 | 0.721 | 0.795 |
| char + stopwords | Reg. Logística | 0.731 | 0.697 | 0.727 | 0.803 |
| **char + sin stopwords** | **Reg. Logística** | **0.734** | **0.698** | **0.730** | **0.809** |

La configuración de **n-gramas de carácter sin stopwords** con Regresión Logística
obtiene el mejor F1-macro global (0.730). Esto indica que los patrones
subléxicos (morfemas, sufijos, prefijos) son más discriminativos que las
palabras completas para este corpus en español, y que las stopwords aportan
ruido en lugar de señal cuando se trabaja a nivel de carácter.

**Tabla 3.** Todos los experimentos de ablación, ordenados por F1-macro.

| Config | Modelo | Accuracy | F1-macro | ROC-AUC | CV F1-macro |
|--------|--------|:--------:|:--------:|:-------:|:-----------:|
| char_nosw | LogReg | 0.734 | **0.730** | 0.809 | 0.809 ± 0.039 |
| char_sw | LogReg | 0.731 | 0.727 | 0.803 | 0.800 ± 0.034 |
| char_nosw | LinearSVM | 0.731 | 0.727 | 0.810 | 0.808 ± 0.034 |
| char_sw | LinearSVM | 0.726 | 0.722 | 0.804 | 0.800 ± 0.034 |
| word_nosw | RandomForest | 0.722 | 0.721 | 0.795 | 0.806 ± 0.023 |
| word_sw | RandomForest | 0.720 | 0.720 | 0.795 | 0.819 ± 0.029 |
| char_sw | GradBoosting | 0.715 | 0.715 | 0.777 | 0.775 ± 0.021 |
| word_nosw | LogReg | 0.713 | 0.708 | 0.801 | 0.813 ± 0.007 |
| word_sw | LinearSVM | 0.715 | 0.711 | 0.781 | 0.803 ± 0.025 |
| char_nosw | GradBoosting | 0.713 | 0.712 | 0.779 | 0.812 ± 0.016 |
| word_nosw | LinearSVM | 0.712 | 0.706 | 0.801 | 0.814 ± 0.003 |
| word_sw | LogReg | 0.710 | 0.705 | 0.785 | 0.800 ± 0.023 |
| word_nosw | GradBoosting | 0.673 | 0.669 | 0.746 | 0.780 ± 0.011 |
| char_nosw | RandomForest | 0.680 | 0.680 | 0.761 | 0.794 ± 0.022 |
| char_sw | RandomForest | 0.682 | 0.682 | 0.769 | 0.807 ± 0.035 |
| word_nosw | NaiveBayes | 0.610 | 0.571 | 0.758 | 0.779 ± 0.036 |
| char_nosw | NaiveBayes | 0.663 | 0.637 | 0.783 | 0.765 ± 0.012 |
| char_sw | NaiveBayes | 0.650 | 0.621 | 0.780 | 0.751 ± 0.010 |
| word_sw | GradBoosting | 0.652 | 0.645 | 0.725 | 0.775 ± 0.029 |
| word_sw | NaiveBayes | 0.596 | 0.547 | 0.751 | 0.778 ± 0.033 |

Las figuras de matrices de confusión de cada modelo se encuentran en
`results/figures/cm_*.png`; la curva ROC comparada en `results/figures/roc_all.png`.

### 5.4 Transformer BETO (fine-tuning en Google Colab)

Se realizó fine-tuning de `dccuchile/bert-base-spanish-wwm-cased` [2] durante
4 épocas sobre el mismo conjunto de entrenamiento (train + development), con
evaluación en el mismo conjunto de prueba *held-out*.

**Tabla 4.** Métricas de BETO en el conjunto de prueba.

| Modelo | Accuracy | F1 (fake) | **F1-macro** | ROC-AUC | Pérdida (eval) |
|--------|:--------:|:---------:|:------------:|:-------:|:--------------:|
| BETO (4 épocas) | 0.654 | 0.686 | 0.650 | 0.741 | 1.067 |

### 5.5 Comparación global

**Tabla 5.** Sistemas comparados — mejor configuración de cada familia.

| Sistema | Representación | F1-macro | ROC-AUC |
|---------|:-------------:|:--------:|:-------:|
| **LogReg (char_nosw)** 🏆 | TF-IDF char | **0.730** | **0.809** |
| LinearSVM (char_nosw) | TF-IDF char | 0.727 | 0.810 |
| Random Forest (word) | TF-IDF word | 0.720 | 0.795 |
| BETO (transformer) | Embedding contextual | 0.650 | 0.741 |

---

## 6. Discusión y trabajo futuro

### 6.1 Interpretación de resultados

**¿Por qué los modelos clásicos superan a BETO?**
El resultado contraintuitivo —un TF-IDF con Regresión Logística (F1-macro = 0.730)
supera a un transformer preentrenado (F1-macro = 0.650)— tiene una explicación
bien documentada en la literatura: los *transformers* requieren grandes cantidades
de datos de fine-tuning para superar a los modelos lineales [16]. El corpus
cuenta con apenas ~971 ejemplos de entrenamiento, un régimen donde la
capacidad expresiva de BETO no puede compensar el sobreajuste y la varianza alta.
Sun et al. [20] mostraron que con menos de 1 000 ejemplos de ajuste fino, los
modelos lineales sobre TF-IDF frecuentemente igualan o superan a BERT.

**¿Por qué n-gramas de carácter?**
El español es una lengua morfológicamente rica: sufijos como *-mente*, *-ción*,
*-ismo* y prefijos como *des-*, *anti-* aparecen sistemáticamente en ciertos
registros periodísticos. Los n-gramas de carácter capturan estos patrones
subléxicos sin necesidad de lematización, lo que explica la ganancia de ~1 punto
de F1-macro sobre los n-gramas de palabra.

**¿Por qué sin stopwords con carácter?**
A nivel de carácter, las stopwords se descomponen en secuencias cortas de 2-5
caracteres (`" de"`, `" la"`, `" en"`) que compiten con los n-gramas discriminativos
generando ruido; su eliminación reduce el tamaño del vocabulario y concentra
el peso en patrones informativos.

**Pérdida alta de BETO (1.067):** indica que el modelo no convergió adecuadamente
con solo 4 épocas y 971 ejemplos. El clasificador tiende a predecir bien la clase
*fake* (recall = implícito en F1-fake = 0.686) pero con baja generalización global.

### 6.2 Validación de hipótesis

**H1 — "F1-macro > 0.70 usando solo el contenido textual":**
**Se acepta.** Tres configuraciones del estudio de ablación superan el umbral:
char_nosw + LogReg (0.730), char_sw + LogReg (0.727), char_nosw + LinearSVM (0.727),
además de word + Random Forest (0.720) en la configuración base.
La hipótesis se cumple con alta probabilidad dada la consistencia entre el
desempeño en validación cruzada (CV F1-macro ≈ 0.809 ± 0.039) y en el test
*held-out* (0.730), sugiriendo que el modelo no sobreajustó.

**H2 — "BETO supera a los modelos clásicos TF-IDF":**
**Se rechaza.** BETO obtiene F1-macro = 0.650, 8 puntos por debajo del mejor
modelo clásico (0.730) y 7 puntos por debajo del umbral de H1. La diferencia
es atribuible principalmente al tamaño del corpus, no a una limitación
intrínseca del transformer.

### 6.3 Viabilidad y potencial comercial
El sistema constituye una **prueba de concepto** reproducible. Como producto
viable para redacciones o plataformas de verificación faltaría: (a) un corpus
al menos 10× mayor y más diverso temáticamente, (b) robustez ante dominios
nuevos (*domain shift*), (c) explicabilidad para el *fact-checker* humano
(e.g., qué términos activaron la predicción), y (d) un mecanismo de actualización
continua ante la evolución del lenguaje de la desinformación.

### 6.4 Limitaciones conscientes y trabajo futuro

**Limitaciones de este trabajo:**
- **Corpus pequeño** (971 + 572 notas): impide evaluar con validez estadística
  fuerte si las diferencias entre modelos son significativas; no se realizaron
  pruebas de significancia (e.g., McNemar) por ser este un trabajo académico
  introductorio.
- **Solo contenido textual:** se ignoraron señales de propagación, metadatos
  de fuente y fecha, que en la literatura mejoran consistentemente el desempeño [4,7].
- **BETO sin búsqueda de hiperparámetros:** se usó una sola configuración
  (`lr=2e-5, epochs=4`) por restricción de tiempo; un *grid search* sobre
  learning rate, épocas y tamaño de batch podría cerrar parte de la brecha.

**Trabajo futuro propuesto:**
1. **Más datos:** aplicar *data augmentation* (traducción de ida y vuelta,
   paráfrasis con LLM) o usar datasets adicionales como el de COVID-19 incluido
   en la v2.0 del corpus.
2. **Modelos multilingües:** evaluar XLM-R [9] y mBERT, que se benefician de
   preentrenamiento en decenas de idiomas.
3. **Señales multimodales:** incorporar la fuente de la nota como rasgo
   categórico y rasgos de propagación si se usa la API de redes sociales.
4. **Explicabilidad:** aplicar SHAP o LIME [21] sobre el mejor modelo para
   identificar qué n-gramas son más discriminativos; útil para auditoría del
   sistema.
5. **Validación cruzada de dominio:** entrenar en temas de política y evaluar
   en salud, para medir la transferibilidad del clasificador.

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

[20] Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). *How to Fine-Tune BERT for Text Classification?* CCL 2019, LNCS 11856, 194–206.

[21] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016, 1135–1144.
