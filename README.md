#Mini-proyecto: Clasificación de texto con BERT

Autores: José Luis Realpe M., Alejandra Forero, Santiago Aristizabal, Sandra Orozco Curso: Procesamiento de Lenguaje Natural — Maestría en IA Aplicada, Universidad Icesi. Basado en: notebook guía `text-classification-with-hf.ipynb`.


# Clasificación de emociones en tweets con BERTweet

## Descripción

Este proyecto estudia la clasificación automática de emociones en tweets utilizando **BERTweet**, un modelo Transformer preentrenado específicamente para texto proveniente de Twitter. El objetivo principal es analizar cómo diferentes estrategias de adaptación de BERTweet afectan el desempeño en la clasificación de siete categorías emocionales.

---

## Dataset

Se utiliza **EmoEvent**, un dataset de tweets etiquetados con siete categorías:

- `anger`
- `fear`
- `joy`
- `sadness`
- `disgust`
- `surprise`
- `others`

---

## Tecnologías

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn

---

El proyecto compara diferentes estrategias de adaptación de BERTweet:

| Modelo | BERTweet | Classification Head |
|---|---|---|
| M0 | Congelado | Linear |
| M1 | Congelado | MLP |
| M2 | Fine-tuning | Linear |
| M3 | Fine-tuning | Linear + regularización |

La métrica principal utilizada es **F1 macro**, debido al desbalance entre las clases.

---

## Resultados

| Modelo | F1 macro en test |
|---|---:|
| M0 | 0.349 ± 0.006 |
| M1 | 0.369 ± 0.008 |
| M2 | **0.489 ± 0.008** |
| M3 | 0.435 ± 0.003 |

El modelo se seleccionó utilizando el **F1 macro promedio en validation**. El conjunto de test se utilizó únicamente para la evaluación final.

El notebook también incluye análisis de distribución de clases, preprocesamiento, matrices de confusión, métricas por emoción y rendimiento por evento.

## Estructura

```text
.
├──mini_proyecto_BERT.ipynb
└── README.md
```

## Cómo ejecutar

### Google Colab

La forma recomendada de ejecutar el proyecto es mediante Google Colab.

1. Abrir `mini_proyecto_BERT.ipynb`.
2. Subir el notebook a Google Colab.
3. Seleccionar una GPU en **Runtime → Change runtime type**.
4. Ejecutar las celdas en orden desde el principio.

El notebook instala las dependencias necesarias y descarga el dataset utilizado.

### Ejecución local

Requiere Python 3.x.

Instalar las dependencias:

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn emoji
```

Iniciar Jupyter:

```bash
jupyter notebook
```

Abrir `mini_proyecto_BERT.ipynb` y ejecutar las celdas en orden.

> **Nota:** el entrenamiento de BERTweet requiere recursos computacionales considerables. Para reproducir los experimentos de fine-tuning se recomienda utilizar una GPU.
