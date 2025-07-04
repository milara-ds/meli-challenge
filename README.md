# 🛍️ Predicción de Ventas de Productos en MercadoLibre

Este proyecto tiene como objetivo **extraer insights claros y accionables dirigidos a un equipo no técnico de Marketing y Negocio**, basándose en el análisis de datos históricos de productos publicados en MercadoLibre. Después de analizar el dataset otorgado se encontró que usando un modelo de machine learning se puede utilizar para anticipar si un producto tendrá al menos una venta, permitiendo apoyar decisiones estratégicas de oferta, visibilidad y contenido.

---

## 📂 Estructura del Proyecto

1. `EDA_to_clean.ipynb`: Limpieza profunda del dataset original
2. `EDA_for_questions.ipynb`: Análisis exploratorio e ingeniería de características
3. `model_experimentation.ipynb`: Entrenamiento del modelo y extracción de explicaciones interpretables

---

## ✅ Descripción de la Solución

### 1. Limpieza del Dataset (`EDA_to_clean.ipynb`)

- Se cargaron 100,000 registros con 26 columnas desde `new_items_dataset.csv`
- Se eliminaron columnas con más del 99% de valores faltantes (`sub_status`)
- Se filtraron fechas inválidas y outliers en precios y cantidades usando el método **IQR**
- Se corrigieron errores en variables categóricas como espacios en blanco
- Resultado: `85,060` registros finales en `new_items_dataset_procesado_v1.csv`

**Decisión clave**: priorizamos confiabilidad y consistencia en los datos, eliminando casos extremos o faltantes que podrían sesgar el análisis o el modelo.

---

### 2. Análisis Exploratorio e Ingeniería de Features (`EDA_for_questions.ipynb`)

- Variable objetivo: `sold` (binaria: vendido/no vendido)
- Se analizaron diferencias entre productos vendidos y no vendidos:
  - Mayor disponibilidad, precio y engagement en productos vendidos
- Se construyeron nuevas variables:
  - Temporales: `post_days`, `weekday`
  - Visuales y textuales: `num_pictures`, `len_title`
  - Categóricas limpias: `shipping_mode`, `seller_loyalty_cat`, etc.
- Se eliminaron columnas irrelevantes o redundantes

**Decisión clave**: reducir complejidad sin perder información relevante para el negocio, generando variables explicables para los equipos de marketing y producto.

---

### 3. Modelado y Explicabilidad (`model_experimentation.ipynb`)

- Se entrenó un modelo de clasificación **Random Forest**
- Se identificaron las 10 variables más importantes para predecir ventas
- Se tradujeron los resultados del modelo a recomendaciones concretas

**Insights accionables para Marketing**:
- **Promocionar productos nuevos**: mayor probabilidad de venta
- **Impulsar interacción temprana** (clics, visitas)
- **Asegurar disponibilidad de inventario**
- **Invertir en fidelidad de vendedores**
- **Optimizar fotos, títulos y precios**

**Decisión clave**: se eligió un modelo explicable para garantizar que los resultados sean comprensibles por perfiles no técnicos.

---

## 🛠️ Cómo ejecutar el proyecto

### 1. Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## File System
```
📁 data/
├── 📁 raw/
│   └── new_items_dataset.csv
├── 📁 processed/
│   ├── new_items_dataset_procesado_v1.csv
│   ├── new_items_dataset_procesado_v1_v2.csv
│   ├── dataset_procesado_featureengineering.csv
│   └── dataset_procesado_featureengineering_v2.csv

📁 notebooks/
├── EDA_to_clean.ipynb
├── EDA_for_questions.ipynb
└── model_experimentation.ipynb

📁 src/
├── 📁 data/
│   └── data_analyzer.py
├── 📁 features/
│   └── feature_engineering.py
└── 📁 models/
    └── model_predictor.py

📁 outputs/
└── (archivos generados por los notebooks)

📄 requirements.txt
📄 README.md
📄 .gitignore
📄 Desafío_técnico_ACQ25.pdf
📄 Reporte_challenge_20250704.pdf
```


## Hacer predicción con Random Forest

Después de entrenar el modelo dentro del notebook "model_experimentation.ipynb" correr el siguiente código, si no se tiene un conjunto de datos para X.
```
result = predictor.predict_random_sample(model_type='rf', random_state=42)
print(f"Probabilidad de clase 1: {result['predicted_proba']:.4f}")
print(f"Etiqueta verdadera: {result['true_label']}")
print("\nFeatures importantes:")
for feature, value in result['features'].items():
    print(f"- {feature}: {value}")
```