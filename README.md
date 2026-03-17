# 🛡️ FraudGuard AI - Sistema Inteligente de Prevención de Fraude

![Python](https://img.shields.io/badge/Python-3.13-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg) ![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg)

Un *pipeline* completo de Machine Learning y *dashboard* analítico interactivo diseñado para detectar transacciones fraudulentas con tarjetas de crédito, optimizando el retorno de inversión (ROI) del negocio.

## 🌟 Puntos Destacados

- **Enfoque de Negocio**: No solo mide precisión técnica, sino **impacto financiero**. El modelo se evalúa por el capital salvado vs. la pérdida potencial.
- **Selección Dinámica de Modelos**: Escoge el modelo campeón evaluando múltiples arquitecturas (Regresión Logística, Random Forest, XGBoost) optimizando el **F1-Score**.
- **Manejo de Desbalanceo Extremo**: Técnicas robustas (`scale_pos_weight`, SMOTE) para abordar un dataset donde el fraude representa solo el 0.17%.
- **Dashboard Interactivo**: Una interfaz limpia, moderna y enfocada a la toma de decisiones construida con Streamlit y Plotly.

## ℹ️ Resumen

FraudGuard AI es la solución integral a la problemática crítica del fraude financiero. Más allá de presentar métricas en el vacío, este proyecto traduce cada *True Positive* en euros recuperados y cada *False Positive* en fricción al cliente evitada.

El componente central es un modelo **XGBoost optimizado** (con hiperparámetros afinados vía búsqueda cruzada estratificada) respaldado por un **Análisis Exploratorio de Datos (EDA)** directamente integrado en el frontend, demostrando qué ocurre exactamente bajo el capó sin perder la perspectiva ejecutiva. 

Este proyecto está diseñado a nivel de un Data Scientist Senior, traduciendo métricas frías en decisiones cuantificables.

## 🚀 Instalación y Uso

Sigue estos pasos para arrancar el proyecto o simular modelos:

### Prerrequisitos
- **Python 3.10+** (Recomendado 3.13)
- Descargar el dataset original desde [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Debes guardar el archivo en la raíz del proyecto comprimido bajo el nombre `creditcard.zip`.

### Pasos

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar los modelos:**
   El pipeline de entrenamiento compite varios modelos, aplica escalado robusto, ajusta hiperparámetros recursivos y guarda automáticamente el ganador ("Campeón"):
   ```bash
   python train_model.py
   ```

3. **Lanzar el Dashboard Operativo:**
   Levanta la interfaz web donde podrás interactuar con las analíticas generadas.
   ```bash
   streamlit run app.py
   ```

## 🧠 Arquitectura y Decisiones de Modelado

Se tomaron decisiones estratégicas agresivas para asegurar que el modelo sea robusto en producción:

*   **Separación Temporal vs Aleatoria**: Para simular la realidad operativa, la separación `train`/`test` (80/20) se realiza cronológicamente. Entrenamos con el pasado para predecir el futuro, evitando de raíz las fugas de información (*data leakage*).
*   **Preprocesamiento Resiliente**: Se utiliza `RobustScaler` basado en el rango intercuartílico, mitigando el enorme efecto enmascarador de los valores extremos presentes naturalmente en los importes de fraude.
*   **F1-Score como Árbitro Principal**: A diferencia de un simple *Accuracy*, *Precision* o *Recall*, el F1-Score garantiza un equilibrio perfecto. Esto es vital en el negocio: no debemos bloquear masivamente tarjetas legítimas generando fricción (Falso Positivo), pero tampoco debemos dejar pasar el ataque (Falso Negativo). El **PR-AUC** actúa como métrica de apoyo indiscutible ante el desbalanceo.
*   **Ajuste y Equilibrio del XGBoost**: Cálculo dinámico del `scale_pos_weight` y ejecución de un `RandomizedSearchCV` estratificado a 5 pliegues para asegurar un entrenamiento equitativo sin sesgos al descubrir al modelo ganador.

## 📊 Vistazo al Dashboard

El monitor directivo en `app.py` expone:
1. **Métricas Financieras de Impacto**: Cálculo del "Riesgo de Exposición Absoluta" frente al "Dinero Salvado por el Modelo".
2. **Análisis Exploratorio Bivariante**: Identificación de la franja horaria crítica (fraudes subyacentes vs volumen general) y distribución de densidad de aportes económicos.
3. **Mapeo Dimensional de Transacciones**: Un *scatter plot* enriquecido submuestreado inteligentemente para ubicar zonas de fraude originadas de un análisis PCA (`V1`, `V2`).
4. **Benchmarking Transparente**: Curva PR (*Precision-Recall*) y comparativa en barras detallando la elección del campeón sobre otros aspirantes.

## 🛠️ Stack Tecnológico

*   **Ciencia de Datos & ML**: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`
*   **Frontend Web**: `streamlit`, `plotly`
*   **Arquitectura de Guardado**: `joblib`, serialización JSON (`metrics.json`)

## 🤝 Contribuir y Feedback

¿Te interesa mejorar la visualización empírica, refinar cómo se evalúan económicamente los falsos positivos o investigar un nuevo clasificador? 

Las aportaciones al ecosistema Open Source son bienvenidas. Por favor **abre una Issue** para iniciar una discusión arquitectónica o envía tu **Pull Request**. Si esto te inspira, ¡déjanos tu feedback y una estrella en el repositorio!

---
✍️ *Este proyecto es una muestra end-to-end de Ingeniería e Inteligencia de Datos enfocado enteramente en un contexto de aplicabilidad corporativa real.*
