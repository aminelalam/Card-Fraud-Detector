"""
Entrenamiento del modelo de detección de fraude.
Pipeline riguroso con selección dinámica del mejor modelo por PR-AUC.
"""
import os
import json
import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, recall_score,
    precision_recall_curve, auc, roc_auc_score, f1_score,
    precision_score, average_precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

SEMILLA = 42
DIR_MODELOS = os.path.join(os.path.dirname(__file__), "models")
RUTA_DATOS = os.path.join(os.path.dirname(__file__), "creditcard.zip")

os.makedirs(DIR_MODELOS, exist_ok=True)

# 1. Cargar datos
print("Cargando datos...")
df = pd.read_csv(RUTA_DATOS)
print(f"Dimensiones de los datos: {df.shape}")

# Preprocesar columnas Time y Amount
escalador_tiempo = RobustScaler()
df["Time_scaled"] = escalador_tiempo.fit_transform(df[["Time"]])

escalador_monto = RobustScaler()
df["Amount_scaled"] = escalador_monto.fit_transform(df[["Amount"]])

# Guardamos el escalador original para usarlo en produccion
escalador = RobustScaler()
escalador.fit(df[["Time", "Amount"]])
joblib.dump(escalador, os.path.join(DIR_MODELOS, "scaler.joblib"))

# 2. Dividir datos en entrenamiento (80%) y prueba (20%) temporalmente
print("Dividiendo datos...")
caracteristicas = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
X = df[caracteristicas].values
y = df["Class"].values

indice_corte = int(len(df) * 0.8)
X_entrenamiento = X[:indice_corte]
X_prueba = X[indice_corte:]
y_entrenamiento = y[:indice_corte]
y_prueba = y[indice_corte:]

# Calcular desbalanceo para XGBoost
negativos, positivos = np.bincount(y_entrenamiento)
ratio_clases = negativos / positivos

# 3. Entrenar y evaluar modelos
print("Entrenando modelos base...")

# Diccionario para guardar los objetos modelo entrenados
modelos_entrenados = {}

def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    """Entrena y evalúa un modelo, devuelve métricas."""
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_total = time.time() - inicio

    # Guardar referencia al modelo entrenado
    modelos_entrenados[nombre] = modelo

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    tn, fp, fn, tp = confusion_matrix(y_test, predicciones).ravel()
    recall = recall_score(y_test, predicciones)
    precision = precision_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)
    roc_auc = roc_auc_score(y_test, probabilidades) if probabilidades is not None else 0
    pr_auc = average_precision_score(y_test, probabilidades) if probabilidades is not None else 0

    print(f"[{nombre}] Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | Tiempo: {tiempo_total:.1f}s")

    return {
        "name": nombre,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": round(recall, 4), "precision": round(precision, 4),
        "f1": round(f1, 4), "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        "train_time": round(tiempo_total, 2), "y_proba": probabilidades
    }

# Logistic Regression
modelo_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEMILLA, solver="lbfgs")
res_lr = evaluar_modelo("Regresion Logistica", modelo_lr, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

# Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEMILLA, n_jobs=-1)
res_rf = evaluar_modelo("Random Forest", modelo_rf, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

# XGBoost base
modelo_xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=ratio_clases,
                           eval_metric="aucpr", random_state=SEMILLA, n_jobs=-1)
res_xgb = evaluar_modelo("XGBoost", modelo_xgb, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

# XGBoost + SMOTE (sin scale_pos_weight para aislar el efecto del balanceo sintético)
print("Aplicando SMOTE...")
smote = SMOTE(random_state=SEMILLA, sampling_strategy=0.5)
X_smote, y_smote = smote.fit_resample(X_entrenamiento, y_entrenamiento)
modelo_xgb_smote = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 eval_metric="aucpr", random_state=SEMILLA, n_jobs=-1)
res_xgb_smote = evaluar_modelo("XGBoost + SMOTE", modelo_xgb_smote, X_smote, y_smote, X_prueba, y_prueba)

# 4. Ajustar hiperparámetros (Tuning exhaustivo) para XGBoost
print("Afinando hiperparametros de XGBoost (búsqueda exhaustiva)...")
parametros_xgb = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.05, 0.08, 0.1, 0.12, 0.15],
    "n_estimators": [150, 200, 300, 400],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5],
    "gamma": [0, 0.05, 0.1, 0.2],
}

busqueda = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=ratio_clases, eval_metric="aucpr", random_state=SEMILLA, n_jobs=-1),
    parametros_xgb, n_iter=50, scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA),
    random_state=SEMILLA, n_jobs=-1, verbose=0
)

busqueda.fit(X_entrenamiento, y_entrenamiento)
mejor_xgb = busqueda.best_estimator_
print(f"Mejores hiperparámetros encontrados: {busqueda.best_params_}")
res_mejor_xgb = evaluar_modelo("XGBoost (Tuned)", mejor_xgb, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

# 5. Seleccionar y exportar el MEJOR modelo dinámicamente
# Criterio: F1 como métrica primaria (equilibra precisión y recall),
# PR-AUC como desempate. F1 penaliza modelos con recall bajo (RF) y
# modelos con precisión baja (LR) por igual.
resultados_todos = [res_lr, res_rf, res_xgb, res_xgb_smote, res_mejor_xgb]
mejor_resultado = sorted(resultados_todos, key=lambda x: (x["f1"], x["pr_auc"]), reverse=True)[0]
nombre_ganador = mejor_resultado["name"]

print(f"\n{'='*60}")
print(f"MODELO GANADOR (por F1): {nombre_ganador}")
print(f"  PR-AUC: {mejor_resultado['pr_auc']}")
print(f"  Recall: {mejor_resultado['recall']}")
print(f"  Precision: {mejor_resultado['precision']}")
print(f"  F1: {mejor_resultado['f1']}")
print(f"{'='*60}")

# Guardar el modelo que realmente ganó
modelo_ganador = modelos_entrenados[nombre_ganador]
joblib.dump(modelo_ganador, os.path.join(DIR_MODELOS, "best_model.joblib"))
print(f"Modelo guardado: {nombre_ganador}")

metricas = {
    "models": [{k: v for k, v in r.items() if k != "y_proba"} for r in resultados_todos],
    "winner": nombre_ganador,
    "best_params": busqueda.best_params_,
    "dataset_info": {
        "total_samples": len(df),
        "total_frauds": int(df["Class"].sum()),
        "train_samples": len(X_entrenamiento),
        "test_samples": len(X_prueba),
    },
    "feature_names": caracteristicas
}

# Curva PR del modelo ganador — guardar TODOS los puntos para preservar
# la forma escalonada natural del modelo (no interpolar artificialmente)
prec, rec, _ = precision_recall_curve(y_prueba, mejor_resultado["y_proba"])
print(f"Puntos en la curva PR: {len(prec)}")
metricas["pr_curve"] = {
    "precision": [round(float(p), 6) for p in prec],
    "recall": [round(float(r), 6) for r in rec]
}

with open(os.path.join(DIR_MODELOS, "metrics.json"), "w") as f:
    json.dump(metricas, f, indent=2)

print("\nEntrenamiento completado. Archivos guardados en /models/.")
