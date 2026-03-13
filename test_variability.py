import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, average_precision_score

RUTA_DATOS = os.path.join(os.path.dirname(__file__), "creditcard.csv")

print("Cargando datos para validación múltiple (3 runs)...")
df = pd.read_csv(RUTA_DATOS)

escalador_tiempo = RobustScaler()
df["Time_scaled"] = escalador_tiempo.fit_transform(df[["Time"]])

escalador_monto = RobustScaler()
df["Amount_scaled"] = escalador_monto.fit_transform(df[["Amount"]])

caracteristicas = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
X = df[caracteristicas].values
y = df["Class"].values

resultados_lr = {"recall": [], "pr_auc": []}
resultados_rf = {"recall": [], "pr_auc": []}
resultados_xgb = {"recall": [], "pr_auc": []}

semillas = [42, 100, 999]

for iteracion, semilla in enumerate(semillas):
    print(f"\n--- Iteracion {iteracion + 1} (Seed: {semilla}) ---")
    
    # Dividir datos temporalmente o aleatoriamente. Por hacer honor al script original, temporal o aleatorio?
    # El script original hace corte temporal. Vamos a usar un split aleatorio para ver varianza
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=semilla, stratify=y)
    
    negativos, positivos = np.bincount(y_train)
    ratio_clases = negativos / positivos

    # 1. Regresión Logística
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=semilla, solver="lbfgs")
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    probs_lr = lr.predict_proba(X_test)[:, 1]
    resultados_lr["recall"].append(recall_score(y_test, preds_lr))
    resultados_lr["pr_auc"].append(average_precision_score(y_test, probs_lr))

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=semilla, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    resultados_rf["recall"].append(recall_score(y_test, preds_rf))
    resultados_rf["pr_auc"].append(average_precision_score(y_test, probs_rf))

    # 3. XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=ratio_clases,
                        eval_metric="aucpr", random_state=semilla, n_jobs=-1)
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    probs_xgb = xgb.predict_proba(X_test)[:, 1]
    resultados_xgb["recall"].append(recall_score(y_test, preds_xgb))
    resultados_xgb["pr_auc"].append(average_precision_score(y_test, probs_xgb))

def mostrar_promedios(nombre, diccionario):
    rec_avg = np.mean(diccionario["recall"])
    prauc_avg = np.mean(diccionario["pr_auc"])
    print(f"{nombre} -> Avg Recall: {rec_avg:.4f} | Avg PR-AUC: {prauc_avg:.4f}")

print("\n=== RESULTADOS FINALES PROMEDIO (3 RUNS) ===")
mostrar_promedios("Logistic Regression", resultados_lr)
mostrar_promedios("Random Forest", resultados_rf)
mostrar_promedios("XGBoost", resultados_xgb)
