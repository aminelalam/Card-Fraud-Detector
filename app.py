"""
Dashboard Empresarial de Detección de Fraude.
Nivel Analítico y Estratégico Senior.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(DIR_RAIZ, "models")
RUTA_DATOS = os.path.join(DIR_RAIZ, "creditcard.zip")

st.set_page_config(page_title="Detección de Fraude", layout="wide", initial_sidebar_state="collapsed")

# ── Sistema de Diseño Bold Typography ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp { background: #0A0A0A; font-family: 'Inter', sans-serif; color: #FAFAFA; }
[data-testid="stHeader"] {background: rgba(10,10,10,0.8);}
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 800; letter-spacing: -0.04em; color: #FAFAFA; }
h1 { font-size: 3.5rem !important; line-height: 1.1 !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 2.5rem !important; border-bottom: 2px solid #262626; padding-bottom: 0.5rem; margin-top: 4rem !important; margin-bottom: 2rem !important;}
h3 { font-size: 1.5rem !important; margin-bottom: 1rem !important;}

.kpi-card { background: transparent; border: 1px solid #262626; padding: 24px; flex: 1; min-width: 200px; }
.kpi-card.highlight { border-top: 3px solid #FF3D00; background: #0F0F0F; }
.kpi-card.success { border-top: 3px solid #14b8a6; background: #0F0F0F; }
.kpi-label { font-size: 12px; font-weight: 600; color: #737373; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;}
.kpi-value { font-size: 42px; font-weight: 800; color: #FAFAFA; line-height: 1; margin: 4px 0; letter-spacing: -0.04em; }

.note-card { background: #0F0F0F; border: 1px solid #262626; padding: 32px; margin-bottom: 16px; }
.note-title { font-weight: 800; color: #FF3D00; margin-bottom: 12px; text-transform: uppercase; font-size: 14px; letter-spacing: 0.1em;}
.note-body { color: #FAFAFA; font-size: 14px; line-height: 1.6; color: #A3A3A3;}

.cm-container { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 2rem;}
.cm-box { border: 1px solid #262626; padding: 20px; background: #0F0F0F;}
.cm-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; font-weight: 600; font-size: 12px; letter-spacing: 0.1em;}
.cm-num { font-size: 32px; font-weight: 800; color: #FAFAFA; }
.cm-desc { font-size: 13px; color: #737373; margin-top: 4px;}

.stButton button { width: 100%; border-radius: 0px !important; background: transparent; border: 1px solid #FAFAFA; color: #FAFAFA; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; padding: 16px 0 !important; }
.stButton button:hover { background: #FAFAFA; color: #0A0A0A; border-color: #FAFAFA; }
</style>""", unsafe_allow_html=True)

# ── Funciones Base ──
@st.cache_data(show_spinner=False)
def cargar_datos_y_metricas():
    df = pd.read_csv(RUTA_DATOS)
    df["Hora"] = (df["Time"] / 3600).astype(int) % 24
    with open(os.path.join(DIR_MODELOS, "metrics.json")) as f:
        metricas = json.load(f)
    return df, metricas

df, metricas = cargar_datos_y_metricas()

# ── Seleccionar modelo campeón dinámicamente ──
nombre_ganador = metricas.get("winner", "")
mejor_modelo_stats = None
for m in metricas["models"]:
    if m["name"] == nombre_ganador:
        mejor_modelo_stats = m
        break
if mejor_modelo_stats is None:
    mejor_modelo_stats = max(metricas["models"], key=lambda x: x["f1"])
    nombre_ganador = mejor_modelo_stats["name"]

# Stats de otros modelos para comparaciones dinámicas
stats_lr = next((m for m in metricas["models"] if "Logistica" in m["name"] or "Logistic" in m["name"]), None)
stats_rf = next((m for m in metricas["models"] if m["name"] == "Random Forest"), None)

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#737373", size=14),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1A1A1A", zeroline=False),
    margin=dict(l=0, r=0, t=40, b=0)
)

# ==============================================================================
# HERO Y KPIs DE NEGOCIO
# ==============================================================================
st.markdown("<h1>SISTEMA DE PREVENCIÓN DE FRAUDE</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.25rem; color: #737373; margin-bottom: 3rem; max-width: 900px; line-height: 1.6;'>Motor inteligente basado en Machine Learning. Analiza patrones históricos para minimizar pérdidas económicas sin fricción para el cliente.</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label" title="Total de operaciones registradas en el histórico auditable.">Operaciones Procesadas</div>
        <div class="kpi-value">{len(df):,}</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''
    <div class="kpi-card highlight">
        <div class="kpi-label" title="Incidencias delictivas confirmadas en el conjunto de datos.">Incidencias Fraudulentas</div>
        <div class="kpi-value" style="color: #FF3D00;">{df["Class"].sum()}</div>
    </div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label" title="Equivalente a 1 incidencia por cada ~578 operaciones legítimas.">Índice de Fraude</div>
        <div class="kpi-value">{(df["Class"].mean() * 100):.3f}%</div>
    </div>''', unsafe_allow_html=True)


# ==============================================================================
# ANÁLISIS EXPLORATORIO (EDA)
# ==============================================================================
st.markdown("<h2>ANÁLISIS EXPLORATORIO</h2>", unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    st.markdown("### Riesgo Histórico por Franja Horaria", help="Superpone el recuento absoluto de fraudes detectados contra la probabilidad porcentual sobre el volumen general operando en esa misma hora.")

    totales_hora = df.groupby("Hora").size().reset_index(name="Transacciones")
    fraudes_hora = df[df["Class"]==1].groupby("Hora").size().reset_index(name="Fraudes")

    df_hora = pd.merge(totales_hora, fraudes_hora, on="Hora", how="left").fillna(0)
    df_hora["Porcentaje"] = (df_hora["Fraudes"] / df_hora["Transacciones"]) * 100

    fig_hora = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hora.add_trace(go.Bar(x=df_hora["Hora"], y=df_hora["Fraudes"], name="Fraudes (Vol.)", marker_color="#262626"), secondary_y=False)
    fig_hora.add_trace(go.Scatter(x=df_hora["Hora"], y=df_hora["Porcentaje"], name="% Riesgo", line=dict(color="#FF3D00", width=3), mode="lines+markers"), secondary_y=True)

    fig_hora.update_layout(**LAYOUT_BASE)
    fig_hora.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0))
    fig_hora.update_yaxes(title_text="Volumen Malicioso", secondary_y=False, showgrid=False)
    fig_hora.update_yaxes(title_text="% Relativo", secondary_y=True, showgrid=True, gridcolor="#1A1A1A", tickformat=".2f")
    st.plotly_chart(fig_hora, use_container_width=True)

with g2:
    st.markdown("### Distribución de Importes (€)", help="Densidad de los importes por clase (≤1000€). Permite visualizar si el fraude se concentra en montos específicos respecto a las transacciones legítimas.")

    legales = df[(df["Class"]==0) & (df["Amount"] <= 1000)]["Amount"].tolist()
    fraudes_dist = df[(df["Class"]==1) & (df["Amount"] <= 1000)]["Amount"].tolist()

    fig_kde = ff.create_distplot([legales, fraudes_dist], ['Legítimas', 'Fraudulentas'], show_hist=False, show_rug=False, colors=["#737373", "#FF3D00"])
    fig_kde.update_layout(**LAYOUT_BASE)
    fig_kde.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1, x=0, font=dict(color="#FAFAFA")))
    fig_kde.update_xaxes(title_text="Importe (€)")
    fig_kde.update_yaxes(title_text="Densidad")
    st.plotly_chart(fig_kde, use_container_width=True)


# ==============================================================================
# MAPA DE TRANSACCIONES (SCATTER PLOT)
# ==============================================================================
st.markdown("<h2>DISTRIBUCIÓN DE TRANSACCIONES</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#737373; margin-bottom: 1rem;'>Representación de las transacciones en el espacio de características anonimizadas del dataset (V1 y V2 son las dos variables con mayor varianza, obtenidas por PCA sobre los datos originales de la tarjeta de crédito). Se observa cómo el fraude se concentra en regiones diferenciadas.</p>", unsafe_allow_html=True)

# Subsamplear para rendimiento (5000 legítimas + todas las fraudulentas)
df_legit_sample = df[df["Class"] == 0].sample(5000, random_state=42)
df_fraud_all = df[df["Class"] == 1]

fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=df_legit_sample["V1"],
    y=df_legit_sample["V2"],
    mode="markers",
    name="Legítimas",
    marker=dict(color="#14b8a6", size=4, opacity=0.4),
    hovertemplate="V1: %{x:.2f}<br>V2: %{y:.2f}<br>Amount: %{customdata:.2f}€<extra>Legítima</extra>",
    customdata=df_legit_sample["Amount"]
))

fig_scatter.add_trace(go.Scatter(
    x=df_fraud_all["V1"],
    y=df_fraud_all["V2"],
    mode="markers",
    name=f"Fraudulentas ({len(df_fraud_all):,} total)",
    marker=dict(color="#FF3D00", size=6, opacity=0.8, line=dict(width=0.5, color="#FAFAFA")),
    hovertemplate="V1: %{x:.2f}<br>V2: %{y:.2f}<br>Amount: %{customdata:.2f}€<extra>Fraude</extra>",
    customdata=df_fraud_all["Amount"]
))

fig_scatter.update_layout(**LAYOUT_BASE)
fig_scatter.update_layout(
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#FAFAFA", size=13)),
    xaxis_title="Variable con mayor varianza (V1 — PCA anonimizado)",
    yaxis_title="Segunda variable con mayor varianza (V2 — PCA anonimizado)",
)
fig_scatter.update_xaxes(showgrid=True, gridcolor="#1A1A1A", title_font=dict(color="#A3A3A3"))
fig_scatter.update_yaxes(showgrid=True, gridcolor="#1A1A1A", title_font=dict(color="#A3A3A3"))
st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# AUDITORÍA DE IMPACTO FINANCIERO
# ==============================================================================
st.markdown("<h2>IMPACTO DE NEGOCIO (DATOS DE VALIDACIÓN)</h2>", unsafe_allow_html=True)

media_fraude = df[df["Class"]==1]["Amount"].mean()
tp = mejor_modelo_stats["tp"]
fn = mejor_modelo_stats["fn"]
fp = mejor_modelo_stats["fp"]
tn = mejor_modelo_stats["tn"]

perdida_potencial = (fn + tp) * media_fraude
capital_salvado = tp * media_fraude
dinero_fugado = fn * media_fraude

col_roi1, col_roi2, col_roi3 = st.columns(3)
with col_roi1:
    st.markdown(f'''
    <div class="kpi-card" style="border: 1px dashed #262626; background: transparent;">
        <div class="kpi-label" title="Capital total en riesgo en el conjunto de test.">Riesgo de Exposición Absoluta</div>
        <div class="kpi-value" style="color: #737373;">{perdida_potencial/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)
with col_roi2:
    st.markdown(f'''
    <div class="kpi-card success">
        <div class="kpi-label" title="Dinero protegido por el modelo (TP × importe medio de fraude).">Dinero Salvado por el Modelo</div>
        <div class="kpi-value" style="color: #14b8a6;">{capital_salvado/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)
with col_roi3:
    st.markdown(f'''
    <div class="kpi-card highlight">
        <div class="kpi-label" title="Fraude no detectado que logró pasar el filtro del modelo (FN × importe medio).">Fraude que nos Han Colado</div>
        <div class="kpi-value" style="color: #FF3D00;">{dinero_fugado/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)


st.markdown("<h2>ARQUITECTURA Y DECISIONES TÉCNICAS</h2>", unsafe_allow_html=True)

m_c1, m_c2 = st.columns([1, 1], gap="large")

with m_c1:
    st.markdown(f"""
    ### Confusion Matrix — {nombre_ganador}
    <div class="cm-container" style="margin-top: 20px;">
        <div class="cm-box" style="border-left: 3px solid #14b8a6;">
            <div class="cm-header">TRUE POSITIVE</div>
            <div class="cm-num">{tp:,}</div>
            <div class="cm-desc">Fraude detectado correctamente. Dinero salvado.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #737373;">
            <div class="cm-header">FALSE POSITIVE</div>
            <div class="cm-num">{fp:,}</div>
            <div class="cm-desc">Transacción legítima bloqueada por precaución. Genera fricción operativa.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #FF3D00;">
            <div class="cm-header">FALSE NEGATIVE</div>
            <div class="cm-num">{fn:,}</div>
            <div class="cm-desc">Fraude no detectado. Pérdida directa para la entidad.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #262626;">
            <div class="cm-header">TRUE NEGATIVE</div>
            <div class="cm-num">{tn:,}</div>
            <div class="cm-desc">Transacciones legítimas clasificadas correctamente.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_c2:
    st.markdown("### Estrategia de Modelado (XGBoost)")

    st.markdown('''
    <div class="note-card">
        <div class="note-title">1. Compensación de Desbalanceo (scale_pos_weight)</div>
        <div class="note-body">Con sólo un 0.17% de fraudes en el dataset, el modelo tendería a clasificar todo como legítimo. El hiperparámetro <code>scale_pos_weight</code> de XGBoost asigna un peso proporcional al ratio de desbalanceo (~578:1), penalizando cada Falso Negativo con mayor intensidad durante el entrenamiento para forzar al modelo a detectar la clase minoritaria.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="note-card">
        <div class="note-title">2. Optimización por F1-Score</div>
        <div class="note-body">En detección de fraude, ni Recall ni Precision por separado son suficientes: un modelo con Recall alto pero Precision baja bloquea cientos de tarjetas legítimas (ej: Regresión Logística). El <strong>F1-Score</strong> (media armónica de Precision y Recall) asegura el mejor equilibrio en el umbral de decisión operativo (0.5), mientras que el <strong>PR-AUC</strong> confirma que el modelo mantiene su robustez en distintos escenarios de riesgo.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="note-card">
        <div class="note-title">3. Validación Cruzada Estratificada</div>
        <div class="note-body">Los hiperparámetros de XGBoost (<code>max_depth</code>, <code>learning_rate</code>, <code>subsample</code>, etc.) se seleccionaron mediante <i>RandomizedSearchCV</i> con 50 combinaciones y validación cruzada estratificada de 5 folds, garantizando que cada fold mantiene la proporción original de fraude (0.17%) para evitar evaluaciones sesgadas.</div>
    </div>
    ''', unsafe_allow_html=True)


# ==============================================================================
# BENCHMARKING DE ARQUITECTURAS ML
# ==============================================================================
st.markdown("<h2 style='margin-top: 4rem !important;'>BENCHMARKING DE ARQUITECTURAS ML</h2>", unsafe_allow_html=True)

p_c1, p_c2 = st.columns([1.5, 1], gap="large")

with p_c1:
    st.markdown("### Comparativa de Modelos (Evaluación en Test)", help="F1-Score (métrica de selección) y PR-AUC de cada modelo evaluado en el conjunto de test (20%).")
    todos_los_modelos = metricas["models"]
    nombres = [m["name"].upper() for m in todos_los_modelos]
    f1_vals = [m["f1"] for m in todos_los_modelos]
    pr_auc_vals = [m["pr_auc"] for m in todos_los_modelos]

    colores_f1 = ["#FF3D00" if m["name"] == nombre_ganador else "#262626" for m in todos_los_modelos]
    colores_prauc = ["#E9532D" if m["name"] == nombre_ganador else "#1A1A1A" for m in todos_los_modelos]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='F1-Score', x=nombres, y=f1_vals, marker_color=colores_f1, text=[f"{v:.2f}" for v in f1_vals], textposition='auto'))
    fig_comp.add_trace(go.Bar(name='PR-AUC', x=nombres, y=pr_auc_vals, marker_color=colores_prauc, text=[f"{v:.2f}" for v in pr_auc_vals], textposition='auto'))

    fig_comp.update_layout(**LAYOUT_BASE)
    fig_comp.update_layout(barmode='group', height=400, legend=dict(orientation="h", yanchor="bottom", y=1.05, font=dict(color="#FAFAFA")))
    fig_comp.update_yaxes(showgrid=True, gridcolor="#1A1A1A")
    st.plotly_chart(fig_comp, use_container_width=True)

with p_c2:
    if "pr_curve" in metricas:
        st.markdown(f"### Curva Precision-Recall ({nombre_ganador})", help="Área bajo la curva del modelo campeón. Cuanto más se acerque a la esquina (1,1), mejor equilibrio entre precisión y detección.")
        pr = metricas["pr_curve"]
        prec_full = pr["precision"]
        rec_full = pr["recall"]

        # Downsamplear preservando forma escalonada real del modelo.
        # Sin esto, 53K+ puntos colapsan visualmente en Plotly.
        pr_sampled = [(rec_full[0], prec_full[0])]
        for i in range(1, len(prec_full)):
            dp = abs(prec_full[i] - pr_sampled[-1][1])
            dr = abs(rec_full[i] - pr_sampled[-1][0])
            if dp > 0.005 or dr > 0.005:
                pr_sampled.append((rec_full[i], prec_full[i]))
        pr_sampled.append((rec_full[-1], prec_full[-1]))
        rec_ds = [p[0] for p in pr_sampled]
        prec_ds = [p[1] for p in pr_sampled]

        fig_pr = go.Figure(go.Scatter(x=rec_ds, y=prec_ds, mode='lines', fill='tozeroy', fillcolor='rgba(255, 61, 0, 0.1)', line=dict(color='#FF3D00', width=3, shape='hv')))
        fig_pr.update_layout(**LAYOUT_BASE)
        fig_pr.update_layout(height=400, xaxis_title="Recall (Fraudes Detectados)", yaxis_title="Precision (Detecciones Correctas)")
        fig_pr.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 1.05])
        fig_pr.update_xaxes(range=[0, 1.05])
        st.plotly_chart(fig_pr, use_container_width=True)

# ── Nota analítica construida dinámicamente desde los datos ──
recall_ganador = mejor_modelo_stats["recall"] * 100
prauc_ganador = mejor_modelo_stats["pr_auc"] * 100
f1_ganador = mejor_modelo_stats["f1"] * 100
fp_ganador = mejor_modelo_stats["fp"]
precision_ganador = mejor_modelo_stats["precision"] * 100

comparativas = []
if stats_lr:
    comparativas.append(f"<b>Regresión Logística</b> obtiene el Recall más alto ({stats_lr['recall']*100:.1f}%), pero su Precision es de sólo {stats_lr['precision']*100:.1f}%, generando {stats_lr['fp']:,} Falsos Positivos. En producción, esto significaría bloquear {stats_lr['fp']:,} tarjetas legítimas para capturar {stats_lr['tp']} fraudes.")
if stats_rf:
    comparativas.append(f"<b>Random Forest</b> tiene la mayor Precision ({stats_rf['precision']*100:.1f}%) y el PR-AUC más alto ({stats_rf['pr_auc']*100:.1f}%), pero su Recall cae al {stats_rf['recall']*100:.1f}%, dejando escapar {stats_rf['fn']} de {stats_rf['fn']+stats_rf['tp']} fraudes reales.")

texto_comparativas = "<br>".join(comparativas)

st.markdown(f"""
<div style="background: #0F0F0F; border: 1px solid #262626; padding: 20px; font-size: 13px; color: #A3A3A3; margin-top: 2rem;">
    <strong style="color: #FAFAFA;">Modelo Campeón: {nombre_ganador}</strong><br>
    Seleccionado por F1-Score ({f1_ganador:.1f}%), que equilibra Recall ({recall_ganador:.1f}%) y Precision ({precision_ganador:.1f}%) con sólo {fp_ganador} Falsos Positivos en el conjunto de test.
    <br><br>
    <strong style="color: #FAFAFA;">¿Por qué no otros modelos?</strong><br>
    {texto_comparativas}
    <br><br>
    <strong style="color: #FAFAFA;">Versiones de XGBoost evaluadas:</strong><br>
    &#x2022; <b>Baseline:</b> XGBoost con <code>scale_pos_weight</code> para compensar el desbalanceo de clases. Sin ajuste adicional de hiperparámetros.<br>
    &#x2022; <b>SMOTE:</b> Datos de entrenamiento aumentados con muestras sintéticas de fraude generadas por interpolación espacial entre vecinos cercanos. No se utilizó <code>scale_pos_weight</code> en esta variante para aislar el efecto del balanceo sintético.<br>
    &#x2022; <b>Tuned:</b> Hiperparámetros optimizados mediante RandomizedSearchCV (50 iteraciones, 5-fold CV, scoring=F1) sobre <code>max_depth</code>, <code>learning_rate</code>, <code>subsample</code>, <code>colsample_bytree</code>, <code>min_child_weight</code> y <code>gamma</code>.
</div>
""", unsafe_allow_html=True)
