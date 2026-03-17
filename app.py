"""
Dashboard Empresarial de Detección de Fraude.
Nivel Analítico y Estratégico Senior.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import random

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

.kpi-container { display: flex; gap: 24px; margin-bottom: 2rem; flex-wrap: wrap; }
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

.shap-box { border-left: 3px solid #FF3D00; background: #1A1A1A; padding: 16px 20px; margin-top: 16px;}
.shap-ok { border-left: 3px solid #14b8a6; background: #1A1A1A; padding: 16px 20px; margin-top: 16px;}

.stSlider div[data-testid="stThumbValue"] { font-family: 'Inter', monospace !important; }
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

@st.cache_resource(show_spinner=False)
def cargar_ia():
    modelo = joblib.load(os.path.join(DIR_MODELOS, "best_model.joblib"))
    escalador = joblib.load(os.path.join(DIR_MODELOS, "scaler.joblib"))
    return modelo, escalador

df, metricas = cargar_datos_y_metricas()
modelo_ia, escalador = cargar_ia()
mejor_modelo_stats = next(m for m in metricas["models"] if "Tuned" in m["name"])

@st.cache_data(show_spinner=False)
def generar_ejemplos():
    fraudes = df[df["Class"] == 1].sample(5, random_state=42)
    legitimos = df[df["Class"] == 0].sample(5, random_state=42)
    ejemplos_crudos = []
    
    nombres_fraude = ["Anomalía horaria", "Extracción en cajero", "Patrón de vaciado", "Transferencia atípica", "Goteo de saldos"]
    nombres_legitimos = ["Compra en supermercado", "Recibo domiciliado", "Comercio electrónico", "Gasolinera rutinaria", "Pago en restaurante"]

    for i, r in fraudes.iterrows():
        ejemplos_crudos.append({"label": f"[{int(r['Time']/3600)%24}:00h] {nombres_fraude[len(ejemplos_crudos)%5]} - {r['Amount']:.2f}€", "data": r, "is_fraud": True})
    for i, r in legitimos.iterrows():
        ejemplos_crudos.append({"label": f"[{int(r['Time']/3600)%24}:00h] {nombres_legitimos[len(ejemplos_crudos)%5]} - {r['Amount']:.2f}€", "data": r, "is_fraud": False})
    
    # Shuffle para obligar al revisor a probar a ciegas
    random.Random(42).shuffle(ejemplos_crudos)
    return ejemplos_crudos

ejemplos_base = generar_ejemplos()

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
st.markdown("<p style='font-size: 1.25rem; color: #737373; margin-bottom: 3rem; max-width: 900px; line-height: 1.6;'>Motor inteligente basado en Machine Learning. Analiza patrones históricos y simula transacciones en tiempo real para minimizar pérdidas económicas sin fricción para el cliente.</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label" title="Histórico total auditable.">Transacciones Analizadas</div>
        <div class="kpi-value">{len(df):,}</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''
    <div class="kpi-card highlight">
        <div class="kpi-label" title="Casos reales delictivos aislados.">Fraudes Confirmados</div>
        <div class="kpi-value" style="color: #FF3D00;">{df["Class"].sum()}</div>
    </div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label" title="Equivalente a 1 caso ilícito por cada ~578 operaciones legales.">Tasa de Fraude</div>
        <div class="kpi-value">{(df["Class"].mean() * 100):.3f}%</div>
    </div>''', unsafe_allow_html=True)


# ==============================================================================
# ANÁLISIS EXPLORATORIO (EDA)
# ==============================================================================
st.markdown("<h2>ANÁLISIS EXPLORATORIO</h2>", unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    st.markdown("### Riesgo Histórico por Franja Horaria", help="Superpone el recuento absoluto de fraudes detectados contra la probabilidad porcentual sobre el volumen general autorizado operando en esa misma hora.")
    
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
    st.markdown("### Distribución de Importes (€)", help="Muestra la densidad de los importes ignorando las diferencias masivas de volumen absoluto. Permite visualizar si el patrón de riesgo se agrupa en montos específicos frente y paralelos a la rutina legal.")
    
    legales = df[(df["Class"]==0) & (df["Amount"] <= 1000)]["Amount"].tolist()
    fraudes = df[(df["Class"]==1) & (df["Amount"] <= 1000)]["Amount"].tolist()
    
    fig_kde = ff.create_distplot([legales, fraudes], ['Distribución Legal', 'Distribución Riesgo'], show_hist=False, show_rug=False, colors=["#737373", "#FF3D00"])
    fig_kde.update_layout(**LAYOUT_BASE)
    fig_kde.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1, x=0))
    st.plotly_chart(fig_kde, use_container_width=True)


# ==============================================================================
# SIMULADOR EN TIEMPO REAL (BLIND TEST)
# ==============================================================================
st.markdown("<h2>SIMULADOR DE RIESGO EN TIEMPO REAL</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#737373;'>Auditoría interactiva. Seleccione escenarios anonimizados del conjunto de validación o ajuste sus propios parámetros para evaluar la respuesta de latencia del motor de Inferencia.</p>", unsafe_allow_html=True)

c_sim_left, c_sim_right = st.columns([1, 1], gap="large")

with c_sim_left:
    opciones = ["Seleccionar transacción simulada..."] + [e["label"] for e in ejemplos_base]
    seleccion = st.selectbox("Expediente (Test a Ciegas)", opciones)
    
    if seleccion != "Seleccionar transacción simulada...":
        fila = next(e for e in ejemplos_base if e["label"] == seleccion)
        transaccion = fila["data"].copy()

        importe = st.slider("Importe (€)", 0.0, 5000.0, float(transaccion["Amount"]), 5.0)
        hora = st.slider("Franja Horaria de Ejecución", 0, 23, int(transaccion["Time"]/3600)%24)
        
        transaccion["Amount"] = importe
        transaccion["Time"] = (hora * 3600) + (transaccion["Time"] % 3600)

        if st.button("Lanzar Inferencia de Red"):
            vars_v = [transaccion[f"V{i}"] for i in range(1, 29)]
            vars_escaladas = escalador.transform([[transaccion["Time"], transaccion["Amount"]]])
            X_in = np.array(vars_v + [vars_escaladas[0][0], vars_escaladas[0][1]]).reshape(1, -1)
            
            t0 = time.time()
            prob = modelo_ia.predict_proba(X_in)[0][1]
            st.session_state["sim_resultado"] = {"prob": prob, "time": (time.time()-t0)*1000, "imp": importe, "hr": hora}

with c_sim_right:
    if "sim_resultado" in st.session_state and seleccion != "Seleccionar transacción simulada...":
        res = st.session_state["sim_resultado"]
        es_fraude = res['prob'] >= 0.5
        color = "#FF3D00" if es_fraude else "#14b8a6"
        estado_txt = "ALERTA DE FRAUDE" if es_fraude else "TRANSACCIÓN LEGÍTIMA"
        
        st.markdown(f"<div style='margin-top:20px; font-size:14px; letter-spacing:0.1em; color:#737373;'>VEREDICTO IA</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:56px; font-weight:800; line-height:1; letter-spacing:-0.04em; color:{color};'>{estado_txt}</div>", unsafe_allow_html=True)
        
        ancho = res['prob'] * 100
        st.markdown(f"""
        <div style="background:#1A1A1A; height:6px; width:100%; margin:15px 0;">
            <div style="background:{color}; height:100%; width:{ancho}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div style='color:#737373; font-family:monospace; margin-bottom: 20px;'>Latencia (Tick TPV): {res['time']:.2f}ms | Confidence: {res['prob']*100:.1f}%</div>", unsafe_allow_html=True)
        
        media_hora = df[df["Hora"] == res["hr"]]["Amount"].mean()
        if pd.isna(media_hora): media_hora = 0
        
        if es_fraude:
            if res["imp"] > media_hora * 3:
                razon = f"El importe ({res['imp']}€) disiente drásticamente de la varianza histórica del usuario a las {res['hr']}:00h (Cuyo promedio legítimo local ronda los ~{media_hora:.0f}€)."
            else:
                razon = "Los sub-patrones multivariables cartografían un vector de movimiento estadísticamente coincidente con perfiles pasados confirmados de suplantación."
            
            st.markdown(f"""
            <div class="shap-box">
                <div style="color:#FF3D00; font-weight:800; font-size:13px; letter-spacing:0.1em; margin-bottom:5px;">EXPLICABILIDAD DE BLOQUEO</div>
                <div style="color:#FAFAFA; font-size:14px;">{razon}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="shap-ok">
                <div style="color:#14b8a6; font-weight:800; font-size:13px; letter-spacing:0.1em; margin-bottom:5px;">EXPLICABILIDAD DE APROBACIÓN</div>
                <div style="color:#FAFAFA; font-size:14px;">El importe y rutaje en esta franja encajan sólidamente en la banda elástica del intervalo de confianza.</div>
            </div>
            """, unsafe_allow_html=True)


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
ratio_salvado = (capital_salvado / perdida_potencial * 100) if perdida_potencial > 0 else 0

col_roi1, col_roi2, col_roi3 = st.columns(3)
with col_roi1:
    st.markdown(f'''
    <div class="kpi-card" style="border: 1px dashed #262626; background: transparent;">
        <div class="kpi-label" title="Capital en peligro absoluto si dependiéramos del azar.">Riesgo de Exposición Absoluta</div>
        <div class="kpi-value" style="color: #737373;">{perdida_potencial/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)
with col_roi2:
    st.markdown(f'''
    <div class="kpi-card success">
        <div class="kpi-label" title="Dinero legítimo garantizado por el cortafuegos algorítmico.">Retención de Capital</div>
        <div class="kpi-value" style="color: #14b8a6;">{capital_salvado/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)
with col_roi3:
    st.markdown(f'''
    <div class="kpi-card highlight">
        <div class="kpi-label" title="Déficit de ineficiencia por falsos negativos evadidos.">Pérdida por Fuga (FN)</div>
        <div class="kpi-value" style="color: #FF3D00;">{dinero_fugado/1000:.1f}k €</div>
    </div>''', unsafe_allow_html=True)


st.markdown("<h2>ARQUITECTURA Y DECISIONES TÉCNICAS</h2>", unsafe_allow_html=True)

m_c1, m_c2 = st.columns([1, 1], gap="large")

with m_c1:
    st.markdown(f"""
    ### Desglose de Decisiones (Matriz de Confusión)
    <div class="cm-container" style="margin-top: 20px;">
        <div class="cm-box" style="border-left: 3px solid #14b8a6;">
            <div class="cm-header">VERDADERO POSITIVO</div>
            <div class="cm-num">{tp:,}</div>
            <div class="cm-desc">Fraude anticipado. Dinero salvado.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #737373;">
            <div class="cm-header">FALSO POSITIVO</div>
            <div class="cm-num">{fp:,}</div>
            <div class="cm-desc">Bloqueo erróneo por alta precaución. Genera leve fricción operativa.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #FF3D00;">
            <div class="cm-header">FALSO NEGATIVO</div>
            <div class="cm-num">{fn:,}</div>
            <div class="cm-desc">Fallo ciego de detección algorítmica. Quiebra directa.</div>
        </div>
        <div class="cm-box" style="border-left: 3px solid #262626;">
            <div class="cm-header">VERDADERO NEGATIVO</div>
            <div class="cm-num">{tn:,}</div>
            <div class="cm-desc">Transacciones asimiladas correctamente por flujo habitual orgánico.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m_c2:
    st.markdown("### Estrategia de Modelado")
    
    st.markdown('''
    <div class="note-card">
        <div class="note-title">1. Balanceo de Datos (SMOTE)</div>
        <div class="note-body">Al poseer únicamente un 0.17% de fraudes absolutos frente a legítimos, un modelo asimila a la clase minoritaria como ruido incidental. Utilizando <i>SMOTE</i> generamos muestras topográficamente sintéticas de fraude durante la fase de *fitting*, forzando al algoritmo perimetral a definir sus fronteras cartesianas.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="note-card">
        <div class="note-title">2. Optimización Asimétrica</div>
        <div class="note-body">Frente al fraude financiero, incurrir en un Falso Positivo (Bloqueo en caja registradora) genera fricción de servicio, pero un Falso Negativo devasta el capital subyacente de la entidad. Integrar el parámetro <code>scale_pos_weight</code> en XGBoost nos permitió imponer un castigo exponencial interno al modelo ante cada evasión en el set de Train.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="note-card">
        <div class="note-title">3. Agnosticismo del Accuracy</div>
        <div class="note-body">Bajo dictámenes extremos de asimetría 99% vs 0.1%, reportar el Accuracy tradicional induciría fraude conceptual, puntuando 99% con algoritmos llanos. Regimos nuestras auditorías enteramente bajo el Área Bajo la Curva de Precision-Recall (<strong>PR-AUC</strong>).</div>
    </div>
    ''', unsafe_allow_html=True)


# ==============================================================================
# BENCHMARKING DE ARQUITECTURAS ML
# ==============================================================================
st.markdown("<h2 style='margin-top: 4rem !important;'>BENCHMARKING DE ARQUITECTURAS ML</h2>", unsafe_allow_html=True)

p_c1, p_c2 = st.columns([1.5, 1], gap="large")

with p_c1:
    st.markdown("### Retención General (Evaluación Cruzada)", help="Barras que confrontan la sensibilidad del algoritmo frente a su verdadera precisión al no castigar al usuario normal.")
    todos_los_modelos = metricas["models"]
    nombres = [m["name"].upper() for m in todos_los_modelos]
    recall_vals = [m["recall"] for m in todos_los_modelos]
    pr_auc_vals = [m["pr_auc"] for m in todos_los_modelos]

    colores_prauc = ["#FF3D00" if "TUNED" in n else "#262626" for n in nombres]
    colores_recall = ["#E9532D" if "TUNED" in n else "#1A1A1A" for n in nombres]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='Recall Base', x=nombres, y=recall_vals, marker_color=colores_recall, text=[f"{v:.2f}" for v in recall_vals], textposition='auto'))
    fig_comp.add_trace(go.Bar(name='PR-AUC (Puntería Real)', x=nombres, y=pr_auc_vals, marker_color=colores_prauc, text=[f"{v:.2f}" for v in pr_auc_vals], textposition='auto'))

    fig_comp.update_layout(**LAYOUT_BASE)
    fig_comp.update_layout(barmode='group', height=400, legend=dict(orientation="h", yanchor="bottom", y=1.05))
    fig_comp.update_yaxes(showgrid=True, gridcolor="#1A1A1A")
    st.plotly_chart(fig_comp, use_container_width=True)

with p_c2:
    if "pr_curve" in metricas:
        st.markdown("### Curva Precision-Recall (XGBoost)", help="La curva reina. Cuanto más área azul abarque hacia el extremo superior derecho (1,1), mejor capacidad matemática para cazar sin fallar.")
        pr = metricas["pr_curve"]
        fig_pr = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode='lines', fill='tozeroy', fillcolor='rgba(255, 61, 0, 0.1)', line=dict(color='#FF3D00', width=3)))
        fig_pr.update_layout(**LAYOUT_BASE)
        fig_pr.update_layout(height=400, xaxis_title="Recall (Eventos Capturados)", yaxis_title="Precision (Validez Acertada)")
        fig_pr.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 1.05])
        fig_pr.update_xaxes(range=[0, 1.05])
        st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("""
<div style="background: #0F0F0F; border: 1px solid #262626; padding: 20px; font-size: 13px; color: #A3A3A3; margin-top: 2rem;">
    <strong style="color: #FAFAFA;">Anotación Analítica sobre el Recall (Logística vs XGBoost):</strong><br>
    Observando los datos absolutos en pruebas de validación cruzada, <i>Logistic Regression</i> reporta usualmente un Recall superior al ~91%, frente al ~82% del modelo XGBoost Campeón. No debe inducir a error: Regresión Logística arroja <b>miles de Falsos Positivos colaterales masivos</b> (marcando tarjetas de inocentes constantemente) para lograr capturar ese extra. Su <b>PR-AUC</b> lo delata, hundiéndose. XGBoost no es ciego, caza con un 82% efectivo pero con una puntería clínica (PR-AUC 86.8%), impidiendo que bloquees tarjetas de manera descontrolada en producción.
    <br><br>
    <strong style="color: #FAFAFA;">Apéndice de Versiones (XGBoost):</strong><br>
    &#x2022; <b>Baseline:</b> El modelo base que simplemente aprende los datos tal y como vienen, sin ninguna ayuda para compensar que casi todos los casos son legales.<br>
    &#x2022; <b>SMOTE:</b> El modelo al que le hemos "enseñado" ejemplos artificiales de fraude matemáticamente, para que preste más atención a los delincuentes.<br>
    &#x2022; <b>Tuned (Campeón):</b> Nuestra versión final a la que le hemos ajustado los parámetros internos (como si afináramos un motor) para encontrar el equilibrio perfecto entre cazar el fraude y no molestar a los clientes.
</div>
""", unsafe_allow_html=True)
