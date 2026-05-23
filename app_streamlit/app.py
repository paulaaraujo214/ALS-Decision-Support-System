# ============================================================
# app.py — Sistema de Ayuda a la Decisión para ELA
# SAD-ELA · TFG Ingeniería Biomédica 2025-2026
# ============================================================

from pathlib import Path as PathLib
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAD-ELA | Predicción de Progresión",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    base = PathLib(__file__).parent
    modelo  = joblib.load(base / "rf_opt.pkl")
    scaler  = joblib.load(base / "scaler.pkl")
    with open(base / "feature_names.json") as f:
        features = json.load(f)
    return modelo, scaler, features

modelo, scaler, feature_names = cargar_modelo()

# ─────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────
AZUL_OSC   = "#1B3A5C"
AZUL_MED   = "#2E6DA4"
AZUL_CLAR  = "#D6E8F7"
VERDE_OSC  = "#1A5C3A"
VERDE_CLAR = "#EAF5EE"
ROJO_OSC   = "#8B1A1A"
ROJO_CLAR  = "#FBEAEA"
GRIS_BORDE = "#CFD8DC"
GRIS_TEXT  = "#546E7A"
FONDO      = "#F7F9FC"

st.markdown(f"""
<style>

/* ── Fondo y tipografía ── */
.stApp {{ background-color: {FONDO}; }}
html, body, [class*="css"] {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #263238;
}}

/* ── Cabecera ── */
.app-header {{
    background: linear-gradient(135deg, {AZUL_OSC} 0%, {AZUL_MED} 100%);
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    border-radius: 10px;
    margin-bottom: 1.8rem;
}}
.app-header h1 {{
    font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem 0;
    letter-spacing: 0.02em; color: white;
}}
.app-header p {{ font-size: 0.95rem; opacity: 0.85; margin: 0; color: white; }}

/* ── Separadores de sección ── */
.section-title {{
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {AZUL_MED};
    border-bottom: 1px solid {GRIS_BORDE};
    padding-bottom: 0.5rem; margin: 1.8rem 0 1rem 0;
}}

/* ── Cuadros de resultado ── */
.result-lenta {{
    background: {VERDE_CLAR}; border: 1.5px solid #A8D5B5;
    border-left: 6px solid {VERDE_OSC}; border-radius: 8px;
    padding: 1.4rem 1.8rem; margin-bottom: 1rem;
}}
.result-rapida {{
    background: {ROJO_CLAR}; border: 1.5px solid #E8AAAA;
    border-left: 6px solid {ROJO_OSC}; border-radius: 8px;
    padding: 1.4rem 1.8rem; margin-bottom: 1rem;
}}

/* ── Línea de resultado: "estimación de PROGRESIÓN LENTA/RÁPIDA" en una sola línea ── */
.result-headline {{
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    margin: 0 0 0.8rem 0;
    flex-wrap: wrap;
}}
.result-label-inline {{
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: lowercase;
    white-space: nowrap;
}}
.result-heading-inline {{
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    line-height: 1;
    white-space: nowrap;
}}
.result-subtext {{
    font-size: 0.88rem; color: #37474F; margin: 0; line-height: 1.7;
}}

/* ── Alertas ── */
.alerta {{
    background: #FFF8E1; border-left: 4px solid #F9A825;
    border-radius: 4px; padding: 0.6rem 1rem; margin: 0.35rem 0;
    font-size: 0.875rem; color: #4E3B00;
}}

/* ── Cajas SHAP info ── */
.shap-info {{
    background: {AZUL_CLAR}; border: 1px solid #90BEE0; border-radius: 6px;
    padding: 0.9rem 1.2rem; font-size: 0.875rem; color: {AZUL_OSC};
    margin-bottom: 1rem; line-height: 1.6;
}}

/* ── Total ALSFRS calculado ── */
.alsfrs-calc {{
    background: {AZUL_CLAR}; border: 1.5px solid {AZUL_MED}; border-radius: 6px;
    padding: 0.7rem 1.2rem; font-weight: 700; color: {AZUL_OSC};
    text-align: center; margin-top: 0.3rem;
}}

/* ── Disclaimer ── */
.disclaimer {{
    background: rgba(0,0,0,0.03); border: 1px solid {GRIS_BORDE}; border-radius: 6px;
    padding: 0.9rem 1.2rem; font-size: 0.78rem; color: {GRIS_TEXT};
    margin-top: 1.5rem; line-height: 1.5;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background-color: {AZUL_OSC}; }}
[data-testid="stSidebar"] * {{ color: #E3EDF7 !important; }}
[data-testid="stSidebar"] h3 {{
    color: white !important; font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding-bottom: 0.4rem; margin: 1.2rem 0 0.7rem 0;
}}

/* ── Radio buttons: texto oscuro — forzado con máxima especificidad ── */
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label div,
div[data-testid="stRadio"] label {{
    color: {AZUL_OSC} !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}}
div[data-testid="stRadio"] > div > label > div > p {{
    color: {AZUL_OSC} !important;
    font-weight: 600 !important;
}}

/* ── Labels de inputs en oscuro ── */
.stNumberInput label, .stSelectbox label, .stRadio label {{
    color: #37474F !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}}

/* ── Botón principal ── */
div[data-testid="stButton"] > button {{
    background: {AZUL_MED}; color: white; font-weight: 600;
    font-size: 1rem; border: none; border-radius: 6px;
    padding: 0.75rem 2rem; letter-spacing: 0.03em;
}}
div[data-testid="stButton"] > button:hover {{ background: {AZUL_OSC}; }}

/* ── Inputs: texto oscuro y visible sobre fondo azul claro ── */
.stNumberInput input {{
    background-color: {AZUL_CLAR} !important;
    border-color: #90BEE0 !important;
    color: {AZUL_OSC} !important;
    font-weight: 600 !important;
}}
/* Selectbox: contenedor, texto seleccionado */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] > div > div > div {{
    background-color: {AZUL_CLAR} !important;
    border-color: #90BEE0 !important;
    color: {AZUL_OSC} !important;
    font-weight: 600 !important;
}}
/* Texto visible dentro del selectbox */
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] p {{
    color: {AZUL_OSC} !important;
    font-weight: 600 !important;
}}
/* Flecha SVG — mismo color que el texto */
div[data-testid="stSelectbox"] svg path,
div[data-testid="stSelectbox"] svg {{
    fill: {AZUL_OSC} !important;
    stroke: {AZUL_OSC} !important;
    color: {AZUL_OSC} !important;
}}
/* Panel del menú desplegado: fondo azul claro */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {{
    background-color: {AZUL_CLAR} !important;
    border: 1px solid #90BEE0 !important;
}}
/* Opciones individuales */
[data-baseweb="menu"] li,
[role="option"] {{
    background-color: {AZUL_CLAR} !important;
    color: {AZUL_OSC} !important;
    font-weight: 500 !important;
}}
/* Opción al pasar el ratón */
[data-baseweb="menu"] li:hover,
[role="option"]:hover {{
    background-color: #B8D8F0 !important;
}}

/* ── Divisor vertical entre datos demográficos y sitio de inicio ── */
.demo-wrapper {{
    display: flex;
    align-items: flex-start;
    gap: 0;
    margin-bottom: 0.5rem;
}}
.demo-left {{
    flex: 2;
    display: flex;
    gap: 1.5rem;
}}
.demo-divider {{
    width: 1px;
    background-color: {GRIS_BORDE};
    margin: 0 1.8rem;
    align-self: stretch;
    min-height: 80px;
}}
.demo-right {{
    flex: 1;
}}
.campo-error .stNumberInput input,
.campo-error div[data-testid="stSelectbox"] > div > div {{
    border: 2px solid #C62828 !important;
    background-color: #FFF5F5 !important;
}}
.error-msg {{
    color: #C62828; font-size: 0.78rem; font-weight: 600;
    margin-top: 0.15rem;
}}

/* ── Barra de probabilidad ── */
.prob-block {{
    margin: 1rem 0 0.2rem 0;
}}
.prob-label {{
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.4rem;
}}
.prob-bar-track {{
    position: relative; height: 24px; border-radius: 12px;
    background: rgba(0,0,0,0.08); overflow: visible;
}}
.prob-bar-fill {{
    height: 100%; border-radius: 12px;
}}
.prob-bar-marker {{
    position: absolute; top: -4px; height: calc(100% + 8px);
    width: 2px; background: #263238; opacity: 0.45; border-radius: 1px;
}}
.prob-marker-label {{
    position: absolute; top: -20px;
    font-size: 0.7rem; color: {GRIS_TEXT}; white-space: nowrap;
    transform: translateX(-50%);
}}
.prob-pct {{
    font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0 0.1rem 0; line-height: 1;
}}
.prob-caption {{
    font-size: 0.82rem; color: #37474F; margin: 0; line-height: 1.5;
}}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Sistema de Ayuda a la Decisión — ELA</h1>
    <p>Predicción de progresión funcional en Esclerosis Lateral Amiotrófica
    basada en aprendizaje automático · Repositorio PRO-ACT</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Modelo")
    st.markdown("""
    **Algoritmo:** Random Forest optimizado  
    **AUC-ROC:** 0.632  
    **Cohorte:** PRO-ACT · n = 2.219  
    **Variable objetivo:** pendiente ALSFRS-R < −1 punto/mes
    """)

    st.markdown("### Umbral de decisión")
    umbral = st.slider(
        "Umbral de probabilidad (τ)",
        min_value=0.20, max_value=0.80, value=0.45, step=0.01,
        help="Punto de corte para clasificar como progresión rápida"
    )
    st.markdown(f"Umbral seleccionado: **τ = {umbral:.2f}**")

    st.markdown("### Guía clínica de umbrales")
    st.markdown("""
    **τ = 0.30 — Cribado**  
    Maximiza la detección de casos graves. Indicado cuando no detectar un progresor rápido tiene consecuencias muy costosas —por ejemplo en la primera visita o al planificar cuidados paliativos—. Genera más alertas, incluyendo algunas en pacientes que en realidad evolucionarán lentamente.

    ---

    **τ = 0.45 — Equilibrado** *(recomendado)*  
    Equilibra la detección de casos reales con la minimización de alertas innecesarias. Apropiado para uso clínico general cuando se quiere optimizar ambos tipos de error simultáneamente.

    ---

    **τ = 0.50 — Conservador**  
    Solo clasifica como progresor rápido cuando la confianza del modelo es alta. Útil para confirmar una sospecha clínica previa o cuando las consecuencias de una alarma falsa son especialmente relevantes.
    """)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8BAFCC; line-height:1.5;">
    Prototipo de investigación. No utilizar para decisiones clínicas reales
    sin validación prospectiva y aprobación regulatoria (EU AI Act).
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FORMULARIO
# ─────────────────────────────────────────────────────────────

# ── Datos demográficos ──
st.markdown('<div class="section-title">Datos demográficos</div>', unsafe_allow_html=True)

# Usamos 5 columnas: [edad] [sexo] [separador visual estrecho] [sitio de inicio]
col1, col2, col_sep, col3 = st.columns([1.2, 1.2, 0.05, 1.4])

with col1:
    Age = st.number_input(
        "Edad (años)",
        min_value=0, max_value=90, value=0,
        help="Edad en el momento del diagnóstico o aleatorización al ensayo."
    )

with col2:
    sex_sel = st.radio(
        "Sexo biológico",
        options=["Hombre", "Mujer"],
        index=0,
        horizontal=True
    )
    Sex_val = 1 if sex_sel == "Hombre" else 0

with col_sep:
    # Línea vertical via HTML — ocupa toda la altura de la fila
    st.markdown(
        f'<div style="width:1px; background:{GRIS_BORDE}; min-height:110px; '
        f'margin-top:1.8rem;"></div>',
        unsafe_allow_html=True
    )

with col3:
    onset_sel = st.selectbox(
        "Sitio de inicio de los síntomas",
        options=["— Seleccione —", "Extremidades", "Bulbar", "Espinal",
                 "Mixto (Bulbar y Extremidades)", "Otro", "Desconocido"],
    )
    onset_limb        = 1 if onset_sel == "Extremidades" else 0
    onset_bulbar      = 1 if onset_sel == "Bulbar" else 0
    onset_spine       = 1 if onset_sel == "Espinal" else 0
    onset_limb_bulbar = 1 if onset_sel == "Mixto (Bulbar y Extremidades)" else 0
    onset_other       = 1 if onset_sel == "Otro" else 0
    onset_unknown     = 1 if onset_sel == "Desconocido" else 0

# ── Escala ALSFRS-R ──
st.markdown('<div class="section-title">Escala ALSFRS-R basal (primera visita)</div>',
            unsafe_allow_html=True)
st.markdown(
    "<small style='color:#546E7A;'>Puntuación de cada dominio en la primera visita. "
    "Rango de cada dominio: 0–12 puntos. "
    "La puntuación total se calcula automáticamente como la suma de los cuatro dominios.</small>",
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    basal_bulbar = st.number_input(
        "Dominio bulbar (0-12)", 0, 12, 0,
        help="Suma de: Q1 Habla + Q2 Salivación + Q3 Deglución. "
             "Puntuación 0 = pérdida total de función, 4 = función normal por ítem."
    )
with c2:
    basal_motor_fino = st.number_input(
        "Motor fino (0-12)", 0, 12, 0,
        help="Suma de: Q4 Escritura + Q5 Corte de alimentos + Q6 Vestido e higiene."
    )
with c3:
    basal_motor_grueso = st.number_input(
        "Motor grueso (0-12)", 0, 12, 0,
        help="Suma de: Q7 Giro en cama + Q8 Caminar + Q9 Subir escaleras."
    )
with c4:
    basal_respiratorio = st.number_input(
        "Respiratorio (0-12)", 0, 12, 0,
        help="Suma de: R1 Disnea + R2 Ortopnea + R3 Insuficiencia respiratoria."
    )
with c5:
    alsfrs_basal_total = (
        basal_bulbar + basal_motor_fino + basal_motor_grueso + basal_respiratorio
    )
    st.markdown(
        f'<div class="alsfrs-calc">'
        f'Total ALSFRS-R<br>'
        f'<span style="font-size:2rem;">{alsfrs_basal_total}</span>'
        f'<span style="font-size:0.85rem; font-weight:400;"> / 48</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Función respiratoria y peso ──
st.markdown('<div class="section-title">Función respiratoria y estado nutricional</div>',
            unsafe_allow_html=True)

col_fvc, col_peso = st.columns(2)
with col_fvc:
    fvc_basal = st.number_input(
        "Capacidad Vital Forzada basal — FVC (litros)",
        min_value=0.0, max_value=8.0, value=0.0, step=0.1,
        help="Volumen de aire exhalado en una espiración máxima forzada. "
             "Valores de referencia en adultos sanos: 3–5 L. "
             "Es el predictor respiratorio de referencia en ELA."
    )
with col_peso:
    weight_basal = st.number_input(
        "Peso corporal basal (kg)",
        min_value=0.0, max_value=200.0, value=0.0, step=0.5,
        help="Peso en la primera visita. Indicador indirecto del estado nutricional "
             "y de la preservación de masa muscular. Mediana de referencia: 73 kg."
    )

# ── Analítica de sangre ──
st.markdown('<div class="section-title">Analítica de sangre basal</div>',
            unsafe_allow_html=True)
st.markdown(
    "<small style='color:#546E7A;'>Valores del análisis más próximo al diagnóstico.</small>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    creatinine = st.number_input(
        "Creatinina (µmol/L)",
        min_value=0.0, max_value=500.0, value=0.0, step=1.0,
        help="Marcador subrogado de masa muscular. "
             "Valores normales: 44–97 µmol/L (mujeres), 62–115 µmol/L (hombres)."
    )
with c2:
    albumin = st.number_input(
        "Albúmina (g/L)",
        min_value=0.0, max_value=60.0, value=0.0, step=0.5,
        help="Proteína plasmática indicadora del estado nutricional. "
             "Valores normales: 35–50 g/L."
    )
with c3:
    creatine_kinase = st.number_input(
        "Creatina quinasa (U/L)",
        min_value=0.0, max_value=5000.0, value=0.0, step=1.0,
        help="Enzima liberada por la degradación muscular. "
             "Valores normales: 24–195 U/L."
    )
with c4:
    uric_acid_input = st.number_input(
        "Ácido úrico (µmol/L)",
        min_value=0.0, max_value=900.0, value=0.0, step=1.0,
        help="Antioxidante endógeno con posible efecto neuroprotector en ELA. "
             "Valores normales: 149–369 µmol/L (mujeres), 208–428 µmol/L (hombres)."
    )

# ─────────────────────────────────────────────────────────────
# BOTÓN DE ANÁLISIS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
boton = st.button("Calcular riesgo de progresión", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────────────────────
if boton:

    # ── Validación: todos los campos obligatorios ──
    # Los dominios ALSFRS-R sí pueden valer 0 clínicamente (caso severo).
    errores = []

    if Age == 0:
        errores.append("Edad")
    if onset_sel == "— Seleccione —":
        errores.append("Sitio de inicio de los síntomas")
    if fvc_basal == 0.0:
        errores.append("FVC basal")
    if weight_basal == 0.0:
        errores.append("Peso corporal basal")
    if creatinine == 0.0:
        errores.append("Creatinina")
    if albumin == 0.0:
        errores.append("Albúmina")
    if creatine_kinase == 0.0:
        errores.append("Creatina quinasa")
    if uric_acid_input == 0.0:
        errores.append("Ácido úrico")

    if errores:
        st.markdown(
            f"""
            <div style="background:#FBEAEA; border: 2px solid #C62828; border-radius:8px;
                        padding:1rem 1.4rem; margin-bottom:1rem;">
                <b style="color:#8B1A1A;">Campos obligatorios sin completar:</b>
                <ul style="margin:0.5rem 0 0 1rem; color:#8B1A1A; font-size:0.9rem;">
                    {''.join(f'<li>{e}</li>' for e in errores)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    # ── Resolver ácido úrico ──
    uric_acid         = uric_acid_input
    uric_acid_unknown = 0

  # ── Construir vector de entrada (orden exacto del feature_names.json) ──
    datos_paciente = {
    'Age'                : Age,
    'Sex'                : Sex_val,
    'alsfrs_basal_total' : alsfrs_basal_total,
    'basal_bulbar'       : basal_bulbar,
    'basal_motor_fino'   : basal_motor_fino,
    'basal_motor_grueso' : basal_motor_grueso,
    'basal_respiratorio' : basal_respiratorio,
    'onset_limb'         : onset_limb,
    'onset_bulbar'       : onset_bulbar,
    'onset_spine'        : onset_spine,
    'onset_limb_bulbar'  : onset_limb_bulbar,
    'onset_other'        : onset_other,
    'onset_unknown'      : onset_unknown,
    'fvc_basal'          : fvc_basal,
    'weight_basal'       : weight_basal,
    'albumin'            : albumin,
    'creatine_kinase'    : creatine_kinase,
    'creatinine'         : creatinine,
    'uric_acid'          : uric_acid,
    'uric_acid_unknown'  : uric_acid_unknown,
    }

    df_input  = pd.DataFrame([datos_paciente])[feature_names]
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_names)

    prob_rapida  = modelo.predict_proba(df_scaled)[0, 1]
    prob_lenta   = 1 - prob_rapida
    prediccion   = 1 if prob_rapida >= umbral else 0

    # ── Cabecera del resultado ──
    st.markdown('<div class="section-title">Resultado del análisis</div>',
                unsafe_allow_html=True)

    # ── Cuadro de resultado ──
    pct_rapida = int(round(prob_rapida * 100))
    pct_lenta  = int(round(prob_lenta  * 100))
    marker_pct = int(round(umbral * 100))

    if prediccion == 1:
        texto_explicacion = (
            f"Basándose en los datos introducidos, el modelo estima que este paciente "
            f"perderá más de 1 punto mensual en la escala ALSFRS-R, lo que indica una "
            f"<b>progresión funcional rápida</b>. Esto significa que su capacidad funcional "
            f"podría deteriorarse de forma significativa en los próximos meses, por lo que "
            f"se recomienda una monitorización estrecha, valoración respiratoria precoz "
            f"y planificación anticipada de cuidados."
        )
        st.markdown(f"""
        <div class="result-rapida">
            <div class="result-headline">
                <span class="result-label-inline" style="color:{ROJO_OSC};">estimación de</span>
                <span class="result-heading-inline" style="color:{ROJO_OSC};">PROGRESIÓN RÁPIDA</span>
            </div>
            <p class="result-subtext">{texto_explicacion}</p>
            <div class="prob-block">
                <div class="prob-label" style="color:{ROJO_OSC};">Probabilidad de progresión rápida</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{pct_rapida}%; background:{ROJO_OSC};"></div>
                    <div class="prob-bar-marker" style="left:{marker_pct}%;">
                        <span class="prob-marker-label" style="left:{marker_pct}%;">umbral τ={umbral:.2f}</span>
                    </div>
                </div>
                <p class="prob-pct" style="color:{ROJO_OSC};">{pct_rapida}%</p>
                <p class="prob-caption">
                    El modelo asigna a este paciente una probabilidad del <b>{pct_rapida}%</b> de progresar
                    rápidamente (pérdida &gt; 1 punto/mes en ALSFRS-R). El umbral de decisión está fijado
                    en <b>{marker_pct}%</b>: por encima de ese valor el caso se clasifica como progresión rápida.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        texto_explicacion = (
            f"Basándose en los datos introducidos, el modelo estima que este paciente "
            f"perderá menos de 1 punto mensual en la escala ALSFRS-R, lo que indica una "
            f"<b>progresión funcional lenta</b>. Se espera un deterioro más gradual "
            f"de la capacidad funcional. Aun así, se recomienda seguimiento periódico "
            f"dado el carácter progresivo e impredecible de la ELA."
        )
        st.markdown(f"""
        <div class="result-lenta">
            <div class="result-headline">
                <span class="result-label-inline" style="color:{VERDE_OSC};">estimación de</span>
                <span class="result-heading-inline" style="color:{VERDE_OSC};">PROGRESIÓN LENTA</span>
            </div>
            <p class="result-subtext">{texto_explicacion}</p>
            <div class="prob-block">
                <div class="prob-label" style="color:{VERDE_OSC};">Probabilidad de progresión rápida</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{pct_rapida}%; background:{VERDE_OSC};"></div>
                    <div class="prob-bar-marker" style="left:{marker_pct}%;">
                        <span class="prob-marker-label" style="left:{marker_pct}%;">umbral τ={umbral:.2f}</span>
                    </div>
                </div>
                <p class="prob-pct" style="color:{VERDE_OSC};">{pct_rapida}%</p>
                <p class="prob-caption">
                    El modelo asigna a este paciente una probabilidad del <b>{pct_rapida}%</b> de progresar
                    rápidamente (pérdida &gt; 1 punto/mes en ALSFRS-R). El umbral de decisión está fijado
                    en <b>{marker_pct}%</b>: por debajo de ese valor el caso se clasifica como progresión lenta.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Alertas clínicas ──
    alertas = []
    if fvc_basal < 2.5:
        alertas.append("FVC inferior a 2,5 L: compromiso respiratorio severo desde el diagnóstico. Valorar soporte ventilatorio precoz.")
    if alsfrs_basal_total < 34:
        alertas.append("Puntuación ALSFRS-R inferior a 34: afectación funcional moderada-severa en la primera visita.")
    if basal_bulbar < 8:
        alertas.append("Dominio bulbar deteriorado (< 8/12): valorar disfagia, riesgo aspirativo y soporte nutricional.")
    if basal_respiratorio < 9:
        alertas.append("Dominio respiratorio reducido (< 9/12): monitorización respiratoria estrecha recomendada.")
    if onset_bulbar == 1:
        alertas.append("Inicio bulbar: el modelo tiene menor precisión en este subgrupo (AUC 0,575 frente a 0,629 global). Interpretar con cautela.")
    if Age >= 65:
        alertas.append("Edad igual o superior a 65 años: el modelo tiene mayor capacidad discriminativa en este grupo (AUC 0,662).")

    if alertas:
        st.markdown("**Consideraciones clínicas destacadas:**")
        for a in alertas:
            st.markdown(f'<div class="alerta">{a}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # EXPLICACIÓN SHAP
    # ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Factores que han influido en esta predicción</div>',
                unsafe_allow_html=True)

    with st.expander("Cómo interpretar este gráfico", expanded=False):
        st.markdown("""
        El gráfico muestra qué variables de este paciente han influido en la predicción y en qué sentido:

        - Las **barras rojas** corresponden a variables que **aumentan** el riesgo de progresión rápida en este paciente.
        - Las **barras azules** corresponden a variables que lo **reducen** (factores protectores).
        - El **tamaño de cada barra** refleja la magnitud de esa influencia: cuanto más larga, mayor impacto.
        - Las variables aparecen ordenadas de mayor a menor importancia relativa para este paciente concreto.

        Este análisis es específico de cada caso y puede variar entre pacientes con perfiles similares.
        """)

    with st.spinner("Calculando la explicación..."):
        try:
            explainer   = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(df_scaled)

            if isinstance(shap_values, list):
                sv       = shap_values[1][0]
                base_val = float(explainer.expected_value[1])
            elif shap_values.ndim == 3:
                sv       = shap_values[0, :, 1]
                base_val = float(explainer.expected_value[1])
            else:
                sv       = shap_values[0]
                base_val = float(explainer.expected_value)

            etiquetas = {
                'alsfrs_basal_total' : 'ALSFRS-R total',
                'basal_bulbar'       : 'Dominio bulbar',
                'basal_motor_fino'   : 'Motor fino',
                'basal_motor_grueso' : 'Motor grueso',
                'basal_respiratorio' : 'Dominio respiratorio',
                'Age'                : 'Edad',
                'Sex'                : 'Sexo',
                'fvc_basal'          : 'FVC basal (L)',
                'weight_basal'       : 'Peso (kg)',
                'creatinine'         : 'Creatinina',
                'uric_acid'          : 'Ácido úrico',
                'creatine_kinase'    : 'Creatina quinasa',
                'albumin'            : 'Albúmina',
                'uric_acid_unknown'  : 'Ác. úrico no disponible',
                'onset_bulbar'       : 'Inicio bulbar',
                'onset_limb'         : 'Inicio en extremidades',
                'onset_limb_bulbar'  : 'Inicio mixto',
                'onset_other'        : 'Inicio otro',
                'onset_spine'        : 'Inicio espinal',
                'onset_unknown'      : 'Inicio desconocido',
            }

            vars_categoricas = {
                'onset_bulbar', 'onset_limb', 'onset_limb_bulbar', 'onset_other',
                'onset_spine', 'onset_unknown', 'Sex', 'uric_acid_unknown'
            }

            valores_orig = df_input.iloc[0].to_dict()
            shap_df = pd.DataFrame({
                'feature'   : feature_names,
                'shap_value': sv,
                'raw_value' : [valores_orig[f] for f in feature_names]
            })
            shap_df['abs_shap'] = shap_df['shap_value'].abs()
            shap_df = shap_df.sort_values('abs_shap', ascending=True).tail(14)

            # ── Gráfico ──
            fig_s, ax_s = plt.subplots(figsize=(9, 6))
            fig_s.patch.set_facecolor(FONDO)
            ax_s.set_facecolor(FONDO)

            colores_barras = ["#C62828" if v > 0 else "#1565C0"
                              for v in shap_df['shap_value']]
            y_pos = list(range(len(shap_df)))

            labels = []
            for _, row in shap_df.iterrows():
                f   = row['feature']
                val = row['raw_value']
                if f in vars_categoricas:
                    labels.append(etiquetas.get(f, f))
                else:
                    labels.append(f"{etiquetas.get(f, f)} = {val:.1f}")

            bars = ax_s.barh(
                y_pos, shap_df['shap_value'].values,
                color=colores_barras, height=0.6,
                edgecolor="white", linewidth=0.5, zorder=2
            )

            ax_s.axvline(0, color="#263238", linewidth=1.0, zorder=3)
            ax_s.set_yticks(y_pos)
            ax_s.set_yticklabels(labels, fontsize=9, color="#263238")

            xmax   = max(abs(shap_df['shap_value'].max()),
                         abs(shap_df['shap_value'].min()))
            margin = xmax * 0.18
            ax_s.set_xlim(-xmax - margin * 3.5, xmax + margin * 3.5)

            for bar, val in zip(bars, shap_df['shap_value']):
                offset = margin * 0.5
                if val >= 0:
                    ax_s.text(val + offset, bar.get_y() + bar.get_height() / 2,
                              f"{val:+.3f}", va='center', ha='left',
                              fontsize=8, color="#263238")
                else:
                    ax_s.text(val - offset, bar.get_y() + bar.get_height() / 2,
                              f"{val:+.3f}", va='center', ha='right',
                              fontsize=8, color="#263238")

            ax_s.set_xlabel(
                "Influencia sobre el riesgo de progresión rápida  "
                "(positivo = aumenta el riesgo · negativo = lo reduce)",
                fontsize=9, color=GRIS_TEXT
            )
            ax_s.set_title(
                "Variables ordenadas por su influencia en la predicción de este paciente",
                fontsize=10, fontweight="600", color=AZUL_OSC, pad=12
            )
            ax_s.spines[['top', 'right']].set_visible(False)
            ax_s.spines[['left', 'bottom']].set_color(GRIS_BORDE)
            ax_s.tick_params(axis='x', colors=GRIS_TEXT, labelsize=8.5)
            ax_s.grid(axis='x', color=GRIS_BORDE, linestyle='--', alpha=0.5, zorder=1)

            parche_r = mpatches.Patch(color="#C62828", label="Aumenta el riesgo")
            parche_b = mpatches.Patch(color="#1565C0", label="Reduce el riesgo")
            ax_s.legend(handles=[parche_r, parche_b],
                        loc='lower right', fontsize=8.5, framealpha=0.7)

            plt.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close()

            # ── Resumen textual dinámico ──
            top_pos = (shap_df[shap_df['shap_value'] > 0]
                       .sort_values('shap_value', ascending=False).head(3))
            top_neg = (shap_df[shap_df['shap_value'] < 0]
                       .sort_values('shap_value', ascending=True).head(3))

            col_i1, col_i2 = st.columns(2)
            with col_i1:
                if not top_pos.empty:
                    items = ", ".join([etiquetas.get(f, f) for f in top_pos['feature']])
                    st.markdown(
                        f'<div class="shap-info">'
                        f'<b>Factores que aumentan el riesgo en este paciente:</b><br>{items}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with col_i2:
                if not top_neg.empty:
                    items = ", ".join([etiquetas.get(f, f) for f in top_neg['feature']])
                    st.markdown(
                        f'<div class="shap-info">'
                        f'<b>Factores que reducen el riesgo en este paciente:</b><br>{items}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"No se pudo calcular la explicación: {e}")

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
    <b>Aviso:</b> Este sistema es un prototipo de investigación desarrollado sobre el repositorio PRO-ACT.
    Su capacidad discriminativa (AUC-ROC = 0,632) es moderada y coherente con la literatura en predicción
    de progresión en ELA a partir de datos basales. No debe utilizarse para la toma de decisiones clínicas
    reales sin validación prospectiva en cohorte independiente, revisión por comité de ética y aprobación
    regulatoria conforme al EU AI Act y la guía FDA sobre Software as a Medical Device.
    El juicio clínico del neurólogo es siempre el criterio principal.
    </div>
    """, unsafe_allow_html=True)
