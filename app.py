# -*- coding: utf-8 -*-
"""
Análisis de Supervivencia de Clientes
Random Survival Forest · Dashboard Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# FIX 1: eliminado 'import matplotlib.patches as mpatches' — nunca se usaba

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retención de Clientes · RSF",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --navy:   #0D1B2A;
    --teal:   #00C9A7;
    --amber:  #FFB347;
    --danger: #FF5555;
    --card:   #132337;
    --border: #1E3450;
    --text:   #E4EDF7;
    --muted:  #7A94B0;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* Background */
.stApp {
    background: var(--navy);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A1520;
    border-right: 1px solid var(--border);
}

/* Header */
.page-header {
    background: linear-gradient(
        135deg,
        #0A1F35 0%,
        #0D2E4A 100%
    );

    border: 1px solid var(--border);
    border-radius: 12px;

    padding: 1.8rem 2rem;
    margin-bottom: 1.6rem;

    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.page-header .icon {
    font-size: 2.6rem;
}

.page-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    margin: 0;
}

.page-header p {
    margin: 0.25rem 0 0;
    color: var(--muted);
    font-size: 0.9rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;

    padding: 1.2rem 1.5rem;

    flex: 1;
    min-width: 140px;
}

.metric-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--teal);
}

.metric-card .sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* Section title */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;

    color: var(--muted);

    margin: 0.4rem 0 1rem;

    border-left: 3px solid var(--teal);
    padding-left: 0.7rem;
}

/* Risk badges */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;

    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
}

.risk-high {
    background: rgba(255,85,85,0.15);
    color: #FF5555;
    border: 1px solid rgba(255,85,85,0.35);
}

.risk-medium {
    background: rgba(255,179,71,0.15);
    color: #FFB347;
    border: 1px solid rgba(255,179,71,0.35);
}

.risk-low {
    background: rgba(0,201,167,0.15);
    color: #00C9A7;
    border: 1px solid rgba(0,201,167,0.35);
}

/* Info box */
.info-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;

    padding: 1.1rem 1.4rem;
    margin-top: 1rem;

    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
}

.info-box strong {
    color: var(--text);
}

/* Button */
.stButton > button {
    background: var(--teal);
    color: #0D1B2A;

    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;

    border: none;
    border-radius: 8px;

    width: 100%;
    padding: 0.55rem 1.6rem;
}

.stButton > button:hover {
    opacity: 0.85;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
NAVY   = "#0D1B2A"
CARD   = "#132337"
TEAL   = "#00C9A7"
AMBER  = "#FFB347"
DANGER = "#FF5555"
MUTED  = "#7A94B0"
TEXT   = "#E4EDF7"


def apply_dark_theme():

    plt.rcParams.update({
        "figure.facecolor": CARD,
        "axes.facecolor": CARD,
        "axes.edgecolor": "#1E3450",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#1E3450",
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
    })


apply_dark_theme()


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generar_datos(n=250, seed=42):

    np.random.seed(seed)

    df = pd.DataFrame({
        "edad": np.random.randint(22, 65, n),
        "antiguedad": np.random.randint(1, 48, n),
        "plan": np.random.choice(["Basico", "Premium"], n),
        "incidencias": np.random.randint(0, 6, n),
        "uso_horas": np.random.randint(5, 100, n),
        "pagos_atrasados": np.random.randint(0, 5, n),
    })

    riesgo = (
        0.25 * df["incidencias"]
        + 0.35 * df["pagos_atrasados"]
        - 0.015 * df["uso_horas"]
        - 0.01 * df["antiguedad"]
    )

    prob = 1 / (1 + np.exp(-riesgo))

    df["evento"] = np.random.binomial(
        1,
        prob
    ).astype(bool)

    # FIX 2: garantizar tiempos estrictamente positivos (sksurv lo requiere)
    df["tiempo"] = np.maximum(1, np.random.randint(1, 48, n))

    return df


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def entrenar_modelo(n_estimators=100):

    datos = generar_datos()

    X = pd.get_dummies(
        datos.drop(columns=["evento", "tiempo"])
    )

    y = Surv.from_arrays(
        event=datos["evento"],
        time=datos["tiempo"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    modelo = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    scores_test = modelo.predict(X_test)

    # FIX 3: concordance_index_censored espera (event, time, estimate)
    # el orden original estaba correcto; se añade aserción defensiva
    eventos_t = np.array([e[0] for e in y_test])
    tiempos_t = np.array([e[1] for e in y_test])

    assert eventos_t.dtype == bool, "eventos_t debe ser booleano"
    assert (tiempos_t > 0).all(), "tiempos_t debe ser > 0"

    c_idx, *_ = concordance_index_censored(
        eventos_t,
        tiempos_t,
        scores_test
    )

    # FIX 4: calcular permutation importance aquí, con cache,
    # en lugar de recalcularlo en cada rerenderizado del Tab3
    perm = permutation_importance(
        modelo,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    return (
        modelo,
        X_train,
        X_test,
        y_train,
        y_test,
        c_idx,
        X.columns.tolist(),
        perm               # ← ahora viaja con el cache del modelo
    )


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Entrenando modelo..."):

    (
        modelo,
        X_train,
        X_test,
        y_train,
        y_test,
        c_idx,
        feat_cols,
        perm               # ← recibido desde el cache
    ) = entrenar_modelo()


datos = generar_datos()

n_total = len(datos)
n_train = len(X_train)
n_test = len(X_test)

n_churned = int(datos["evento"].sum())
churn_rate = n_churned / n_total

med_antig = int(datos["antiguedad"].median())


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("### 🔎 Evaluar cliente")

    edad = st.slider("Edad", 22, 65, 33)

    antiguedad = st.slider(
        "Antigüedad (meses)",
        1,
        48,
        10
    )

    plan = st.selectbox(
        "Plan",
        ["Basico", "Premium"]
    )

    incidencias = st.slider(
        "Incidencias",
        0,
        5,
        2
    )

    uso_horas = st.slider(
        "Uso semanal (horas)",
        5,
        100,
        30
    )

    pagos_at = st.slider(
        "Pagos atrasados",
        0,
        4,
        1
    )

    st.markdown("<br>", unsafe_allow_html=True)

    btn_predict = st.button("Calcular riesgo")

    st.markdown("---")

    st.markdown("### ⚙️ Datos de entrenamiento")

    st.caption(
        f"{n_total} registros sintéticos"
    )

    st.caption(
        f"Train / Test: {n_train} / {n_test}"
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
# FIX: st.markdown con divs flex no renderiza bien en Streamlit
# porque el wrapper interno fuerza display:block.
# Se reemplaza por st.title + st.caption nativos.
st.markdown(
    "<h1 style='font-size:1.75rem; font-weight:700; margin-bottom:0;'>"
    "📡 Análisis de Supervivencia de Clientes</h1>",
    unsafe_allow_html=True,
)
st.caption("Random Survival Forest · Predicción de churn")
st.divider()


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
# FIX: reemplazado HTML custom con flex por st.columns + st.metric nativos.
# Los divs con display:flex dentro de st.markdown no se respetan porque
# Streamlit inyecta un contenedor block intermedio.
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="C-index",
    value=f"{c_idx:.4f}",
    help="Capacidad discriminativa del modelo",
)
kpi2.metric(
    label="Tasa churn",
    value=f"{churn_rate*100:.1f}%",
    help=f"{n_churned} clientes",
)
kpi3.metric(
    label="Antigüedad mediana",
    value=f"{med_antig} m",
    help="Meses en plataforma",
)
kpi4.metric(
    label="Árboles RSF",
    value="100",
    help="Estimadores del bosque",
)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Exploración",
    "🎯 Predicción",
    "🌲 Importancia"
])


# ─────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────
with tab1:

    st.markdown(
        '<p class="section-title">Distribución de datos</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Survival times
    with col1:

        fig, ax = plt.subplots(figsize=(6, 3.5))

        colors = [
            DANGER if e else TEAL
            for e in datos["evento"]
        ]

        ax.bar(
            datos.index,
            datos["tiempo"],
            color=colors,
            width=1
        )

        ax.set_title(
            "Tiempo hasta churn/censura",
            fontsize=9
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        st.pyplot(fig)

        plt.close()

    # Churn by plan
    with col2:

        fig, ax = plt.subplots(figsize=(6, 3.5))

        plan_g = (
            datos.groupby("plan")["evento"]
            .mean() * 100
        )

        ax.bar(
            plan_g.index,
            plan_g.values,
            color=[AMBER, TEAL],
            width=0.45
        )

        ax.set_title(
            "Tasa churn por plan",
            fontsize=9
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        st.pyplot(fig)

        plt.close()

    st.markdown(
        '<p class="section-title">Dataset</p>',
        unsafe_allow_html=True
    )

    st.dataframe(
        datos.head(20),
        use_container_width=True,
        height=320
    )


# ─────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────
with tab2:

    if btn_predict:

        # FIX 5: construir nuevo con todas las columnas de feat_cols
        # y rellenar con 0 para evitar KeyError / columnas faltantes
        nuevo_raw = pd.DataFrame({
            "edad": [edad],
            "antiguedad": [antiguedad],
            "incidencias": [incidencias],
            "uso_horas": [uso_horas],
            "pagos_atrasados": [pagos_at],
            "plan_Basico":  [1 if plan == "Basico"  else 0],
            "plan_Premium": [1 if plan == "Premium" else 0],
        })

        # Alinear columnas exactamente con las del modelo (orden + completitud)
        nuevo = nuevo_raw.reindex(columns=feat_cols, fill_value=0)

        riesgo = modelo.predict(nuevo)[0]

        curva = modelo.predict_survival_function(
            nuevo
        )[0]

        train_scores = modelo.predict(X_train)

        r_norm = (
            (riesgo - train_scores.min())
            /
            (
                train_scores.max()
                - train_scores.min()
                + 1e-9
            )
        )

        r_norm = float(
            np.clip(r_norm, 0, 1)
        )

        if r_norm >= 0.65:

            badge_cls = "risk-high"
            badge_txt = "Alto riesgo"
            badge_icon = "🔴"

        elif r_norm >= 0.35:

            badge_cls = "risk-medium"
            badge_txt = "Riesgo medio"
            badge_icon = "🟡"

        else:

            badge_cls = "risk-low"
            badge_txt = "Bajo riesgo"
            badge_icon = "🟢"

        x_vals = np.asarray(curva.x)
        y_vals = np.asarray(curva.y)

        # FIX 6: usar clip para que x=12 nunca quede fuera del rango
        # (np.interp devuelve el valor del extremo si está fuera, pero
        # ser explícitos evita resultados silenciosamente incorrectos)
        t_clipped = np.clip(12, x_vals[0], x_vals[-1])
        prob_12m = float(np.interp(t_clipped, x_vals, y_vals))

        st.markdown(
            '<p class="section-title">Resultado</p>',
            unsafe_allow_html=True
        )

        col_res, col_curve = st.columns([1, 2])

        # Result card — FIX: display:flex dentro de st.markdown no funciona;
        # se reemplaza por st.metric nativos apilados verticalmente.
        with col_res:

            st.markdown(
                f"<div style='text-align:center; font-size:2.8rem; "
                f"margin-bottom:0.4rem;'>{badge_icon}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center; margin-bottom:1rem;'>"
                f"<span class='risk-badge {badge_cls}'>{badge_txt}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.metric("Score RSF",       f"{riesgo:.3f}")
            st.metric("Percentil riesgo", f"{r_norm*100:.1f}%")
            st.metric("Retención 12m",   f"{prob_12m*100:.1f}%")

        # Survival curve
        with col_curve:

            fig, ax = plt.subplots(figsize=(7, 3.8))

            ax.fill_between(
                curva.x,
                curva.y,
                alpha=0.12,
                color=TEAL
            )

            ax.plot(
                curva.x,
                curva.y,
                color=TEAL,
                linewidth=2.5
            )

            ax.axhline(
                0.5,
                linestyle="--",
                color=AMBER,
                alpha=0.6
            )

            ax.axvline(
                12,
                linestyle=":",
                color=MUTED,
                alpha=0.6
            )

            ax.scatter(
                [12],
                [prob_12m],
                color=AMBER,
                s=60
            )

            ax.set_ylim(0, 1.05)

            ax.set_xlabel(
                "Tiempo (meses)"
            )

            ax.set_ylabel(
                "Probabilidad retención"
            )

            ax.set_title(
                "Curva supervivencia",
                fontsize=9
            )

            ax.grid(
                linestyle="--",
                alpha=0.2
            )

            st.pyplot(fig)

            plt.close()

    else:

        st.info(
            "Configura un cliente y presiona "
            "'Calcular riesgo'."
        )


# ─────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────
with tab3:

    st.markdown(
        '<p class="section-title">Importancia de variables</p>',
        unsafe_allow_html=True
    )

    # FIX 4 (uso): perm ya viene del cache del modelo, no se recalcula aquí
    feat_df = pd.DataFrame({

        "variable": feat_cols,

        "importancia":
            perm.importances_mean

    }).sort_values(
        "importancia",
        ascending=True
    )

    col_bar, col_tbl = st.columns([2, 1])

    # Bar chart
    with col_bar:

        fig, ax = plt.subplots(
            figsize=(6.5, 3.8)
        )

        palette = [

            TEAL
            if v >= feat_df["importancia"].quantile(0.6)
            else MUTED

            for v in feat_df["importancia"]
        ]

        bars = ax.barh(
            feat_df["variable"],
            feat_df["importancia"],
            color=palette,
            height=0.55
        )

        for bar, val in zip(
            bars,
            feat_df["importancia"]
        ):

            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}",
                va="center",
                fontsize=7,
                color=TEXT
            )

        ax.set_xlabel(
            "Permutation importance",
            fontsize=8
        )

        ax.set_title(
            "Feature importance · RSF",
            fontsize=9
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.25
        )

        st.pyplot(fig)

        plt.close()

    # Table
    with col_tbl:

        display_df = (
            feat_df[
                ["variable", "importancia"]
            ]
            .sort_values(
                "importancia",
                ascending=False
            )
            .reset_index(drop=True)
        )

        display_df.index += 1

        display_df.columns = [
            "Variable",
            "Importancia"
        ]

        display_df["Importancia"] = (
            display_df["Importancia"]
            .map("{:.5f}".format)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=300
        )

    st.markdown("""
<div class="info-box">

<strong>Interpretación:</strong><br>

Las variables con mayor importancia
son las que más afectan el riesgo
de churn del cliente.

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="
    border-top:1px solid #1E3450;
    padding-top:0.9rem;
    text-align:center;
    color:#4A6580;
    font-size:0.75rem;
">

Análisis de Supervivencia de Clientes ·
Random Survival Forest ·
scikit-survival

</div>
""", unsafe_allow_html=True)
