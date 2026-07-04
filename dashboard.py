"""
dashboard.py
============
Dashboard Streamlit para visualizar resultados de evaluación RAGAS.

Uso:
    streamlit run dashboard.py

Estructura esperada de los JSON en results/final_results/:
{
    "data_config": {
        "experiment_name": "...",
        "chunking": { "strategy": "...", "params": {...} },
        "retrieval": { "method": "...", "params": { "reranking": bool, "top_k": int, ... } }
    },
    "results": {
        "Faithfulness": float,
        "Answer_Relevancy": float,
        "Context_Recall": float,
        "Context_Precision": float,
        "detail": [...]
    }
}
"""

import json
import difflib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Benchmark · RAGAS Dashboard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS personalizado
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    :root {
        --bg: #0d0f14;
        --surface: #151820;
        --surface2: #1c2030;
        --border: #2a2f3d;
        --accent: #00e5a0;
        --accent2: #7b61ff;
        --accent3: #ff6b6b;
        --accent4: #ffd93d;
        --text: #e8eaf0;
        --muted: #6b7280;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Syne', sans-serif;
    }

    .stApp > header { background: transparent !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Títulos */
    h1 { font-family: 'Syne', sans-serif; font-weight: 800; color: var(--accent) !important; letter-spacing: -1px; }
    h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; color: var(--text) !important; }

    /* Métricas */
    [data-testid="stMetric"] {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.8rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--muted) !important;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: var(--bg) !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Cards */
    .metric-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .metric-card:hover { border-color: var(--accent); transition: border-color 0.2s; }

    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .tag-chunking { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
    .tag-retrieval { background: rgba(123,97,255,0.15); color: #7b61ff; border: 1px solid rgba(123,97,255,0.3); }
    .tag-rerank { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }
    .tag-no-rerank { background: rgba(107,114,128,0.15); color: #6b7280; border: 1px solid rgba(107,114,128,0.3); }

    .divider { height: 1px; background: var(--border); margin: 24px 0; }

    /* Plotly charts dark background */
    .js-plotly-plot { border-radius: 12px; }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] { background: var(--surface2) !important; }

    /* Warning / info */
    .stAlert { background: var(--surface2) !important; border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "final_results"
GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "golden_dataset"
    / "golden_dataset_single-turn.jsonl"
)

METRICS = ["Faithfulness", "Answer_Relevancy", "Context_Recall", "Context_Precision"]
METRIC_COLORS = {
    "Faithfulness":     "#00e5a0",
    "Answer_Relevancy": "#7b61ff",
    "Context_Recall":   "#ff6b6b",
    "Context_Precision":"#ffd93d",
}
METRIC_LABELS = {
    "Faithfulness":     "Faithfulness",
    "Answer_Relevancy": "Answer Relevancy",
    "Context_Recall":   "Context Recall",
    "Context_Precision":"Context Precision",
}

DETAIL_METRIC_KEYS = {
    "Faithfulness": "faithfulness",
    "Answer_Relevancy": "answer_relevancy",
    "Context_Recall": "context_recall",
    "Context_Precision": "context_precision",
}

NO_INFO_MARKER = "No dispongo"
EXPECTED_NO_ANSWER_MARKER = "no se puede responder"
OUT_OF_DOMAIN_CATEGORY = "out_of_domain"


def text_key(value: object) -> str:
    """Normaliza texto para cruzar resultados limpios y resultados con mojibake."""
    text = " ".join(str(value or "").split())
    if "\u00c2" in text or "\u00c3" in text:
        try:
            text = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text.casefold()


@st.cache_data
def load_question_categories(golden_path: Path) -> dict:
    """Carga categorias del golden dataset indexadas por pregunta."""
    categories = {"questions": {}, "references": {}}
    if not golden_path.exists():
        return categories

    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            category = (row.get("metadata") or {}).get("category")
            if category:
                categories["questions"][text_key(row.get("user_input"))] = category
                categories["references"][text_key(row.get("reference"))] = category
    return categories


def detail_category(detail_row: dict, question_categories: dict) -> str | None:
    """Obtiene la categoria desde el detalle o desde el golden dataset."""
    metadata = detail_row.get("metadata") or {}
    category = metadata.get("category") or detail_row.get("category")
    if category:
        return category

    question_index = question_categories.get("questions", {})
    reference_index = question_categories.get("references", {})

    question_key = text_key(detail_row.get("user_input"))
    category = question_index.get(question_key)
    if category:
        return category

    reference_key = text_key(detail_row.get("reference"))
    category = reference_index.get(reference_key)
    if category:
        return category

    matches = difflib.get_close_matches(question_key, question_index.keys(), n=1, cutoff=0.82)
    if matches:
        return question_index.get(matches[0])

    matches = difflib.get_close_matches(reference_key, reference_index.keys(), n=1, cutoff=0.82)
    return reference_index.get(matches[0]) if matches else None


def is_out_of_domain_question(detail_row: dict, question_categories: dict) -> bool:
    """Detecta preguntas fuera de dominio."""
    return detail_category(detail_row, question_categories) == OUT_OF_DOMAIN_CATEGORY


def has_no_info_response(detail_row: dict) -> bool:
    """Detecta respuestas de abstencion del modelo."""
    return NO_INFO_MARKER.lower() in str(detail_row.get("response", "")).lower()


def has_expected_no_answer_reference(detail_row: dict) -> bool:
    """Detecta preguntas cuya referencia indica que no debian responderse."""
    reference = str(detail_row.get("reference", "")).lower()
    return EXPECTED_NO_ANSWER_MARKER in reference


def has_expected_no_answer_abstention(detail_row: dict) -> bool:
    """Detecta abstenciones correctas: el modelo se abstiene y la referencia tambien."""
    return has_no_info_response(detail_row) and has_expected_no_answer_reference(detail_row)


def has_all_zero_metrics(detail_row: dict) -> bool:
    """Detecta preguntas donde todas las metricas RAGAS valen 0.0."""
    values = []
    for detail_key in DETAIL_METRIC_KEYS.values():
        val = detail_row.get(detail_key)
        if val is None:
            return False
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            return False
    return bool(values) and all(val == 0.0 for val in values)


def filtered_detail_rows(
    detail_rows: list,
    question_categories: dict,
    exclude_out_of_domain: bool,
    exclude_expected_no_answer: bool,
    exclude_no_info: bool,
    exclude_all_zero: bool,
) -> list:
    """Aplica filtros de preguntas antes de recalcular medias."""
    filtered = []
    for detail_row in detail_rows:
        if exclude_out_of_domain and is_out_of_domain_question(detail_row, question_categories):
            continue
        if exclude_expected_no_answer and has_expected_no_answer_abstention(detail_row):
            continue
        if exclude_no_info and has_no_info_response(detail_row):
            continue
        if exclude_all_zero and has_all_zero_metrics(detail_row):
            continue
        filtered.append(detail_row)
    return filtered


def mean_metric_from_detail(detail_rows: list, metric: str):
    values = []
    detail_key = DETAIL_METRIC_KEYS[metric]
    for detail_row in detail_rows:
        val = detail_row.get(detail_key)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 4) if values else None


@st.cache_data
def load_results(
    results_dir: Path,
    question_categories: dict,
    exclude_out_of_domain: bool = False,
    exclude_expected_no_answer: bool = False,
    exclude_no_info: bool = False,
    exclude_all_zero: bool = False,
) -> pd.DataFrame:
    """Carga todos los JSONs de resultados y los convierte en un DataFrame."""
    rows = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = data.get("data_config", {})
            res = data.get("results", {})

            chunking_cfg = cfg.get("chunking", {})
            retrieval_cfg = cfg.get("retrieval", {})
            retrieval_params = retrieval_cfg.get("params", {})
            chunking_params = chunking_cfg.get("params", {})

            reranking = retrieval_params.get("reranking", False)
            top_k_raw = retrieval_params.get("top_k")
            top_n_raw = retrieval_params.get("top_n")
            top_k = "" if top_k_raw is None else str(top_k_raw)
            top_n = "" if top_n_raw is None else str(top_n_raw)
            # chunk_size: para hierarchical usa child_chunks, para el resto chunk_size
            chunk_size_raw = chunking_params.get("chunk_size") or chunking_params.get("child_chunks")
            chunk_size = int(chunk_size_raw) if chunk_size_raw is not None else None

            chunk_overlap_raw = chunking_params.get("chunk_overlap") or chunking_params.get("child_overlap")
            chunk_overlap = round(int(chunk_overlap_raw), 0) if chunk_overlap_raw is not None else None

            parent_size_raw = chunking_params.get("parent_chunks", None)
            parent_size = int(parent_size_raw) if parent_size_raw is not None else None

            label = (
                f"{chunking_cfg.get('strategy','?')} | "
                f"{retrieval_cfg.get('method','?')} | "
                f"{'rerank' if reranking else 'no-rerank'}"
            )

            row = {
                "file": json_file.name,
                "experiment_name": json_file.stem,
                "config_experiment_name": cfg.get("experiment_name", json_file.stem),
                "chunking_strategy": chunking_cfg.get("strategy", "unknown"),
                "retrieval_method": retrieval_cfg.get("method", "unknown"),
                "reranking": reranking,
                "top_k": top_k,
                "top_n": top_n,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "parent_size": parent_size,
                "label": label,
            }

            detail_rows = res.get("detail", [])
            if not isinstance(detail_rows, list):
                detail_rows = []
            selected_detail_rows = filtered_detail_rows(
                detail_rows,
                question_categories=question_categories,
                exclude_out_of_domain=exclude_out_of_domain,
                exclude_expected_no_answer=exclude_expected_no_answer,
                exclude_no_info=exclude_no_info,
                exclude_all_zero=exclude_all_zero,
            )

            row["questions_total"] = len(detail_rows)
            row["questions_out_of_domain"] = sum(
                1 for detail_row in detail_rows
                if is_out_of_domain_question(detail_row, question_categories)
            )
            row["questions_factual_trap"] = sum(
                1 for detail_row in detail_rows
                if detail_category(detail_row, question_categories) == "factual_trap"
            )
            row["questions_used"] = len(selected_detail_rows) if detail_rows else None
            row["questions_excluded"] = (
                len(detail_rows) - len(selected_detail_rows) if detail_rows else None
            )

            for metric in METRICS:
                if exclude_out_of_domain or exclude_expected_no_answer or exclude_no_info or exclude_all_zero:
                    val = mean_metric_from_detail(selected_detail_rows, metric)
                else:
                    val = res.get(metric)
                row[metric] = round(float(val), 4) if val is not None and not (isinstance(val, float) and val != val) else None

            rows.append(row)
        except Exception as e:
            st.warning(f"No se pudo cargar {json_file.name}: {e}")

    return pd.DataFrame(rows)


def make_bar_chart(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()

    for _, row in df.iterrows():
        val = row[metric]
        if val is None:
            continue

        hover = (
            f"<b>{row['experiment_name']}</b><br>"
            f"Chunking: {row['chunking_strategy']}<br>"
            f"Chunk size: {row['chunk_size']} | Overlap: {row['chunk_overlap']}<br>"
            f"Retrieval: {row['retrieval_method']}<br>"
            f"Top-K: {row['top_k']} | Top-N: {row['top_n']}<br>"
            f"Reranking: {'✓' if row['reranking'] else '✗'}<br>"
            f"<b>{METRIC_LABELS[metric]}: {val:.4f}</b>"
        )

        color = METRIC_COLORS[metric]
        opacity = 1.0 if row["reranking"] else 0.55

        fig.add_trace(go.Bar(
            x=[row["label"]],
            y=[val],
            name=row["experiment_name"],
            marker=dict(color=color, opacity=opacity, line=dict(color=color, width=1.5)),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151820",
        font=dict(family="JetBrains Mono, monospace", color="#e8eaf0", size=11),
        title=dict(text=title, font=dict(size=13, color=METRIC_COLORS[metric])),
        barmode="group",
        xaxis=dict(gridcolor="#2a2f3d", zerolinecolor="#2a2f3d", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#2a2f3d", zerolinecolor="#2a2f3d", range=[0, 1], title="Score"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f3d", borderwidth=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte hex a rgba string válido para Plotly."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_radar_chart(df: pd.DataFrame, selected_experiments: list) -> go.Figure:
    fig = go.Figure()
    palette = ["#00e5a0", "#7b61ff", "#ff6b6b", "#ffd93d", "#38bdf8", "#fb923c"]

    for i, exp_name in enumerate(selected_experiments):
        row = df[df["experiment_name"] == exp_name]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [row[m] if row[m] is not None else 0 for m in METRICS]
        vals_closed = vals + [vals[0]]
        labels_closed = [METRIC_LABELS[m] for m in METRICS] + [METRIC_LABELS[METRICS[0]]]

        color = palette[i % len(palette)]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2),
            name=row["label"],
            hovertemplate="<b>%{theta}</b>: %{r:.4f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#e8eaf0", size=11),
        polar=dict(
            bgcolor="#151820",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2f3d", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#2a2f3d"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f3d", borderwidth=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=420,
        showlegend=True,
    )
    return fig


def make_scatter_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    palette = {"recursive": "#00e5a0", "semantic": "#7b61ff", "parent_child": "#ff6b6b"}

    for strategy in df["chunking_strategy"].unique():
        sub = df[df["chunking_strategy"] == strategy].copy()
        sub = sub[sub[metric].notna()]
        if sub.empty:
            continue

        color = palette.get(strategy, "#ffd93d")

        for _, row in sub.iterrows():
            symbol = "circle" if row["reranking"] else "circle-open"
            hover = (
                f"<b>{row['experiment_name']}</b><br>"
                f"Chunk size: {row['chunk_size']}<br>"
                f"Retrieval: {row['retrieval_method']}<br>"
                f"Reranking: {'✓' if row['reranking'] else '✗'}<br>"
                f"<b>{METRIC_LABELS[metric]}: {row[metric]:.4f}</b>"
            )
            fig.add_trace(go.Scatter(
                x=[row["chunk_size"]],
                y=[row[metric]],
                mode="markers",
                marker=dict(size=14, color=color, symbol=symbol, line=dict(color=color, width=2)),
                name=strategy,
                legendgroup=strategy,
                showlegend=True,
                hovertemplate=hover + "<extra></extra>",
            ))

    seen = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        seen.add(trace.name)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151820",
        font=dict(family="JetBrains Mono, monospace", color="#e8eaf0", size=11),
        title=dict(text=f"{METRIC_LABELS[metric]} vs Chunk Size", font=dict(size=13, color=METRIC_COLORS[metric])),
        xaxis=dict(gridcolor="#2a2f3d", zerolinecolor="#2a2f3d", title="Chunk Size", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#2a2f3d", zerolinecolor="#2a2f3d", range=[0, 1], title="Score"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f3d", borderwidth=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("## ⚗️ RAG Benchmark")
st.markdown(
    "<p style='color:#6b7280; font-family: JetBrains Mono, monospace; font-size:0.85rem; margin-top:-12px;'>"
    "Evaluación de estrategias RAG · RAGAS Dashboard</p>",
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    results_path = st.text_input(
        "Carpeta de resultados",
        value=str(RESULTS_DIR),
        help="Ruta a la carpeta con los JSON evaluados"
    )
    RESULTS_DIR = Path(results_path)

    st.markdown("#### Filtros de preguntas")
    exclude_out_of_domain = st.checkbox(
        "Quitar out_of_domain",
        value=False,
        help="Excluye solo las preguntas con categoria out_of_domain del golden dataset."
    )
    exclude_expected_no_answer = st.checkbox(
        "Quitar abstenciones esperadas",
        value=False,
        help="Excluye solo cuando response contiene 'No dispongo' y reference indica 'No se puede responder'."
    )
    exclude_no_info = st.checkbox(
        "Quitar cualquier 'No dispongo'",
        value=False,
        help="Excluye todas las filas cuyo campo response contiene 'No dispongo', incluso si la pregunta si debia responderse."
    )
    exclude_all_zero = st.checkbox(
        "Quitar preguntas con todo 0.0",
        value=False,
        help="Excluye las filas donde faithfulness, answer_relevancy, context_recall y context_precision son 0.0."
    )

    if st.button("🔄 Recargar datos", use_container_width=True):
        st.cache_data.clear()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7280; font-size:0.75rem; font-family: JetBrains Mono;'>"
        "Los JSON se cargan automáticamente.<br>Añade nuevos archivos y recarga.</p>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────

if not RESULTS_DIR.exists():
    st.error(f"La carpeta `{RESULTS_DIR}` no existe. Revisa la ruta en el sidebar.")
    st.stop()

question_categories = load_question_categories(GOLDEN_DATASET_PATH)
if not question_categories.get("questions"):
    st.warning(
        f"No se pudo cargar el golden dataset en `{GOLDEN_DATASET_PATH}`. "
        "El filtro out_of_domain no podra usar categorias."
    )

df = load_results(
    RESULTS_DIR,
    question_categories=question_categories,
    exclude_out_of_domain=exclude_out_of_domain,
    exclude_expected_no_answer=exclude_expected_no_answer,
    exclude_no_info=exclude_no_info,
    exclude_all_zero=exclude_all_zero,
)

if df.empty:
    st.warning("No se encontraron archivos JSON en la carpeta especificada.")
    st.stop()

# ──────────────────────────────────────────────
# KPIs globales
# ──────────────────────────────────────────────

if exclude_out_of_domain or exclude_expected_no_answer or exclude_no_info or exclude_all_zero:
    active_filters = []
    if exclude_out_of_domain:
        active_filters.append("preguntas out_of_domain")
    if exclude_expected_no_answer:
        active_filters.append("abstenciones esperadas")
    if exclude_no_info:
        active_filters.append("cualquier respuesta 'No dispongo'")
    if exclude_all_zero:
        active_filters.append("preguntas con todas las metricas a 0.0")
    st.info(
        "Medias recalculadas excluyendo: "
        + " y ".join(active_filters)
        + ". Los JSON originales no se modifican."
    )

if exclude_out_of_domain and "questions_out_of_domain" in df.columns:
    out_of_domain_counts = df["questions_out_of_domain"].dropna().unique().tolist()
    if len(out_of_domain_counts) == 1:
        st.caption(f"Preguntas out_of_domain detectadas por experimento: {int(out_of_domain_counts[0])}.")
    else:
        st.warning(
            "El numero de preguntas out_of_domain detectadas no es constante entre experimentos: "
            + ", ".join(str(int(count)) for count in sorted(out_of_domain_counts))
            + "."
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("### 📊 Mejor resultado por métrica")

cols = st.columns(4)
for i, metric in enumerate(METRICS):
    valid = df[metric].dropna()
    if not valid.empty:
        best_val = valid.max()
        best_exp = df.loc[df[metric] == best_val, "experiment_name"].values[0]
        cols[i].metric(
            label=METRIC_LABELS[metric],
            value=f"{best_val:.3f}",
            delta=f"↑ {best_exp}",
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab1, tab2 = st.tabs(["📈 Comparativa Global", "🔬 Análisis por Chunking"])

# ══════════════════════════════════════════════
# TAB 1 — Comparativa Global
# ══════════════════════════════════════════════

with tab1:

    # Radar de comparativa
    st.markdown("#### Radar · Comparativa de experimentos")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.8rem; font-family: JetBrains Mono;'>"
        "Selecciona los experimentos a comparar</p>",
        unsafe_allow_html=True
    )

    all_experiments = df["experiment_name"].tolist()
    selected = st.multiselect(
        "Experimentos",
        options=all_experiments,
        default=all_experiments[:min(4, len(all_experiments))],
        label_visibility="collapsed"
    )

    if selected:
        st.plotly_chart(make_radar_chart(df, selected), width='stretch')
    else:
        st.info("Selecciona al menos un experimento.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Barras por métrica
    st.markdown("#### Barras · Todas las métricas")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.8rem; font-family: JetBrains Mono;'>"
        "Barras sólidas = con reranking · Barras semitransparentes = sin reranking</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_bar_chart(df, "Faithfulness", "Faithfulness"), width='stretch')
        st.plotly_chart(make_bar_chart(df, "Context_Recall", "Context Recall"), width='stretch')
    with col2:
        st.plotly_chart(make_bar_chart(df, "Answer_Relevancy", "Answer Relevancy"), width='stretch')
        st.plotly_chart(make_bar_chart(df, "Context_Precision", "Context Precision"), width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Scatter chunk_size vs métrica
    st.markdown("#### Scatter · Chunk Size vs Métricas")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.8rem; font-family: JetBrains Mono;'>"
        "Puntos rellenos = reranking · Puntos vacíos = sin reranking</p>",
        unsafe_allow_html=True
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(make_scatter_chart(df, "Faithfulness"), width='stretch')
        st.plotly_chart(make_scatter_chart(df, "Context_Recall"), width='stretch')
    with sc2:
        st.plotly_chart(make_scatter_chart(df, "Answer_Relevancy"), width='stretch')
        st.plotly_chart(make_scatter_chart(df, "Context_Precision"), width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Tabla resumen
    st.markdown("#### Tabla resumen")
    display_cols = [
        "experiment_name", "chunking_strategy", "chunk_size", "retrieval_method",
        "reranking", "top_k", "questions_used", "questions_excluded",
        "questions_out_of_domain", "questions_factual_trap"
    ] + METRICS
    display_df = df[display_cols].copy()
    display_df.columns = [
        "Experimento", "Chunking", "Chunk Size", "Retrieval", "Reranking",
        "Top-K", "Preguntas usadas", "Preguntas excluidas",
        "Out of domain", "Factual trap"
    ] + [METRIC_LABELS[m] for m in METRICS]
    st.dataframe(
        display_df.style.format({METRIC_LABELS[m]: "{:.4f}" for m in METRICS}),
        width='stretch',
        hide_index=True,
    )


# ══════════════════════════════════════════════
# TAB 2 — Análisis por Chunking
# ══════════════════════════════════════════════

with tab2:

    strategies = df["chunking_strategy"].unique().tolist()
    selected_strategy = st.selectbox(
        "Estrategia de chunking",
        options=strategies,
        format_func=lambda x: x.upper()
    )

    filtered = df[df["chunking_strategy"] == selected_strategy]

    if filtered.empty:
        st.warning("No hay experimentos con esta estrategia.")
    else:
        # Tags de los experimentos filtrados
        st.markdown("#### Experimentos con esta estrategia")
        for _, row in filtered.iterrows():
            rerank_tag = (
                "<span class='tag tag-rerank'>reranking ✓</span>"
                if row["reranking"]
                else "<span class='tag tag-no-rerank'>sin reranking</span>"
            )
            st.markdown(
                f"<div class='metric-card'>"
                f"<b style='font-family: JetBrains Mono; color:#e8eaf0;'>{row['experiment_name']}</b><br><br>"
                f"<span class='tag tag-chunking'>{row['chunking_strategy']}</span>"
                f"<span class='tag tag-retrieval'>{row['retrieval_method']}</span>"
                f"{rerank_tag}"
                f"<span class='tag' style='background:rgba(255,217,61,0.15);color:#ffd93d;border:1px solid rgba(255,217,61,0.3);'>"
                f"chunk_size: {row['chunk_size']}</span>"
                f"<span class='tag' style='background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);'>"
                f"top_k: {row['top_k']}</span>"
                f"<div style='margin-top:12px; display:flex; gap:24px;'>"
                + "".join([
                    f"<span style='font-family:JetBrains Mono; font-size:0.8rem;'>"
                    f"<span style='color:{METRIC_COLORS[m]};'>{METRIC_LABELS[m]}:</span> "
                    f"<b>{row[m]:.4f}</b></span>"
                    for m in METRICS if row[m] is not None
                ])
                + "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Gráficas filtradas por estrategia
        st.markdown(f"#### Métricas · Estrategia `{selected_strategy.upper()}`")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(filtered, "Faithfulness", "Faithfulness"), width='stretch')
            st.plotly_chart(make_bar_chart(filtered, "Context_Recall", "Context Recall"), width='stretch')
        with col2:
            st.plotly_chart(make_bar_chart(filtered, "Answer_Relevancy", "Answer Relevancy"), width='stretch')
            st.plotly_chart(make_bar_chart(filtered, "Context_Precision", "Context Precision"), width='stretch')

        # Radar solo para esta estrategia
        if len(filtered) > 1:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown(f"#### Radar · Comparativa dentro de `{selected_strategy.upper()}`")
            st.plotly_chart(
                make_radar_chart(filtered, filtered["experiment_name"].tolist()),
                width='stretch'
            )

        # Tabla de esta estrategia
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Detalle de parámetros")
        detail_cols = [
            "experiment_name", "chunk_size", "chunk_overlap", "retrieval_method",
            "reranking", "top_k", "top_n", "questions_used", "questions_excluded",
            "questions_out_of_domain", "questions_factual_trap"
        ] + METRICS
        detail_df = filtered[[c for c in detail_cols if c in filtered.columns]].copy()
        st.dataframe(
            detail_df.style.format({m: "{:.4f}" for m in METRICS}),
            width='stretch',
            hide_index=True,
        )
