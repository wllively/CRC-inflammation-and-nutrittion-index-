import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pickle


st.set_page_config(
    page_title="GBSA Survival Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner=False)
def load_setting():
    settings = {
        'Age': {
            'values': [18, 90], 'type': 'slider', 'init_value': 60,
            'add_after': ' year', 'model_key': 'Age', 'group': 'baseline'
        },
        'NAD': {
            'values': ['No', 'Yes'], 'type': 'selectbox', 'init_value': 0,
            'add_after': '', 'model_key': 'NAD', 'group': 'baseline'
        },
        'Overall complications': {
            'values': ['No', 'Yes'], 'type': 'selectbox', 'init_value': 0,
            'add_after': '', 'model_key': 'Overall_complications', 'group': 'baseline'
        },
        'Stage': {
            'values': ['I', 'II', 'III', 'IV'], 'type': 'selectbox', 'init_value': 0,
            'add_after': '', 'model_key': 'Stage', 'group': 'baseline'
        },
        'CA19-9': {
            'type': 'number_input', 'init_value': 0.0, 'min_value': 0.0,
            'add_after': ' U/mL', 'model_key': 'CA199', 'group': 'tumor'
        },
        'CEA': {
            'type': 'number_input', 'init_value': 0.0, 'min_value': 0.0,
            'add_after': ' ng/mL', 'model_key': 'CEA', 'group': 'tumor'
        },
        'Albumin': {
            'type': 'number_input', 'init_value': 40.0, 'min_value': 0.01,
            'add_after': ' g/L', 'model_key': 'Alb', 'group': 'lab'
        },
        'Fibrinogen': {
            'type': 'number_input', 'init_value': 3.0, 'min_value': 0.0,
            'add_after': ' g/L', 'model_key': 'FIB', 'group': 'lab'
        },
        'Neutrophil count': {
            'type': 'number_input', 'init_value': 3.5, 'min_value': 0.0,
            'add_after': ' ×10^9/L', 'model_key': 'NEUT', 'group': 'lab'
        },
        'Lymphocyte count': {
            'type': 'number_input', 'init_value': 1.5, 'min_value': 0.01,
            'add_after': ' ×10^9/L', 'model_key': 'LYM', 'group': 'lab'
        },
        'Monocyte count': {
            'type': 'number_input', 'init_value': 0.4, 'min_value': 0.01,
            'add_after': ' ×10^9/L', 'model_key': 'MONO', 'group': 'lab'
        },
        'Platelet count': {
            'type': 'number_input', 'init_value': 200.0, 'min_value': 0.01,
            'add_after': ' ×10^9/L', 'model_key': 'PLT', 'group': 'lab'
        },
        'Total bilirubin': {
            'type': 'number_input', 'init_value': 12.0, 'min_value': 0.01,
            'add_after': ' μmol/L', 'model_key': 'TBIL', 'group': 'lab'
        },
        'Alkaline phosphatase': {
            'type': 'number_input', 'init_value': 80.0, 'min_value': 0.01,
            'add_after': ' U/L', 'model_key': 'ALP', 'group': 'lab'
        }
    }

    input_keys = [
        'Age', 'NAD', 'Overall_complications', 'Stage', 'CA199', 'CEA',
        'FAR', 'PNI', 'LMR', 'ALBI', 'NPR', 'SII', 'AAPR', 'PAR'
    ]
    return settings, input_keys

@st.cache_data(show_spinner=False)
def get_model():
    with open('./gbsa_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def get_scaler():
    with open("./scaler_transform.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# =========================
# 2. UI输入组件
# =========================



def render_widget(label, cfg):
    if cfg['type'] == 'slider':
        st.slider(
            label + cfg['add_after'],
            min_value=cfg['values'][0],
            max_value=cfg['values'][1],
            value=cfg['init_value'],
            key=label
        )

    elif cfg['type'] == 'selectbox':
        st.selectbox(
            label + cfg['add_after'],
            options=cfg['values'],
            index=cfg['init_value'],
            key=label
        )

    elif cfg['type'] == 'number_input':
        st.number_input(
            label + cfg['add_after'],
            min_value=float(cfg.get('min_value', 0.0)),
            value=float(cfg['init_value']),
            step=0.01,
            key=label
        )
        

# =========================
# 3. 输入转模型特征
# =========================
final_features = [
    'Age', 'NAD', 'Overall_complications', 'Stage',
    'CA199', 'CEA', 'FAR', 'PNI', 'LMR', 'ALBI', 'NPR', 'SII', 'AAPR', 'PAR'
]

def get_input_dataframe(settings, input_keys, scaler):
    raw_data = {}

    for ui_key, config in settings.items():
        value = st.session_state.get(ui_key)
        model_key = config['model_key']
        raw_data[model_key] = value

    input_df = pd.DataFrame([raw_data])

    stage_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    yes_no_mapping = {"No": 0, "Yes": 1}

    input_df['Stage'] = input_df['Stage'].map(stage_mapping)
    input_df['NAD'] = input_df['NAD'].map(yes_no_mapping)
    input_df['Overall_complications'] = input_df['Overall_complications'].map(yes_no_mapping)

    numeric_cols = ['Age', 'CA199', 'CEA', 'Alb', 'FIB', 'NEUT', 'LYM', 'MONO', 'PLT', 'TBIL', 'ALP']
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    if input_df[numeric_cols].isnull().any().any():
        st.error("Some numeric inputs are invalid. Please check all laboratory values.")
        return None, None

    eps = 1e-8
    input_df['FAR'] = input_df['FIB'] / (input_df['Alb'] + eps)
    input_df['PNI'] = input_df['Alb'] + 5 * input_df['LYM']
    input_df['LMR'] = input_df['LYM'] / (input_df['MONO'] + eps)
    input_df['ALBI'] = 0.66 * np.log10(input_df['TBIL'] + eps) - 0.085 * input_df['Alb']
    input_df['NPR'] = input_df['NEUT'] / (input_df['PLT'] + eps)
    input_df['SII'] = input_df['PLT'] * input_df['NEUT'] / (input_df['LYM'] + eps)
    input_df['AAPR'] = input_df['Alb'] / (input_df['ALP'] + eps)
    input_df['PAR'] = input_df['PLT'] / (input_df['Alb'] + eps)

    model_input_df = input_df[final_features].copy()

    catogory_vars = ['Stage', 'Overall_complications', 'NAD']
    continus_var = np.setdiff1d(final_features, catogory_vars).tolist()   # 关键改这里

    model_input_df[continus_var] = scaler.transform(model_input_df[continus_var])

    return model_input_df, input_df


# =========================
# 4. 结果展示函数
# =========================


def show_risk_stratification(risk_score, cutoff):
    st.subheader("Risk stratification")

    c1, c2 = st.columns(2)
    c1.metric("Risk score", f"{risk_score:.3f}")
    c2.metric("Cut-off", f"{cutoff:.3f}")

    if risk_score < cutoff:
        st.success("Current patient: Low-risk group")
    else:
        st.error("Current patient: High-risk group")



def show_survival_metrics(chf):
    times = [12, 36, 60]
    labels = ["1-year survival", "3-year survival", "5-year survival"]

    cols = st.columns(3)
    for col, t, lab in zip(cols, times, labels):
        surv = float(np.exp(-chf(t)))
        col.metric(lab, f"{surv:.1%}")


def show_survival_table(chf):
    rows = []
    for t in [12, 36, 60]:
        rows.append({
            "Time": f"{t // 12}-year",
            "Cumulative hazard": round(float(chf(t)), 4),
            "Survival probability": f"{np.exp(-float(chf(t))):.4f}"
        })
    st.table(pd.DataFrame(rows))





def show_calculated_indices_pretty(full_df):
    st.subheader("Calculated inflammatory and nutritional indices")

    items = [
        ("FAR", full_df.loc[0, 'FAR']),
        ("PNI", full_df.loc[0, 'PNI']),
        ("LMR", full_df.loc[0, 'LMR']),
        ("ALBI", full_df.loc[0, 'ALBI']),
        ("NPR", full_df.loc[0, 'NPR']),
        ("SII", full_df.loc[0, 'SII']),
        ("AAPR", full_df.loc[0, 'AAPR']),
        ("PAR", full_df.loc[0, 'PAR']),
    ]

    for i in range(0, len(items), 2):
        c1, c2 = st.columns(2)

        k1, v1 = items[i]
        c1.metric(k1, f"{v1:.3f}")

        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            c2.metric(k2, f"{v2:.3f}")


def plot_cumulative_hazard_curve(chf, risk_score):
    x_plot = chf.x
    y_plot = chf(x_plot)   # 关键：不要直接用 chf.y

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_plot, y_plot, linewidth=2.5)

    key_times = [12, 36, 60]
    key_points_x = []
    key_points_y = []

    for t in key_times:
        if t <= max(x_plot):
            key_points_x.append(t)
            key_points_y.append(float(chf(t)))

    if key_points_x:
        ax.scatter(key_points_x, key_points_y, s=35, zorder=3)

        for x, y, lab in zip(key_points_x, key_points_y, ["1-year", "3-year", "5-year"]):
            ax.annotate(
                lab,
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=10
            )

    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Cumulative hazard", fontsize=11)
    ax.set_title("Individual cumulative hazard curve", fontsize=14, pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.25)

    ax.text(
        0.98, 0.05,
        f"Risk score = {risk_score:.3f}",
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.08)
    )

    plt.tight_layout()
    return fig

##结果预测


q = 0.57


def predict(settings, input_keys, model, scaler):
    model_input_df, full_df = get_input_dataframe(settings, input_keys, scaler)

    if model_input_df is None:
        st.error("Data processing error. Prediction cannot be performed.")
        return

    if not hasattr(model, "predict_cumulative_hazard_function"):
        st.error("The loaded model does not support cumulative hazard prediction.")
        return

    predicted_risk = model.predict(model_input_df)
    cumulative_hazard_functions = model.predict_cumulative_hazard_function(model_input_df)

    risk_score = float(predicted_risk[0])
    chf = cumulative_hazard_functions[0]

    # 展示结果
    left_col, right_col = st.columns([1.0, 1.25], gap="large")

    with left_col:
        show_risk_stratification(risk_score, q)

        st.markdown("### Survival probability")
        show_survival_metrics(chf)

        with st.expander("Detailed survival table", expanded=False):
            show_survival_table(chf)

        with st.expander("Model input features", expanded=False):
            st.dataframe(model_input_df.round(4), hide_index=True, use_container_width=True)

        with st.expander("Calculated indices", expanded=True):
            show_calculated_indices_pretty(full_df)

    with right_col:
        st.subheader("Cumulative hazard curve")
        fig = plot_cumulative_hazard_curve(chf, risk_score)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =========================
# 6. 主页面
# =========================

settings, input_keys = load_setting()
model = get_model()
scaler = get_scaler()

st.header("GBSA-based postoperative survival prediction model for colorectal cancer")
st.markdown(
    """
    <p style="font-size:30px;">
    Enter postoperative clinicopathological and laboratory parameters in the sidebar 
    to obtain individualized survival estimates.
    </p>
    """,
    unsafe_allow_html=True
)

result_box = st.container()

with st.sidebar:
    st.title("Patient parameter entry")

    with st.form("my_form", clear_on_submit=False):
        st.markdown("### Baseline characteristics")
        for label, cfg in settings.items():
            if cfg['group'] == 'baseline':
                render_widget(label, cfg)

        st.markdown("### Tumor markers")
        for label, cfg in settings.items():
            if cfg['group'] == 'tumor':
                render_widget(label, cfg)

        st.markdown("### Laboratory parameters")
        for label, cfg in settings.items():
            if cfg['group'] == 'lab':
                render_widget(label, cfg)

        submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    with result_box:
        predict(settings, input_keys, model, scaler)