import streamlit as st
import numpy as np
import sksurv
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pickle
import shap
import warnings

warnings.filterwarnings("ignore")


# =========================
# 0. 页面设置
# =========================
st.set_page_config(
    page_title="GBSA Survival Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 1. 全局参数
# =========================
RISK_CUTOFF = 0.77

MODEL_FEATURES = [
    'Age', 'NAD', 'Stage', 'CEA', 'CA199',
    'FAR', 'PNI', 'NLR', 'ALBI', 'LMR'
]


# =========================
# 2. 加载设置、模型、scaler、SHAP background
# =========================
@st.cache_data(show_spinner=False)
def load_setting():
    settings = {
        'Age': {
            'values': [18, 90],
            'type': 'slider',
            'init_value': 60,
            'add_after': ' year',
            'model_key': 'Age',
            'group': 'baseline'
        },
        'NAD': {
            'values': ['No', 'Yes'],
            'type': 'selectbox',
            'init_value': 0,
            'add_after': '',
            'model_key': 'NAD',
            'group': 'baseline'
        },
        'Stage': {
            'values': ['I', 'II', 'III', 'IV'],
            'type': 'selectbox',
            'init_value': 0,
            'add_after': '',
            'model_key': 'Stage',
            'group': 'baseline'
        },
        'CA19-9': {
            'type': 'number_input',
            'init_value': 0.0,
            'min_value': 0.0,
            'add_after': ' U/mL',
            'model_key': 'CA199',
            'group': 'tumor'
        },
        'CEA': {
            'type': 'number_input',
            'init_value': 0.0,
            'min_value': 0.0,
            'add_after': ' ng/mL',
            'model_key': 'CEA',
            'group': 'tumor'
        },
        'Albumin': {
            'type': 'number_input',
            'init_value': 40.0,
            'min_value': 0.01,
            'add_after': ' g/L',
            'model_key': 'Alb',
            'group': 'lab'
        },
        'Fibrinogen': {
            'type': 'number_input',
            'init_value': 3.0,
            'min_value': 0.0,
            'add_after': ' g/L',
            'model_key': 'FIB',
            'group': 'lab'
        },
        'Neutrophil count': {
            'type': 'number_input',
            'init_value': 3.5,
            'min_value': 0.0,
            'add_after': ' ×10^9/L',
            'model_key': 'NEUT',
            'group': 'lab'
        },
        'Lymphocyte count': {
            'type': 'number_input',
            'init_value': 1.5,
            'min_value': 0.01,
            'add_after': ' ×10^9/L',
            'model_key': 'LYM',
            'group': 'lab'
        },
        'Monocyte count': {
            'type': 'number_input',
            'init_value': 0.4,
            'min_value': 0.01,
            'add_after': ' ×10^9/L',
            'model_key': 'MONO',
            'group': 'lab'
        },
        'Total bilirubin': {
            'type': 'number_input',
            'init_value': 12.0,
            'min_value': 0.01,
            'add_after': ' μmol/L',
            'model_key': 'TBIL',
            'group': 'lab'
        }
    }

    input_keys = MODEL_FEATURES.copy()

    return settings, input_keys


@st.cache_resource(show_spinner=False)
def get_model():
    with open('./gbsa_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_resource(show_spinner=False)
def get_scaler():
    with open("./scaler_transform.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler


@st.cache_resource(show_spinner=False)
def get_shap_background():
    with open("./shap_background_kmeans.pkl", "rb") as f:
        shap_background = pickle.load(f)
    return shap_background


# =========================
# 3. UI输入组件
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
# 4. 输入转模型特征
# =========================
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

    numeric_cols = [
        'Age', 'CA199', 'CEA',
        'Alb', 'FIB', 'NEUT', 'LYM', 'MONO', 'TBIL'
    ]

    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    if input_df[numeric_cols].isnull().any().any():
        st.error("Some numeric inputs are invalid. Please check all laboratory values.")
        return None, None

    eps = 1e-8

    input_df['FAR'] = input_df['FIB'] / (input_df['Alb'] + eps)
    input_df['PNI'] = input_df['Alb'] + 5 * input_df['LYM']
    input_df['NLR'] = input_df['NEUT'] / (input_df['LYM'] + eps)
    input_df['ALBI'] = 0.66 * np.log10(input_df['TBIL'] + eps) - 0.085 * input_df['Alb']
    input_df['LMR'] = input_df['LYM'] / (input_df['MONO'] + eps)

    model_input_df = input_df[MODEL_FEATURES].copy()

    # scaler 训练时的连续变量顺序必须保持一致
    if hasattr(scaler, "feature_names_in_"):
        continus_var = list(scaler.feature_names_in_)
    else:
        continus_var = [
            'Age', 'CEA', 'CA199',
            'FAR', 'PNI', 'NLR', 'ALBI', 'LMR'
        ]

    model_input_df[continus_var] = scaler.transform(model_input_df[continus_var])

    return model_input_df, input_df


# =========================
# 5. 结果展示函数
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
        ("NLR", full_df.loc[0, 'NLR']),
        ("LMR", full_df.loc[0, 'LMR']),
        ("ALBI", full_df.loc[0, 'ALBI'])
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
    y_plot = chf(x_plot)

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


# =========================
# 6. SHAP计算与绘图
# =========================
def calculate_patient_shap(model, model_input_df, shap_background, nsamples=200):
    """
    计算当前患者的 individual SHAP values.
    这里解释的是 model.predict() 输出的 GBSA risk score.
    """

    feature_order = list(model_input_df.columns)

    def predict_risk(X):
        X = np.asarray(X)
        X_df = pd.DataFrame(X, columns=feature_order)
        return model.predict(X_df)

    explainer = shap.KernelExplainer(
        predict_risk,
        shap_background
    )

    shap_values = explainer.shap_values(
        model_input_df,
        nsamples=nsamples
    )

    shap_values = np.asarray(shap_values).reshape(-1)
    base_value = float(np.ravel(explainer.expected_value)[0])

    return shap_values, base_value


def format_feature_value(feature, value):
    if feature == "NAD":
        return "Yes" if int(value) == 1 else "No"

    if feature == "Stage":
        stage_reverse = {
            1: "I",
            2: "II",
            3: "III",
            4: "IV"
        }
        return stage_reverse.get(int(value), str(value))

    return f"{float(value):.3f}"

def plot_individual_shap_bar(
    model_input_df,
    full_df,
    shap_values,
    risk_score,
    cutoff
):
    """
    Individual SHAP risk contribution plot.
    红色：增加风险
    蓝色：降低风险
    """

    feature_names = model_input_df.columns.tolist()

    display_values = []
    for feature in feature_names:
        raw_value = full_df.loc[0, feature]
        display_values.append(format_feature_value(feature, raw_value))

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": display_values,
        "SHAP": shap_values
    })

    shap_df["AbsSHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("AbsSHAP", ascending=True)

    y_labels = [
        f"{row['Feature']}={row['Value']}"
        for _, row in shap_df.iterrows()
    ]

    colors = [
        "#b2182b" if v >= 0 else "#2166ac"
        for v in shap_df["SHAP"]
    ]

    risk_group = "High-risk" if risk_score >= cutoff else "Low-risk"

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.barh(
        y_labels,
        shap_df["SHAP"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1.2
    )

    for i, v in enumerate(shap_df["SHAP"]):
        ha = "left" if v >= 0 else "right"
        offset = 0.01 if v >= 0 else -0.01

        ax.text(
            v + offset,
            i,
            f"{v:+.3f}",
            va="center",
            ha=ha,
            fontsize=9
        )

    max_abs = max(abs(shap_df["SHAP"].min()), abs(shap_df["SHAP"].max()))
    if max_abs == 0:
        max_abs = 0.1

    ax.set_xlim(-max_abs * 1.30, max_abs * 1.30)

    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    ax.set_title(
        f"Individual SHAP risk contribution\n",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.text(
        0.98,
        0.03,
        "Red: increase risk\nBlue: decrease risk",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75)
    )

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig

# =========================
# 7. 预测函数
# =========================
def predict(settings, input_keys, model, scaler, shap_background):
    model_input_df, full_df = get_input_dataframe(settings, input_keys, scaler)

    if model_input_df is None:
        st.error("Data processing error. Prediction cannot be performed.")
        return

    # 保证输入列顺序和模型训练时一致
    if hasattr(model, "feature_names_in_"):
        model_input_df = model_input_df[list(model.feature_names_in_)]

    if not hasattr(model, "predict_cumulative_hazard_function"):
        st.error("The loaded model does not support cumulative hazard prediction.")
        return

    predicted_risk = model.predict(model_input_df)
    cumulative_hazard_functions = model.predict_cumulative_hazard_function(model_input_df)

    risk_score = float(predicted_risk[0])
    chf = cumulative_hazard_functions[0]

    with st.spinner("Calculating individual SHAP values..."):
        shap_values, base_value = calculate_patient_shap(
            model=model,
            model_input_df=model_input_df,
            shap_background=shap_background,
            nsamples=200
        )

    # 展示结果
    left_col, right_col = st.columns([1.0, 1.35], gap="large")

    with left_col:
        show_risk_stratification(risk_score, RISK_CUTOFF)

        st.markdown("### Survival probability")
        show_survival_metrics(chf)

        with st.expander("Detailed survival table", expanded=False):
            show_survival_table(chf)

        with st.expander("Model input features", expanded=False):
            st.dataframe(
                model_input_df.round(4),
                hide_index=True,
                use_container_width=True
            )

        with st.expander("Calculated indices", expanded=True):
            show_calculated_indices_pretty(full_df)


    with right_col:
        st.subheader("Cumulative hazard curve")
        fig = plot_cumulative_hazard_curve(chf, risk_score)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Individual SHAP risk contribution")
        shap_fig = plot_individual_shap_bar(
        model_input_df=model_input_df,
        full_df=full_df,
        shap_values=shap_values,
        risk_score=risk_score,
        cutoff=RISK_CUTOFF)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)


# =========================
# 8. 主页面
# =========================
settings, input_keys = load_setting()
model = get_model()
scaler = get_scaler()
shap_background = get_shap_background()

st.header("GBSA-based postoperative survival prediction model for colorectal cancer")

st.markdown(
    """
    <p style="font-size:24px;">
    Enter postoperative clinicopathological and laboratory parameters in the sidebar 
    to obtain individualized survival estimates and SHAP-based risk explanations.
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
        predict(
            settings=settings,
            input_keys=input_keys,
            model=model,
            scaler=scaler,
            shap_background=shap_background
        )