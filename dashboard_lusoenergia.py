from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO EXECUTIVO
# ==============================================================================
st.set_page_config(
    page_title="LusoEnergia — Portal Comercial & Sala de Controlo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 12px 16px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SOMA_ALUNOS = 101833

# Tarifário de referência residencial (Portugal - €/kWh sem IVA)
TARIFA_SIMPLES = 0.215
TARIFA_FORA_VAZIO = 0.265  # 08:00 às 22:00
TARIFA_VAZIO = 0.118       # 22:00 às 08:00


# ==============================================================================
# 1. LOCALIZAÇÃO E PREPARAÇÃO DOS DADOS (COM CACHE)
# ==============================================================================
def encontrar_dataset():
    """Procura automaticamente o ficheiro ZIP ou TXT do dataset no projeto."""
    base = Path(__file__).resolve().parent
    for raiz in [Path.cwd(), base, base.parent]:
        if raiz.exists():
            for padrao in ("*.zip", "*.txt"):
                for fich in raiz.rglob(padrao):
                    if not any(p in fich.parts for p in (".venv", "venv", "env", ".git", "__pycache__")):
                        if fich.stat().st_size > 1_000_000:
                            return str(fich)
    return None


@st.cache_data(show_spinner="A preparar indicadores comerciais e operacionais (18 meses)...")
def carregar_dados_grupo(fonte):
    df_raw = pd.read_csv(
        fonte,
        sep=";",
        na_values=["?", "nan", "NaN", ""],
        low_memory=False,
    )

    df_raw.index = pd.to_datetime(
        df_raw["Date"] + " " + df_raw["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df_raw = df_raw.drop(columns=["Date", "Time"]).sort_index()

    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    grelha = pd.date_range(df_raw.index.min(), df_raw.index.max(), freq="1min")
    df_raw = df_raw.reindex(grelha)

    # Janela determinística de 18 meses do grupo
    meses_unicos = df_raw.index.to_period("M").unique()
    idx_inicio = SOMA_ALUNOS % max(1, len(meses_unicos) - 18)
    meses_grupo = meses_unicos[idx_inicio : idx_inicio + 18]
    df_bruto = df_raw.loc[df_raw.index.to_period("M").isin(meses_grupo)].copy()

    # Qualidade dos dados brutos
    pct_missing = float(df_bruto["Global_active_power"].isna().mean() * 100.0)
    leituras_dia = df_bruto["Global_active_power"].resample("D").count()
    dias_sem_leituras = int((leituras_dia == 0).sum())
    dias_incompletos = int((leituras_dia < 1200).sum())

    # Interpolação curta (até 60 min)
    df_clean = df_bruto.interpolate(method="time", limit=60)
    df_clean["Energia_Total_Wh"] = (df_clean["Global_active_power"] * 1000.0) / 60.0
    soma_subs = (
        df_clean["Sub_metering_1"]
        + df_clean["Sub_metering_2"]
        + df_clean["Sub_metering_3"]
    )
    df_clean["Sub_outros_Wh"] = (df_clean["Energia_Total_Wh"] - soma_subs).clip(lower=0)

    # Indicadores de horário vazio (22h - 08h) vs. fora de vazio (08h - 22h)
    is_vazio = (df_clean.index.hour >= 22) | (df_clean.index.hour < 8)
    energia_total_kwh = float(df_clean["Energia_Total_Wh"].sum() / 1000.0)
    energia_vazio_kwh = float(df_clean.loc[is_vazio, "Energia_Total_Wh"].sum() / 1000.0)
    energia_fora_vazio_kwh = energia_total_kwh - energia_vazio_kwh
    pct_vazio = (energia_vazio_kwh / max(energia_total_kwh, 1.0)) * 100.0

    # Custos comparativos (Simples vs. Bi-Horária)
    custo_simples_total = energia_total_kwh * TARIFA_SIMPLES
    custo_bihoraria_total = (energia_vazio_kwh * TARIFA_VAZIO) + (energia_fora_vazio_kwh * TARIFA_FORA_VAZIO)

    carga_basal_kw = float(df_clean["Global_active_power"].quantile(0.10))
    energia_basal_kwh_dia = carga_basal_kw * 24.0

    # Tabela diária
    df_diario = pd.DataFrame(index=df_clean.resample("D").mean().index)
    df_diario["Energia_Total_kWh"] = df_clean["Energia_Total_Wh"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub1_Cozinha_kWh"] = df_clean["Sub_metering_1"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub2_Lavandaria_kWh"] = df_clean["Sub_metering_2"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub3_Climatizacao_kWh"] = df_clean["Sub_metering_3"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub_Outros_kWh"] = df_clean["Sub_outros_Wh"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Energia_Variavel_kWh"] = (df_diario["Energia_Total_kWh"] - energia_basal_kwh_dia).clip(lower=0)
    df_diario["Voltage_Media"] = df_clean["Voltage"].resample("D").mean()
    df_diario["Reactive_Media"] = df_clean["Global_reactive_power"].resample("D").mean()
    df_diario["Pico_Diario_kW"] = df_clean["Global_active_power"].resample("D").max()

    meta = {
        "inicio": str(df_clean.index.min().date()),
        "fim": str(df_clean.index.max().date()),
        "consumo_medio_diario": float(df_diario["Energia_Total_kWh"].mean()),
        "consumo_mediano_diario": float(df_diario["Energia_Total_kWh"].median()),
        "pct_missing": pct_missing,
        "dias_sem_leituras": dias_sem_leituras,
        "dias_incompletos": dias_incompletos,
        "carga_basal_kw": carga_basal_kw,
        "energia_basal_kwh_dia": energia_basal_kwh_dia,
        "leituras_dia": leituras_dia,
        "pct_vazio": pct_vazio,
        "custo_mensal_simples": custo_simples_total / 18.0,
        "custo_mensal_bihoraria": custo_bihoraria_total / 18.0,
        "poupanca_anual_bihoraria": ((custo_simples_total - custo_bihoraria_total) / 18.0) * 12.0,
    }
    return df_clean, df_diario, meta


# ==============================================================================
# 2. PROJEÇÃO 2D DOS PERFIS DIÁRIOS (PCA & t-SNE COM CACHE)
# ==============================================================================
@st.cache_data(show_spinner="A calcular mapa de perfis diários...")
def calcular_projecao_2d(df_clean, df_diario, metodo="PCA", perplexity=30):
    df_perf = df_clean[["Global_active_power"]].copy()
    df_perf["Data"] = df_perf.index.date
    df_perf["Hora"] = df_perf.index.hour
    mat_24h = df_perf.pivot_table(
        index="Data", columns="Hora", values="Global_active_power", aggfunc="median"
    ).dropna()

    X_scaled = StandardScaler().fit_transform(mat_24h.values)
    idx_datas = pd.to_datetime(mat_24h.index)
    dias_pt = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    tipo_dia = np.where(idx_datas.dayofweek >= 5, "Fim de Semana", "Dia Útil")

    if metodo == "PCA":
        modelo = PCA(n_components=2, random_state=42)
        coords = modelo.fit_transform(X_scaled)
        var_exp = modelo.explained_variance_ratio_ * 100
        eixo_x = f"Componente 1 ({var_exp[0]:.1f}% variância)"
        eixo_y = f"Componente 2 ({var_exp[1]:.1f}% variância)"
    else:
        perp_segura = min(perplexity, max(5, len(X_scaled) - 1))
        modelo = TSNE(
            n_components=2,
            perplexity=perp_segura,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        coords = modelo.fit_transform(X_scaled)
        eixo_x = "Dimensão t-SNE 1 (Vizinhança Local)"
        eixo_y = "Dimensão t-SNE 2 (Vizinhança Local)"

    df_plot = pd.DataFrame({
        "Dim1": coords[:, 0],
        "Dim2": coords[:, 1],
        "Data": idx_datas.strftime("%Y-%m-%d"),
        "Tipo de Dia": tipo_dia,
        "Dia da Semana": [dias_pt[d] for d in idx_datas.dayofweek],
        "Consumo (kWh)": df_diario.reindex(idx_datas)["Energia_Total_kWh"].round(2).values,
    })
    return df_plot, mat_24h, tipo_dia, eixo_x, eixo_y


# ==============================================================================
# 3. MODELO PREDITIVO DA PARTE III (COM CACHE)
# ==============================================================================
NOMES_NEGOCIO_VARS = {
    "lag_1": "Consumo de Ontem",
    "lag_2": "Consumo há 2 Dias",
    "lag_3": "Consumo há 3 Dias",
    "lag_7": "Mesmo Dia da Semana Passada",
    "sub1_cozinha_lag1": "Cozinha Ontem (Sub 1)",
    "sub2_lavandaria_lag1": "Lavandaria Ontem (Sub 2)",
    "sub3_climatizacao_lag1": "Climatização / Águas Ontem (Sub 3)",
    "sub_outros_lag1": "Restante Habitação Ontem (Outros)",
    "energia_variavel_lag1": "Consumo Variável Ontem",
    "voltage_lag1": "Tensão Média da Rede Ontem",
    "reactive_lag1": "Potência Reativa Ontem",
    "roll_mean_3": "Média dos Últimos 3 Dias",
    "roll_mean_7": "Média da Última Semana",
    "roll_median_7": "Mediana da Última Semana",
    "roll_std_7": "Variabilidade Semanal",
    "roll_mean_14": "Média das Últimas 2 Semanas",
    "dia_semana": "Dia da Semana",
    "is_fim_de_semana": "Fim de Semana",
    "mes": "Mês do Ano",
    "is_feriado": "Feriado Nacional",
    "mes_sin": "Ciclo Sazonal (Seno)",
    "mes_cos": "Ciclo Sazonal (Cosseno)",
}


@st.cache_data(show_spinner="A calcular previsões e erros por período...")
def executar_modelo_parte3(df_diario):
    dados = pd.DataFrame(index=df_diario.index)
    dados["Target"] = df_diario["Energia_Total_kWh"]

    for lag in (1, 2, 3, 7):
        dados[f"lag_{lag}"] = df_diario["Energia_Total_kWh"].shift(lag)

    dados["sub1_cozinha_lag1"] = df_diario["Sub1_Cozinha_kWh"].shift(1)
    dados["sub2_lavandaria_lag1"] = df_diario["Sub2_Lavandaria_kWh"].shift(1)
    dados["sub3_climatizacao_lag1"] = df_diario["Sub3_Climatizacao_kWh"].shift(1)
    dados["sub_outros_lag1"] = df_diario["Sub_Outros_kWh"].shift(1)
    dados["energia_variavel_lag1"] = df_diario["Energia_Variavel_kWh"].shift(1)
    dados["voltage_lag1"] = df_diario["Voltage_Media"].shift(1)
    dados["reactive_lag1"] = df_diario["Reactive_Media"].shift(1)

    passado = df_diario["Energia_Total_kWh"].shift(1)
    dados["roll_mean_3"] = passado.rolling(3, min_periods=2).mean()
    dados["roll_mean_7"] = passado.rolling(7, min_periods=5).mean()
    dados["roll_median_7"] = passado.rolling(7, min_periods=5).median()
    dados["roll_std_7"] = passado.rolling(7, min_periods=5).std()
    dados["roll_mean_14"] = passado.rolling(14, min_periods=10).mean()

    try:
        import holidays
        anos = list(range(df_diario.index.year.min(), df_diario.index.year.max() + 1))
        feriados_fr = holidays.France(years=anos)
        dados["is_feriado"] = dados.index.normalize().isin(feriados_fr).astype(int)
    except ImportError:
        feriados_fixos = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
        dados["is_feriado"] = [int((d.month, d.day) in feriados_fixos) for d in dados.index]

    dados["dia_semana"] = dados.index.dayofweek
    dados["is_fim_de_semana"] = (dados.index.dayofweek >= 5).astype(int)
    dados["mes"] = dados.index.month
    dados["mes_sin"] = np.sin(2 * np.pi * dados["mes"] / 12.0)
    dados["mes_cos"] = np.cos(2 * np.pi * dados["mes"] / 12.0)

    validos = dados.dropna().copy()
    X = validos.drop(columns=["Target"])
    y = validos["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            LassoCV(alphas=np.logspace(-2, 0.3, 30), cv=TimeSeriesSplit(n_splits=3), random_state=42, max_iter=10000)
        )),
        ("regressor", LinearRegression()),
    ])

    metricas, preds = [], []
    selecoes = pd.DataFrame(0, index=X.columns, columns=[f"Fold {i+1}" for i in range(5)])

    for i, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pipe_lasso.fit(X_tr, y_tr)
        y_pred = pipe_lasso.predict(X_va)
        y_base = X_va["lag_1"].values

        suporte = pipe_lasso.named_steps["selector"].get_support()
        selecoes.iloc[:, i] = suporte.astype(int)

        nome_periodo = f"Trimestre Validação {i+1}"
        intervalo_str = f"{y_va.index.min().strftime('%d/%m/%Y')} a {y_va.index.max().strftime('%d/%m/%Y')}"
        metricas.append({
            "Período": nome_periodo,
            "Datas": intervalo_str,
            "MAE Modelo (kWh)": round(mean_absolute_error(y_va, y_pred), 2),
            "MAE Baseline (kWh)": round(mean_absolute_error(y_va, y_base), 2),
            "RMSE (kWh)": round(root_mean_squared_error(y_va, y_pred), 2),
            "R²": round(r2_score(y_va, y_pred), 3),
        })

        preds.append(pd.DataFrame({
            "Data": y_va.index,
            "Real_kWh": y_va.values,
            "Previsto_kWh": y_pred,
            "Baseline_kWh": y_base,
            "Período": nome_periodo,
        }))

    df_metricas = pd.DataFrame(metricas)
    df_preds = pd.concat(preds, ignore_index=True)

    freq = (selecoes.mean(axis=1) * 100).sort_values(ascending=False)
    df_vars = pd.DataFrame({
        "Variável": [NOMES_NEGOCIO_VARS.get(c, c) for c in freq.index],
        "Importância (%)": freq.values,
    })

    return df_metricas, df_preds, df_vars


# ==============================================================================
# 4. REPLAY EM TEMPO REAL O(1) COM DIAGNÓSTICO AUTOMÁTICO DE CAUSA RAIZ
# ==============================================================================
def diagnosticar_causa_alarme(x_kw, sub1_wh, sub2_wh, sub3_wh, sub_out_wh, pot_contratada_kw):
    """Identifica automaticamente o circuito dominante ou sobrecarga no minuto do alarme."""
    prefixo = "⚡ Risco Disjuntor (> Pot. Contratada) | " if x_kw > pot_contratada_kw else ""

    circuitos = {
        "🍳 Cozinha (Forno/Loiça)": sub1_wh,
        "🧺 Lavandaria (Máq. Roupa)": sub2_wh,
        "🌡️ Climatização / Termoacumulador": sub3_wh,
        "🔌 Circuito Geral / Frio": sub_out_wh,
    }
    dominante = max(circuitos, key=circuitos.get)
    valor_dom = circuitos[dominante]

    ativos = [nome for nome, val in list(circuitos.items())[:3] if val >= 15]
    if len(ativos) >= 2:
        return prefixo + "Uso Simultâneo: " + " + ".join(ativos)
    if valor_dom < 12 and x_kw < 1.6:
        return "❄️ Arranque Normal de Compressor (Baixa Variância)"
    return prefixo + dominante


@st.cache_data(show_spinner="A processar fluxo em tempo real...")
def executar_replay_streaming(df_clean, dia_str, janela_min, limiar_z, pot_contratada_kw):
    ts_sec_arr = df_clean.index.to_numpy().astype("datetime64[s]").astype(np.int64)
    val_arr = df_clean["Global_active_power"].to_numpy(dtype=float)
    s1_arr = df_clean["Sub_metering_1"].to_numpy(dtype=float)
    s2_arr = df_clean["Sub_metering_2"].to_numpy(dtype=float)
    s3_arr = df_clean["Sub_metering_3"].to_numpy(dtype=float)
    sout_arr = df_clean["Sub_outros_Wh"].to_numpy(dtype=float)

    dia_ini_sec = int(pd.Timestamp(dia_str).value // 10**9)
    dia_fim_sec = dia_ini_sec + 86400

    n_welford = 0
    mean_welford = 0.0
    m2_welford = 0.0

    janela = deque(maxlen=janela_min)
    soma_win = 0.0
    soma_sq_win = 0.0

    alpha = 2.0 / (janela_min + 1.0)
    ewma_val = None

    total_eventos_alarme = 0
    ultimo_alarme_sec = -999999
    dias_com_alarme_set = set()

    ts_dia, raw_dia, win_dia, ewma_dia, z_dia, alarme_dia, causa_dia = [], [], [], [], [], [], []
    lista_alarmes_dia = []

    min_amostras_z = max(10, janela_min // 2)

    for ts_sec, x, s1, s2, s3, sout in zip(ts_sec_arr, val_arr, s1_arr, s2_arr, s3_arr, sout_arr):
        if np.isnan(x):
            continue

        z_atual = 0.0
        disparou = False
        causa_txt = ""
        n_win = len(janela)

        if n_win >= min_amostras_z:
            mu_loc = soma_win / n_win
            var_loc = max(0.0, (soma_sq_win - (soma_win * soma_win) / n_win) / (n_win - 1))
            std_loc = max(np.sqrt(var_loc), 0.03)
            z_atual = (x - mu_loc) / std_loc

            if z_atual > limiar_z:
                disparou = True
                causa_txt = diagnosticar_causa_alarme(x, s1, s2, s3, sout, pot_contratada_kw)
                if ts_sec - ultimo_alarme_sec >= 3600:
                    total_eventos_alarme += 1
                    ultimo_alarme_sec = ts_sec
                    dias_com_alarme_set.add(int(ts_sec // 86400))

        n_welford += 1
        delta = x - mean_welford
        mean_welford += delta / n_welford
        delta2 = x - mean_welford
        m2_welford += delta * delta2

        if n_win == janela_min:
            antigo = janela[0]
            soma_win -= antigo
            soma_sq_win -= antigo * antigo
        janela.append(x)
        soma_win += x
        soma_sq_win += x * x
        media_win = soma_win / len(janela)

        ewma_val = x if ewma_val is None else (alpha * x + (1.0 - alpha) * ewma_val)

        if dia_ini_sec <= ts_sec < dia_fim_sec:
            ts_pd = pd.Timestamp(ts_sec, unit="s")
            ts_dia.append(ts_pd)
            raw_dia.append(x)
            win_dia.append(media_win)
            ewma_dia.append(ewma_val)
            z_dia.append(z_atual)
            alarme_dia.append(disparou)
            causa_dia.append(causa_txt)
            if disparou:
                lista_alarmes_dia.append({
                    "Hora": ts_pd.strftime("%H:%M"),
                    "Potência (kW)": round(x, 2),
                    "Referência (kW)": round(media_win, 2),
                    "Z-Score": round(z_atual, 1),
                    "Causa Provável": causa_txt,
                })

    var_welford = m2_welford / max(1, n_welford - 1)
    dias_com_alarme = sorted(
        str(pd.Timestamp(d * 86400, unit="s").date()) for d in dias_com_alarme_set
    )

    df_dia = pd.DataFrame({
        "Timestamp": ts_dia,
        "Bruto_kW": raw_dia,
        "Janela_kW": win_dia,
        "EWMA_kW": ewma_dia,
        "Z_Score": z_dia,
        "Alarme": alarme_dia,
        "Causa": causa_dia,
    })

    return {
        "mean_welford": mean_welford,
        "var_welford": var_welford,
        "total_eventos": total_eventos_alarme,
        "dias_com_alarme": dias_com_alarme,
        "df_dia": df_dia,
        "tabela_alarmes_dia": pd.DataFrame(lista_alarmes_dia),
    }


# ==============================================================================
# BARRA LATERAL (NAVEGAÇÃO + PARÂMETROS COMERCIAIS DO CLIENTE)
# ==============================================================================
fonte_dados = encontrar_dataset()

with st.sidebar:
    st.markdown("## ⚡ LusoEnergia")
    st.caption("Gestão Comercial & Sala de Controlo")
    st.markdown("---")

    pagina = st.radio(
        "Área de Trabalho",
        [
            "1. Visão Geral & Faturação",
            "2. Padrões & Perfis de Cliente",
            "3. Previsão de Consumo",
            "4. Sala de Controlo (Tempo Real)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ Perfil Contratual Simulado**")
    tarifa_escolhida = st.selectbox(
        "Tarifa Atual do Cliente",
        ["Simples (0,215 €/kWh)", "Bi-Horária (0,118 € / 0,265 €)"],
    )
    pot_contratada = st.selectbox(
        "Potência Contratada",
        [3.45, 4.60, 5.75, 6.90, 10.35],
        index=3,
        format_func=lambda x: f"{x:.2f} kVA",
    )

    if fonte_dados is None:
        st.markdown("---")
        uploaded = st.file_uploader("Carregar dataset (.zip ou .txt)", type=["zip", "txt"])
        fonte_dados = uploaded

if fonte_dados is None:
    st.error("⚠️ Dataset não encontrado. Coloca o ficheiro `.zip` em `data/raw/` ou carrega-o na barra lateral.")
    st.stop()

df_clean, df_diario, meta = carregar_dados_grupo(fonte_dados)

preco_kwh_ativo = TARIFA_SIMPLES if "Simples" in tarifa_escolhida else (
    (meta["pct_vazio"] / 100.0) * TARIFA_VAZIO + (1.0 - meta["pct_vazio"] / 100.0) * TARIFA_FORA_VAZIO
)

with st.sidebar:
    st.markdown("---")
    st.caption(f"📍 Janela Grupo `{SOMA_ALUNOS}`: **{meta['inicio']}** a **{meta['fim']}**")


# ==============================================================================
# PÁGINA 1: VISÃO GERAL, FATURAÇÃO & QUALIDADE DOS DADOS
# ==============================================================================
if pagina == "1. Visão Geral & Faturação":
    st.title("📊 Visão Geral, Faturação & Qualidade dos Dados")
    st.caption("Resumo executivo do consumo, custo estimado e integridade das leituras do contador inteligente.")

    custo_medio_mensal = meta["consumo_medio_diario"] * 30.0 * preco_kwh_ativo
    dias_excesso_pot = int((df_diario["Pico_Diario_kW"] > pot_contratada).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Consumo Médio Diário",
        f"{meta['consumo_medio_diario']:.2f} kWh/dia",
        f"~{custo_medio_mensal:.2f} € / mês",
    )
    k2.metric(
        "Carga Basal Permanente",
        f"{meta['energia_basal_kwh_dia']:.2f} kWh/dia",
        f"Custo fixo: ~{meta['energia_basal_kwh_dia']*30*preco_kwh_ativo:.2f} € / mês",
    )
    k3.metric(
        "% de Missing (Minutos)",
        f"{meta['pct_missing']:.2f} %",
        "Interpolado até 60 min",
    )
    k4.metric(
        "Dias Sem Leituras (0 min)",
        f"{meta['dias_sem_leituras']} dias",
        f"{meta['dias_incompletos']} dias com < 20h válidas",
        delta_color="inverse",
    )

    if meta["poupanca_anual_bihoraria"] > 15 and "Simples" in tarifa_escolhida:
        st.success(
            f"💡 **Oportunidade Comercial:** Este cliente realiza **{meta['pct_vazio']:.1f}%** do consumo em horário vazio (22h–08h). "
            f"A mudança para **Tarifa Bi-Horária** gera uma poupança estimada de **{meta['poupanca_anual_bihoraria']:.2f} €/ano**."
        )
    elif dias_excesso_pot > 0:
        st.warning(
            f"⚠️ **Alerta de Potência Contratada ({pot_contratada:.2f} kVA):** Foram registados picos acima da potência contratada em **{dias_excesso_pot} dias**. "
            "Recomenda-se propor subida de escalão ou gestão de cargas."
        )
    else:
        st.info(
            f"✅ **Enquadramento Tarifário:** Consumo em vazio de **{meta['pct_vazio']:.1f}%**. "
            f"A potência contratada de **{pot_contratada:.2f} kVA** suporta todos os picos registados sem risco de disparo."
        )

    with st.container(border=True):
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.subheader("Evolução Diária do Consumo e Faturação")
        with col_t2:
            unidade_v1 = st.selectbox("Unidade:", ["Energia (kWh/dia)", "Custo Estimado (€/dia)", "Decomposição por Circuito"], label_visibility="collapsed")

        fig1 = go.Figure()
        if unidade_v1 == "Energia (kWh/dia)":
            fig1.add_trace(go.Scatter(
                x=df_diario.index, y=df_diario["Energia_Total_kWh"],
                mode="lines", name="Consumo Diário (kWh)",
                line=dict(color="#1f618d", width=1.6), fill="tozeroy", fillcolor="rgba(31, 97, 141, 0.10)",
            ))
            fig1.add_hline(y=meta["energia_basal_kwh_dia"], line_dash="dash", line_color="#c0392b", annotation_text="Carga Basal")
            y_label = "Energia (kWh/dia)"
        elif unidade_v1 == "Custo Estimado (€/dia)":
            fig1.add_trace(go.Scatter(
                x=df_diario.index, y=df_diario["Energia_Total_kWh"] * preco_kwh_ativo,
                mode="lines", name="Custo Diário (€)",
                line=dict(color="#117864", width=1.6), fill="tozeroy", fillcolor="rgba(17, 120, 100, 0.10)",
            ))
            fig1.add_hline(y=meta["energia_basal_kwh_dia"] * preco_kwh_ativo, line_dash="dash", line_color="#c0392b", annotation_text="Custo Basal")
            y_label = "Custo (€/dia)"
        else:
            for nome, col, cor in [
                ("Climatização & Águas (Sub 3)", "Sub3_Climatizacao_kWh", "#2980b9"),
                ("Geral & Frigorífico (Outros)", "Sub_Outros_kWh", "#7f8c8d"),
                ("Cozinha (Sub 1)", "Sub1_Cozinha_kWh", "#e67e22"),
                ("Lavandaria (Sub 2)", "Sub2_Lavandaria_kWh", "#16a085"),
            ]:
                fig1.add_trace(go.Scatter(
                    x=df_diario.index, y=df_diario[col],
                    mode="lines", name=nome, stackgroup="one", line=dict(width=0.8, color=cor),
                ))
            y_label = "Energia (kWh/dia)"

        fig1.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=y_label, hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

    c_esq, c_dir = st.columns([1.45, 1.0])
    with c_esq:
        with st.container(border=True):
            st.subheader("Integridade das Leituras por Dia")
            df_cob = meta["leituras_dia"].reset_index()
            df_cob.columns = ["Data", "Minutos"]
            df_cob["Estado"] = np.where(df_cob["Minutos"] == 0, "Sem Leituras (0 min)",
                               np.where(df_cob["Minutos"] < 1200, "Incompleto (< 20h)", "Completo (≥ 20h)"))
            fig_cob = px.bar(
                df_cob, x="Data", y="Minutos", color="Estado",
                color_discrete_map={"Completo (≥ 20h)": "#2e86c1", "Incompleto (< 20h)": "#f39c12", "Sem Leituras (0 min)": "#c0392b"},
            )
            fig_cob.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Minutos / Dia")
            st.plotly_chart(fig_cob, use_container_width=True)

    with c_dir:
        with st.container(border=True):
            st.subheader("Distribuição da Fatura por Circuito")
            df_pie = pd.DataFrame({
                "Circuito": ["Climatização (Sub 3)", "Geral & Frio (Outros)", "Cozinha (Sub 1)", "Lavandaria (Sub 2)"],
                "Euros_Mes": [
                    df_diario["Sub3_Climatizacao_kWh"].mean() * 30 * preco_kwh_ativo,
                    df_diario["Sub_Outros_kWh"].mean() * 30 * preco_kwh_ativo,
                    df_diario["Sub1_Cozinha_kWh"].mean() * 30 * preco_kwh_ativo,
                    df_diario["Sub2_Lavandaria_kWh"].mean() * 30 * preco_kwh_ativo,
                ],
            })
            fig_pie = px.pie(df_pie, names="Circuito", values="Euros_Mes", hole=0.5,
                             color_discrete_sequence=["#2980b9", "#7f8c8d", "#e67e22", "#16a085"])
            fig_pie.update_traces(texttemplate="%{label}<br>%{value:.1f} €/mês")
            fig_pie.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# PÁGINA 2: PADRÕES DE CONSUMO & PERFIS DE CLIENTE (PCA / t-SNE)
# ==============================================================================
elif pagina == "2. Padrões & Perfis de Cliente":
    st.title("📈 Padrões de Consumo & Perfis de Cliente")
    st.caption("Análise de sazonalidade e segmentação dos hábitos diários (Dia Útil vs. Fim de Semana).")

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.0, 1.3, 1.3])
        with f1:
            granularidade = st.radio("Granularidade", ["Horária", "Diária", "Mensal"], horizontal=True, index=1)
        with f2:
            opcoes_vars = {
                "Potência Ativa Global (kW)": "Global_active_power",
                "Climatização / Águas — Sub 3 (Wh/min)": "Sub_metering_3",
                "Cozinha — Sub 1 (Wh/min)": "Sub_metering_1",
                "Lavandaria — Sub 2 (Wh/min)": "Sub_metering_2",
                "Geral / Frigorífico — Outros (Wh/min)": "Sub_outros_Wh",
            }
            var_nome = st.selectbox("Variável / Circuito", list(opcoes_vars.keys()))
            var_col = opcoes_vars[var_nome]
        with f3:
            d_min, d_max = df_clean.index.min().date(), df_clean.index.max().date()
            datas = st.date_input("Intervalo de Datas", value=(d_min, d_max), min_value=d_min, max_value=d_max)

        d_ini, d_fim = datas if (isinstance(datas, tuple) and len(datas) == 2) else (d_min, d_max)
        df_sub = df_clean.loc[str(d_ini):str(d_fim), [var_col]]

        if granularidade == "Horária":
            serie = df_sub.groupby(df_sub.index.floor("h"))[var_col].median().reset_index()
        elif granularidade == "Diária":
            serie = df_sub.resample("D")[var_col].mean().reset_index()
        else:
            serie = df_sub.groupby(df_sub.index.to_period("M").to_timestamp())[var_col].median().reset_index()
        serie.columns = ["Data", "Valor"]

        fig_saz = px.line(serie, x="Data", y="Valor", color_discrete_sequence=["#1f618d"])
        if var_col == "Global_active_power":
            fig_saz.add_hline(y=meta["carga_basal_kw"], line_dash="dash", line_color="#c0392b", annotation_text="Carga Basal")
        fig_saz.update_layout(
            title=f"Evolução ({granularidade}) — {var_nome}",
            height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=var_nome, xaxis_title="",
        )
        st.plotly_chart(fig_saz, use_container_width=True)

    col_pca, col_curva = st.columns([1.3, 1.0])

    with col_pca:
        with st.container(border=True):
            c_top1, c_top2 = st.columns([1.35, 1.0])
            with c_top1:
                st.subheader("Mapa de Perfis Diários (24h)")
            with c_top2:
                metodo_proj = st.radio(
                    "Método de Projeção",
                    ["PCA", "t-SNE"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

            if metodo_proj == "t-SNE":
                perp_val = st.slider(
                    "Perplexity t-SNE (vizinhos locais por grupo)",
                    min_value=10, max_value=50, value=30, step=5,
                )
                st.caption("t-SNE agrupa dias com rotinas semelhantes em ilhas locais. Passe o rato para ver a data e tipo de dia.")
            else:
                perp_val = 30
                st.caption("PCA (padrão): Componente 1 reflete o volume de consumo e Componente 2 a distribuição horária.")

            df_proj, mat_24h, tipo_dia, eixo_x, eixo_y = calcular_projecao_2d(
                df_clean, df_diario, metodo=metodo_proj, perplexity=perp_val
            )

            fig_pca = px.scatter(
                df_proj, x="Dim1", y="Dim2", color="Tipo de Dia",
                color_discrete_map={"Dia Útil": "#2980b9", "Fim de Semana": "#e67e22"},
                hover_data={
                    "Data": True,
                    "Tipo de Dia": True,
                    "Dia da Semana": True,
                    "Consumo (kWh)": True,
                    "Dim1": ":.2f",
                    "Dim2": ":.2f",
                },
            )
            fig_pca.update_traces(marker=dict(size=8, opacity=0.82))
            fig_pca.update_layout(
                height=320 if metodo_proj == "t-SNE" else 355,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title=eixo_x,
                yaxis_title=eixo_y,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

    with col_curva:
        with st.container(border=True):
            st.subheader("Curva Horária Típica (24h)")
            st.caption("Consumo típico ao longo do dia (zona sombreada = horário vazio).")
            mat_24h_tipo = mat_24h.copy()
            mat_24h_tipo["Tipo de Dia"] = tipo_dia
            perfis_medios = mat_24h_tipo.groupby("Tipo de Dia").median().T.reset_index()
            perfis_melt = perfis_medios.melt(id_vars="Hora", var_name="Tipo de Dia", value_name="Potência (kW)")

            fig_hora = px.line(
                perfis_melt, x="Hora", y="Potência (kW)", color="Tipo de Dia", markers=True,
                color_discrete_map={"Dia Útil": "#2980b9", "Fim de Semana": "#e67e22"},
            )
            fig_hora.add_vrect(x0=0, x1=8, fillcolor="green", opacity=0.07, line_width=0)
            fig_hora.add_vrect(x0=22, x1=23, fillcolor="green", opacity=0.07, line_width=0)
            fig_hora.update_layout(height=355, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(dtick=3))
            st.plotly_chart(fig_hora, use_container_width=True)


# ==============================================================================
# PÁGINA 3: PREVISÃO DO DIA SEGUINTE
# ==============================================================================
elif pagina == "3. Previsão de Consumo":
    st.title("🔮 Previsão de Compra de Energia (Dia Seguinte)")
    st.caption("Estimativa diária para planeamento de compra de energia no mercado grossista.")

    df_metricas, df_preds, df_vars = executar_modelo_parte3(df_diario)

    c_f1, c_f2 = st.columns([1.2, 1.8])
    with c_f1:
        periodo_sel = st.selectbox(
            "Período de Avaliação",
            ["Todos os Períodos (Global)"] + df_metricas["Período"].tolist(),
        )
    with c_f2:
        mostrar_baseline = st.checkbox("Comparar com Referência Ingénua (Repetir Consumo de Ontem)", value=True)

    if periodo_sel == "Todos os Períodos (Global)":
        df_preds_view = df_preds
        mae_mod = df_metricas["MAE Modelo (kWh)"].mean()
        mae_bas = df_metricas["MAE Baseline (kWh)"].mean()
        rmse_mod = df_metricas["RMSE (kWh)"].mean()
        r2_mod = df_metricas["R²"].mean()
    else:
        df_preds_view = df_preds[df_preds["Período"] == periodo_sel]
        linha_m = df_metricas[df_metricas["Período"] == periodo_sel].iloc[0]
        mae_mod = linha_m["MAE Modelo (kWh)"]
        mae_bas = linha_m["MAE Baseline (kWh)"]
        rmse_mod = linha_m["RMSE (kWh)"]
        r2_mod = linha_m["R²"]

    ganho_pct = ((mae_bas - mae_mod) / mae_bas) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Desvio Médio (MAE Modelo)", f"{mae_mod:.2f} kWh/dia", f"{ganho_pct:+.1f}% vs. Referência")
    m2.metric("Impacto Financeiro do Desvio", f"±{mae_mod * preco_kwh_ativo:.2f} € / dia", f"Ref.: ±{mae_bas * preco_kwh_ativo:.2f} € / dia", delta_color="off")
    m3.metric("Erro Quadrático (RMSE)", f"{rmse_mod:.2f} kWh/dia")
    m4.metric("Qualidade de Ajuste (R²)", f"{r2_mod:.3f}")

    with st.container(border=True):
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_preds_view["Data"], y=df_preds_view["Real_kWh"],
            mode="lines", name="Consumo Real (kWh)", line=dict(color="#1c2833", width=2.0),
        ))
        fig_pred.add_trace(go.Scatter(
            x=df_preds_view["Data"], y=df_preds_view["Previsto_kWh"],
            mode="lines", name="Previsão Modelo LusoEnergia", line=dict(color="#e67e22", width=1.9, dash="dash"),
        ))
        if mostrar_baseline:
            fig_pred.add_trace(go.Scatter(
                x=df_preds_view["Data"], y=df_preds_view["Baseline_kWh"],
                mode="lines", name="Referência (Ontem)", line=dict(color="#95a5a6", width=1.2, dash="dot"),
            ))
        fig_pred.add_hline(y=meta["energia_basal_kwh_dia"], line_dash="dash", line_color="#c0392b", annotation_text="Carga Basal")
        fig_pred.update_layout(
            title=f"Consumo Real vs. Previsto ({periodo_sel})",
            height=350, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Energia (kWh/dia)", hovermode="x unified",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    col_err, col_vars = st.columns([1.25, 1.0])
    with col_err:
        with st.container(border=True):
            st.subheader("Erro de Previsão por Período")
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(x=df_metricas["Período"], y=df_metricas["MAE Modelo (kWh)"], name="Modelo LusoEnergia", marker_color="#1f618d"))
            fig_err.add_trace(go.Bar(x=df_metricas["Período"], y=df_metricas["MAE Baseline (kWh)"], name="Referência (Ontem)", marker_color="#bdc3c7"))
            fig_err.update_layout(barmode="group", height=230, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MAE (kWh/dia)")
            st.plotly_chart(fig_err, use_container_width=True)
            st.dataframe(df_metricas[["Período", "Datas", "MAE Modelo (kWh)", "MAE Baseline (kWh)", "R²"]], use_container_width=True, hide_index=True)

    with col_vars:
        with st.container(border=True):
            st.subheader("Indicadores Selecionados (Parte III)")
            st.caption("Variáveis com maior peso na previsão diária.")
            df_vars_ativas = df_vars[df_vars["Importância (%)"] > 0]
            fig_v = px.bar(
                df_vars_ativas, x="Importância (%)", y="Variável", orientation="h",
                color_discrete_sequence=["#2e86c1"],
            )
            fig_v.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_v, use_container_width=True)


# ==============================================================================
# PÁGINA 4: SALA DE CONTROLO — MONITORIZAÇÃO EM TEMPO REAL
# ==============================================================================
elif pagina == "4. Sala de Controlo (Tempo Real)":
    st.title("🚨 Sala de Controlo: Monitorização em Tempo Real")
    st.caption("Deteção online de picos anómalos (Welford / EWMA / Z-Score) com diagnóstico automático de circuito.")

    p1, p2, p3, _ = st.columns([1, 1, 1, 1.4])
    if p1.button("Modo Sensível (30m | Z=2.5)", use_container_width=True):
        st.session_state["win_slider"] = 30
        st.session_state["z_slider"] = 2.5
    if p2.button("Modo Padrão (60m | Z=3.5)", use_container_width=True):
        st.session_state["win_slider"] = 60
        st.session_state["z_slider"] = 3.5
    if p3.button("Modo Recomendado (120m | Z=3.5)", use_container_width=True):
        st.session_state["win_slider"] = 120
        st.session_state["z_slider"] = 3.5

    if "win_slider" not in st.session_state:
        st.session_state["win_slider"] = 120
    if "z_slider" not in st.session_state:
        st.session_state["z_slider"] = 3.5

    with st.container(border=True):
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            janela_val = st.slider(
                "Janela de Referência Deslizante (minutos)",
                min_value=15, max_value=180, step=15, key="win_slider",
            )
        with c_s2:
            limiar_val = st.slider(
                "Sensibilidade do Alarme — Limiar Z-Score (Z > θ)",
                min_value=2.0, max_value=5.5, step=0.25, key="z_slider",
            )

    dias_disp = sorted(df_clean.index.normalize().unique())
    dia_default = str(dias_disp[len(dias_disp) // 2].date())
    res_meta = executar_replay_streaming(df_clean, dia_default, janela_val, limiar_val, pot_contratada)

    col_d1, col_d2 = st.columns([1.2, 1.5])
    with col_d1:
        modo_dia = st.radio(
            "Seleção do Dia de Operação",
            ["Saltar para dias com alarmes", "Escolher data no calendário"],
            horizontal=True,
        )
    with col_d2:
        if modo_dia == "Saltar para dias com alarmes" and len(res_meta["dias_com_alarme"]) > 0:
            dia_sel_str = st.selectbox(
                f"Dias com eventos detetados ({len(res_meta['dias_com_alarme'])} dias):",
                res_meta["dias_com_alarme"],
            )
        else:
            dia_dt = st.date_input(
                "Data para inspeção ao minuto:",
                value=dias_disp[len(dias_disp) // 2].date(),
                min_value=dias_disp[0].date(),
                max_value=dias_disp[-1].date(),
            )
            dia_sel_str = str(dia_dt)

    res = executar_replay_streaming(df_clean, dia_sel_str, janela_val, limiar_val, pot_contratada)
    df_dia_st = res["df_dia"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eventos de Alarme (18 Meses)", f"{res['total_eventos']} eventos", f"Média: {res['total_eventos']/18:.1f} / mês", delta_color="off")
    k2.metric(f"Minutos em Alerta ({dia_sel_str})", f"{int(df_dia_st['Alarme'].sum())} min")
    k3.metric("Potência Média Global (Welford)", f"{res['mean_welford']:.3f} kW")
    k4.metric("Variância Global (Welford)", f"{res['var_welford']:.3f} kW²")

    with st.container(border=True):
        fig_st = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.70, 0.30],
            subplot_titles=(
                f"Monitorização em Tempo Real ({dia_sel_str}) — Potência Instantânea vs. Tendência",
                f"Indicador de Anomalia (Z-Score Positivo > {limiar_val})",
            ),
        )

        fig_st.add_trace(go.Scatter(
            x=df_dia_st["Timestamp"], y=df_dia_st["Bruto_kW"],
            mode="lines", name="Potência Instantânea (kW)", line=dict(color="#95a5a6", width=1.0), opacity=0.65,
        ), row=1, col=1)
        fig_st.add_trace(go.Scatter(
            x=df_dia_st["Timestamp"], y=df_dia_st["Janela_kW"],
            mode="lines", name=f"Média Deslizante ({janela_val} min)", line=dict(color="#1f618d", width=2.2),
        ), row=1, col=1)
        fig_st.add_trace(go.Scatter(
            x=df_dia_st["Timestamp"], y=df_dia_st["EWMA_kW"],
            mode="lines", name="Tendência Rápida (EWMA)", line=dict(color="#d35400", width=2.0, dash="dash"),
        ), row=1, col=1)

        fig_st.add_hline(
            y=pot_contratada, line_dash="dot", line_color="#8e44ad",
            annotation_text=f"Pot. Contratada ({pot_contratada:.2f} kVA)", row=1, col=1,
        )

        pts_alarme = df_dia_st[df_dia_st["Alarme"]]
        if not pts_alarme.empty:
            fig_st.add_trace(go.Scatter(
                x=pts_alarme["Timestamp"], y=pts_alarme["Bruto_kW"],
                mode="markers", name="Alarme Disparado",
                text=pts_alarme["Causa"],
                hovertemplate="<b>%{x|%H:%M}</b><br>Potência: %{y:.2f} kW<br>%{text}<extra></extra>",
                marker=dict(color="#c0392b", size=9, symbol="x"),
            ), row=1, col=1)

        fig_st.add_trace(go.Scatter(
            x=df_dia_st["Timestamp"], y=df_dia_st["Z_Score"],
            mode="lines", name="Z-Score", line=dict(color="#5b2c6f", width=1.4), showlegend=False,
        ), row=2, col=1)
        fig_st.add_hline(y=limiar_val, line_dash="dot", line_color="#c0392b", row=2, col=1)

        fig_st.update_yaxes(title_text="Potência (kW)", row=1, col=1)
        fig_st.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig_st.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), hovermode="x unified")
        st.plotly_chart(fig_st, use_container_width=True)

    if not res["tabela_alarmes_dia"].empty:
        st.subheader(f"📋 Diagnóstico dos {len(res['tabela_alarmes_dia'])} Minutos em Alarme ({dia_sel_str})")
        st.dataframe(res["tabela_alarmes_dia"], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum pico anómalo registado neste dia com os parâmetros atuais.")