import os
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Configuração da página
st.set_page_config(
    page_title="LusoEnergia — Plataforma de Gestão & Controlo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

SOMA_ALUNOS = 101833


def encontrar_dataset():
    """Procura automaticamente o ficheiro ZIP ou TXT do dataset em data/raw e subpastas."""
    base_script = Path(__file__).resolve().parent
    caminhos_diretos = [
        Path.cwd() / "data" / "raw" / "individual+household+electric+power+consumption.zip",
        base_script / "data" / "raw" / "individual+household+electric+power+consumption.zip",
        base_script.parent / "data" / "raw" / "individual+household+electric+power+consumption.zip",
    ]
    for c in caminhos_diretos:
        if c.exists():
            return str(c)

    raizes = [Path.cwd(), base_script, base_script.parent]
    for raiz in raizes:
        if raiz.exists():
            for padrao in ("*.zip", "*.txt"):
                for fich in raiz.rglob(padrao):
                    if not any(p in fich.parts for p in (".venv", "venv", "env", ".git", "__pycache__")):
                        if fich.stat().st_size > 1_000_000:
                            return str(fich)
    return None


# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS (COM CACHE PARA RAPIDEZ)
# ==============================================================================
@st.cache_data(show_spinner="A carregar e preparar a janela de 18 meses...")
def carregar_dados_grupo(fonte):
    df_raw = pd.read_csv(
        fonte,
        sep=";",
        na_values=["?", "nan", "NaN", ""],
        low_memory=False
    )

    dt = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"], format="%d/%m/%Y %H:%M:%S")
    df_raw = df_raw.drop(columns=["Date", "Time"])
    df_raw.index = dt
    df_raw = df_raw.sort_index()

    # Converter colunas para numérico
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Reindexar para garantir grelha contínua de 1 minuto
    grelha_completa = pd.date_range(df_raw.index.min(), df_raw.index.max(), freq="1min")
    df_raw = df_raw.reindex(grelha_completa)

    # Selecionar janela determinística de 18 meses do grupo
    meses_unicos = df_raw.index.to_period("M").unique()
    total_meses = len(meses_unicos)
    max_inicio = max(1, total_meses - 18)
    idx_inicio = SOMA_ALUNOS % max_inicio
    meses_grupo = meses_unicos[idx_inicio : idx_inicio + 18]

    mask_grupo = df_raw.index.to_period("M").isin(meses_grupo)
    df_bruto_grupo = df_raw.loc[mask_grupo].copy()

    # Métricas de qualidade ANTES da interpolação
    pct_missing = float(df_bruto_grupo["Global_active_power"].isna().mean() * 100.0)
    leituras_por_dia = df_bruto_grupo["Global_active_power"].resample("D").count()
    dias_sem_leituras = int((leituras_por_dia == 0).sum())
    dias_incompletos = int((leituras_por_dia < 1200).sum())

    # Interpolação curta (até 60 min) igual às Partes I, II e III
    df_clean = df_bruto_grupo.interpolate(method="time", limit=60)

    # Variáveis derivadas físicas
    df_clean["Energia_Total_Wh"] = (df_clean["Global_active_power"] * 1000.0) / 60.0
    soma_subs = df_clean["Sub_metering_1"] + df_clean["Sub_metering_2"] + df_clean["Sub_metering_3"]
    df_clean["Sub_outros_Wh"] = (df_clean["Energia_Total_Wh"] - soma_subs).clip(lower=0)

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

    meta_qualidade = {
        "inicio": str(df_clean.index.min().date()),
        "fim": str(df_clean.index.max().date()),
        "total_dias": len(df_diario),
        "consumo_medio_diario": float(df_diario["Energia_Total_kWh"].mean()),
        "consumo_mediano_diario": float(df_diario["Energia_Total_kWh"].median()),
        "pct_missing": pct_missing,
        "dias_sem_leituras": dias_sem_leituras,
        "dias_incompletos": dias_incompletos,
        "carga_basal_kw": carga_basal_kw,
        "energia_basal_kwh_dia": energia_basal_kwh_dia,
        "leituras_por_dia": leituras_por_dia,
    }

    return df_clean, df_diario, meta_qualidade


# ==============================================================================
# 2. MODELO PREDITIVO DA PARTE III (CACHE)
# ==============================================================================
@st.cache_data(show_spinner="A executar validação temporal da Parte III...")
def executar_modelo_parte3(df_diario):
    dados = pd.DataFrame(index=df_diario.index)
    dados["Target_Energia_kWh"] = df_diario["Energia_Total_kWh"]

    # Lags
    dados["lag_1"] = df_diario["Energia_Total_kWh"].shift(1)
    dados["lag_2"] = df_diario["Energia_Total_kWh"].shift(2)
    dados["lag_3"] = df_diario["Energia_Total_kWh"].shift(3)
    dados["lag_7"] = df_diario["Energia_Total_kWh"].shift(7)

    # Circuitos ontem
    dados["sub1_cozinha_lag1"] = df_diario["Sub1_Cozinha_kWh"].shift(1)
    dados["sub2_lavandaria_lag1"] = df_diario["Sub2_Lavandaria_kWh"].shift(1)
    dados["sub3_climatizacao_lag1"] = df_diario["Sub3_Climatizacao_kWh"].shift(1)
    dados["sub_outros_lag1"] = df_diario["Sub_Outros_kWh"].shift(1)
    dados["energia_variavel_lag1"] = df_diario["Energia_Variavel_kWh"].shift(1)
    dados["voltage_lag1"] = df_diario["Voltage_Media"].shift(1)
    dados["reactive_lag1"] = df_diario["Reactive_Media"].shift(1)

    # Janelas móveis causais (shift antes do rolling)
    passado = df_diario["Energia_Total_kWh"].shift(1)
    dados["roll_mean_3"] = passado.rolling(3, min_periods=2).mean()
    dados["roll_mean_7"] = passado.rolling(7, min_periods=5).mean()
    dados["roll_median_7"] = passado.rolling(7, min_periods=5).median()
    dados["roll_std_7"] = passado.rolling(7, min_periods=5).std()
    dados["roll_mean_14"] = passado.rolling(14, min_periods=10).mean()

    # Calendário
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

    dados_validos = dados.dropna().copy()
    X = dados_validos.drop(columns=["Target_Energia_kWh"])
    y = dados_validos["Target_Energia_kWh"]

    tscv = TimeSeriesSplit(n_splits=5)
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            LassoCV(alphas=np.logspace(-2, 0.3, 30), cv=TimeSeriesSplit(n_splits=3), random_state=42, max_iter=10000)
        )),
        ("regressor", LinearRegression())
    ])

    metricas_folds = []
    previsoes_oos = []
    selecoes = pd.DataFrame(0, index=X.columns, columns=[f"Fold {i+1}" for i in range(5)])

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        pipe_lasso.fit(X_tr, y_tr)
        y_pred = pipe_lasso.predict(X_va)
        y_base = X_va["lag_1"].values

        suporte = pipe_lasso.named_steps["selector"].get_support()
        selecoes.iloc[:, fold_idx] = suporte.astype(int)

        periodo_str = f"{y_va.index.min().strftime('%Y-%m-%d')} a {y_va.index.max().strftime('%Y-%m-%d')}"
        metricas_folds.append({
            "Período (Fold)": f"Fold {fold_idx+1} ({periodo_str})",
            "MAE Modelo (kWh)": round(mean_absolute_error(y_va, y_pred), 2),
            "MAE Baseline (kWh)": round(mean_absolute_error(y_va, y_base), 2),
            "RMSE Modelo (kWh)": round(root_mean_squared_error(y_va, y_pred), 2),
            "R2 Modelo": round(r2_score(y_va, y_pred), 3),
            "Nº Variáveis": int(suporte.sum())
        })

        df_fold_pred = pd.DataFrame({
            "Data": y_va.index,
            "Real_kWh": y_va.values,
            "Previsto_Lasso_kWh": y_pred,
            "Baseline_Ontem_kWh": y_base,
            "Fold": f"Fold {fold_idx+1}"
        })
        previsoes_oos.append(df_fold_pred)

    df_metricas = pd.DataFrame(metricas_folds)
    df_preds = pd.concat(previsoes_oos, ignore_index=True)
    freq_selecao = (selecoes.mean(axis=1) * 100).sort_values(ascending=False)

    return df_metricas, df_preds, freq_selecao


# ==============================================================================
# 3. REPLAY DE STREAMING DA PARTE IV (ELEMENTO A ELEMENTO)
# ==============================================================================
@st.cache_data(show_spinner="A simular fluxo em tempo real (Welford + EWMA + Z-Score)...")
def executar_replay_streaming(df_clean, dia_selecionado_str, janela_min, limiar_z):
    timestamps_raw = df_clean.index.to_numpy()
    valores_raw = df_clean["Global_active_power"].to_numpy(dtype=float)

    n_welford = 0
    mean_welford = 0.0
    m2_welford = 0.0

    janela = deque(maxlen=janela_min)
    soma_janela = 0.0

    alpha = 2.0 / (janela_min + 1.0)
    ewma_val = None

    total_alarmes_globais = 0
    ultimo_alarme_global = None

    dia_alvo = pd.to_datetime(dia_selecionado_str).date()
    ts_dia, raw_dia, win_dia, ewma_dia, z_dia, is_alarme_dia = [], [], [], [], [], []
    alarmes_dia_lista = []

    for ts, x in zip(timestamps_raw, valores_raw):
        if np.isnan(x):
            continue

        ts_pd = pd.Timestamp(ts)

        # Avaliar Z-Score ANTES de atualizar a janela (sem contaminação)
        z_atual = 0.0
        disparou = False
        if len(janela) >= max(10, janela_min // 2):
            mu_loc = soma_janela / len(janela)
            var_loc = sum((v - mu_loc) ** 2 for v in janela) / (len(janela) - 1)
            std_loc = max(np.sqrt(var_loc), 0.02)
            z_atual = (x - mu_loc) / std_loc
            if abs(z_atual) > limiar_z:
                disparou = True
                if ultimo_alarme_global is None or (ts_pd - ultimo_alarme_global).total_seconds() >= 3600:
                    total_alarmes_globais += 1
                    ultimo_alarme_global = ts_pd

        # Welford online
        n_welford += 1
        delta = x - mean_welford
        mean_welford += delta / n_welford
        delta2 = x - mean_welford
        m2_welford += delta * delta2

        # Janela deslizante O(1)
        if len(janela) == janela_min:
            soma_janela -= janela[0]
        janela.append(x)
        soma_janela += x
        media_win = soma_janela / len(janela)

        # EWMA online
        ewma_val = x if ewma_val is None else (alpha * x + (1.0 - alpha) * ewma_val)

        # Registar apenas o dia escolhido para visualização detalhada
        if ts_pd.date() == dia_alvo:
            ts_dia.append(ts_pd)
            raw_dia.append(x)
            win_dia.append(media_win)
            ewma_dia.append(ewma_val)
            z_dia.append(z_atual)
            is_alarme_dia.append(disparou)
            if disparou:
                alarmes_dia_lista.append({
                    "Hora": ts_pd.strftime("%H:%M"),
                    "Potência (kW)": round(x, 3),
                    "Média Janela (kW)": round(media_win, 3),
                    "Z-Score": round(z_atual, 2)
                })

    var_welford = m2_welford / max(1, n_welford - 1)

    df_dia_stream = pd.DataFrame({
        "Timestamp": ts_dia,
        "Bruto_kW": raw_dia,
        "Janela_kW": win_dia,
        "EWMA_kW": ewma_dia,
        "Z_Score": z_dia,
        "Alarme": is_alarme_dia
    })

    return {
        "n_leituras": n_welford,
        "mean_welford": mean_welford,
        "var_welford": var_welford,
        "total_alarmes_globais": total_alarmes_globais,
        "df_dia": df_dia_stream,
        "alarmes_dia": pd.DataFrame(alarmes_dia_lista)
    }


# ==============================================================================
# INTERFACE PRINCIPAL DO DASHBOARD
# ==============================================================================
st.sidebar.title("⚡ LusoEnergia")
st.sidebar.caption("Gestão Comercial & Sala de Controlo")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Carregar ficheiro ZIP/TXT (opcional se já estiver em data/raw/)",
    type=["zip", "txt"]
)

fonte_dados = uploaded_file if uploaded_file is not None else encontrar_dataset()

if fonte_dados is None:
    st.error(f"⚠️ Nenhum ficheiro >1MB encontrado a partir de `{Path.cwd()}`. Carrega-o na barra lateral à esquerda.")
    st.stop()

df_clean, df_diario, meta = carregar_dados_grupo(fonte_dados)

st.sidebar.success(f"Janela Ativa (Grupo {SOMA_ALUNOS}):\n**{meta['inicio']}** a **{meta['fim']}**")
st.sidebar.info(f"Carga Basal Constante (P10):\n**{meta['carga_basal_kw']:.3f} kW** ({meta['energia_basal_kwh_dia']:.2f} kWh/dia)")

st.title("⚡ LusoEnergia — Dashboard Operacional e Analítico")
st.markdown("Ferramenta interativa de apoio à decisão comercial (perfilagem e previsão) e operação em tempo real (monitorização e alarmística).")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Visão Geral & Qualidade",
    "2. Padrões & PCA",
    "3. Previsão (Dia Seguinte)",
    "4. Monitorização em Tempo Real"
])

# ------------------------------------------------------------------------------
# TAB 1: VISÃO GERAL E QUALIDADE DOS DADOS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Indicadores-Chave (KPIs) da Janela de 18 Meses")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Consumo Médio Diário",
        f"{meta['consumo_medio_diario']:.2f} kWh/dia",
        delta=f"Mediana: {meta['consumo_mediano_diario']:.2f} kWh/dia",
        delta_color="off"
    )
    c2.metric(
        "% de Missing (Minutos)",
        f"{meta['pct_missing']:.2f} %",
        delta="Interpolação até 60 min",
        delta_color="off"
    )
    c3.metric(
        "Dias Sem Leituras (0 min)",
        f"{meta['dias_sem_leituras']} dias",
        delta=f"{meta['dias_incompletos']} dias com < 20h válidas",
        delta_color="inverse"
    )
    c4.metric(
        "Carga Basal Constante (P10)",
        f"{meta['energia_basal_kwh_dia']:.2f} kWh/dia",
        delta=f"{meta['carga_basal_kw']*1000:.0f} W contínuos",
        delta_color="off"
    )

    st.markdown("---")
    col_esq, col_dir = st.columns([1.6, 1.0])

    with col_esq:
        st.markdown("**Evolução Diária do Consumo Total vs. Carga Basal Inevitável**")
        fig_geral = go.Figure()
        fig_geral.add_trace(go.Scatter(
            x=df_diario.index, y=df_diario["Energia_Total_kWh"],
            mode="lines", name="Consumo Diário (kWh)",
            line=dict(color="#1f618d", width=1.4)
        ))
        fig_geral.add_hline(
            y=meta["energia_basal_kwh_dia"], line_dash="dash", line_color="#c0392b",
            annotation_text=f"Carga Basal Constante ({meta['energia_basal_kwh_dia']:.2f} kWh/dia)"
        )
        fig_geral.update_layout(height=370, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Energia (kWh/dia)")
        st.plotly_chart(fig_geral, use_container_width=True)

    with col_dir:
        st.markdown("**Repartição Média de Energia por Circuito**")
        circuitos_kwh = pd.DataFrame({
            "Circuito": ["Climatização/Águas (Sub 3)", "Cozinha (Sub 1)", "Lavandaria (Sub 2)", "Geral & Frio (Outros)"],
            "kWh_dia": [
                df_diario["Sub3_Climatizacao_kWh"].mean(),
                df_diario["Sub1_Cozinha_kWh"].mean(),
                df_diario["Sub2_Lavandaria_kWh"].mean(),
                df_diario["Sub_Outros_kWh"].mean()
            ]
        })
        fig_pie = px.pie(
            circuitos_kwh, names="Circuito", values="kWh_dia", hole=0.45,
            color_discrete_sequence=["#2980b9", "#e67e22", "#16a085", "#7f8c8d"]
        )
        fig_pie.update_layout(height=370, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Cobertura de Leituras Válidas por Dia (Minutos lidos / 1440)**")
    df_cobertura = meta["leituras_por_dia"].reset_index()
    df_cobertura.columns = ["Data", "Minutos_Validos"]
    fig_cob = px.bar(df_cobertura, x="Data", y="Minutos_Validos", color_discrete_sequence=["#2e86c1"])
    fig_cob.add_hline(y=1200, line_dash="dot", line_color="#c0392b", annotation_text="Limiar 20h (1200 min)")
    fig_cob.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Minutos/dia")
    st.plotly_chart(fig_cob, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: PADRÕES DE CONSUMO & PCA
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Exploração de Sazonalidade e Segmentação de Perfis Diários (PCA)")

    f1, f2, f3 = st.columns([1.0, 1.2, 1.4])
    with f1:
        granularidade = st.selectbox(
            "Granularidade Temporal",
            ["Horária", "Diária", "Mensal"],
            index=1
        )
    with f2:
        mapa_vars = {
            "Potência Ativa Global (kW)": "Global_active_power",
            "Cozinha — Sub 1 (Wh/min)": "Sub_metering_1",
            "Lavandaria — Sub 2 (Wh/min)": "Sub_metering_2",
            "Climatização/Águas — Sub 3 (Wh/min)": "Sub_metering_3",
            "Geral/Frio — Outros (Wh/min)": "Sub_outros_Wh"
        }
        var_label = st.selectbox("Variável / Circuito", list(mapa_vars.keys()))
        var_col = mapa_vars[var_label]
    with f3:
        data_min = df_clean.index.min().date()
        data_max = df_clean.index.max().date()
        intervalo = st.date_input(
            "Intervalo de Datas",
            value=(data_min, data_max),
            min_value=data_min,
            max_value=data_max
        )

    if isinstance(intervalo, tuple) and len(intervalo) == 2:
        d_ini, d_fim = intervalo
    else:
        d_ini, d_fim = data_min, data_max

    df_filtrado = df_clean.loc[str(d_ini):str(d_fim), [var_col]]

    if granularidade == "Horária":
        serie_saz = df_filtrado.groupby(df_filtrado.index.floor("h"))[var_col].median().reset_index()
        serie_saz.columns = ["Data_Hora", "Valor"]
    elif granularidade == "Diária":
        serie_saz = df_filtrado.resample("D")[var_col].mean().reset_index()
        serie_saz.columns = ["Data_Hora", "Valor"]
    else:
        serie_saz = df_filtrado.groupby(df_filtrado.index.to_period("M").to_timestamp())[var_col].median().reset_index()
        serie_saz.columns = ["Data_Hora", "Valor"]

    fig_saz = px.line(
        serie_saz, x="Data_Hora", y="Valor",
        title=f"Série Temporal ({granularidade}) — {var_label}",
        color_discrete_sequence=["#1f618d"]
    )
    if var_col == "Global_active_power":
        fig_saz.add_hline(
            y=meta["carga_basal_kw"], line_dash="dash", line_color="#c0392b",
            annotation_text=f"Carga Basal P10 ({meta['carga_basal_kw']:.3f} kW)"
        )
    fig_saz.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), yaxis_title=var_label, xaxis_title="Data")
    st.plotly_chart(fig_saz, use_container_width=True)

    st.markdown("---")
    st.markdown("**Projeção PCA dos Perfis Diários de 24 Horas (Mediana Horária Robusta)**")

    df_pca_base = df_clean[["Global_active_power"]].copy()
    df_pca_base["Data"] = df_pca_base.index.date
    df_pca_base["Hora"] = df_pca_base.index.hour
    matriz_24h = df_pca_base.pivot_table(index="Data", columns="Hora", values="Global_active_power", aggfunc="median").dropna()

    X_pca_scaled = StandardScaler().fit_transform(matriz_24h.values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_pca_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    datas_idx = pd.to_datetime(matriz_24h.index)
    dias_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    df_pca_plot = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Data": datas_idx.strftime("%Y-%m-%d"),
        "Tipo_Dia": np.where(datas_idx.dayofweek >= 5, "Fim de Semana", "Dia Útil"),
        "Dia_Semana": [dias_pt[d] for d in datas_idx.dayofweek],
        "Consumo_kWh": df_diario.reindex(datas_idx)["Energia_Total_kWh"].round(2).values
    })

    fig_pca = px.scatter(
        df_pca_plot,
        x="PC1",
        y="PC2",
        color="Tipo_Dia",
        color_discrete_map={"Dia Útil": "#2980b9", "Fim de Semana": "#e67e22"},
        hover_data={
            "Data": True,
            "Tipo_Dia": True,
            "Dia_Semana": True,
            "Consumo_kWh": True,
            "PC1": ":.2f",
            "PC2": ":.2f"
        },
        title=f"Mapa PCA de Perfis Diários (Variância Explicada: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%)"
    )
    fig_pca.update_traces(marker=dict(size=8, opacity=0.8))
    fig_pca.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pca, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: PREVISÃO (PARTE III)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Previsão do Consumo do Dia Seguinte (Validação Temporal sem Data Leakage)")

    df_metricas, df_preds, freq_selecao = executar_modelo_parte3(df_diario)

    m1, m2, m3 = st.columns(3)
    mae_mod = df_metricas["MAE Modelo (kWh)"].mean()
    mae_bas = df_metricas["MAE Baseline (kWh)"].mean()
    melhoria = ((mae_bas - mae_mod) / mae_bas) * 100

    m1.metric("MAE Médio — Modelo (LassoCV)", f"{mae_mod:.2f} kWh/dia", delta=f"{melhoria:+.1f}% vs. Baseline")
    m2.metric("MAE Médio — Baseline (Ontem)", f"{mae_bas:.2f} kWh/dia")
    m3.metric("R² Médio (Fora da Amostra)", f"{df_metricas['R2 Modelo'].mean():.3f}")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_preds["Data"], y=df_preds["Real_kWh"],
        mode="lines", name="Consumo Real (kWh)", line=dict(color="#2c3e50", width=1.8)
    ))
    fig_pred.add_trace(go.Scatter(
        x=df_preds["Data"], y=df_preds["Previsto_Lasso_kWh"],
        mode="lines", name="Previsto — Embedded LassoCV", line=dict(color="#e67e22", width=1.6, dash="dash")
    ))
    fig_pred.add_trace(go.Scatter(
        x=df_preds["Data"], y=df_preds["Baseline_Ontem_kWh"],
        mode="lines", name="Baseline Ingénua (Ontem)", line=dict(color="#95a5a6", width=1.1, dash="dot"),
        visible="legendonly"
    ))
    fig_pred.add_hline(
        y=meta["energia_basal_kwh_dia"], line_dash="dash", line_color="#c0392b",
        annotation_text="Carga Basal Constante"
    )
    fig_pred.update_layout(
        title="Previsão Out-of-Sample (Todos os 5 Folds Temporais)",
        height=380, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Energia (kWh/dia)"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    c_tab, c_vars = st.columns([1.2, 1.0])
    with c_tab:
        st.markdown("**Desempenho por Período Temporal (`TimeSeriesSplit`)**")
        st.dataframe(df_metricas, use_container_width=True, hide_index=True)

    with c_vars:
        st.markdown("**Variáveis Selecionadas na Parte III (% de Folds)**")
        vars_ativas = freq_selecao[freq_selecao > 0].reset_index()
        vars_ativas.columns = ["Variável", "Frequência (%)"]
        fig_vars = px.bar(
            vars_ativas, x="Frequência (%)", y="Variável", orientation="h",
            color_discrete_sequence=["#1f618d"]
        )
        fig_vars.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_vars, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MONITORIZAÇÃO EM TEMPO REAL (PARTE IV — CONTROLO INTERATIVO)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Sala de Controlo: Replay Online (Welford / EWMA) & Alarme Interativo")

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.2, 1.2])
    with ctrl1:
        dias_disponiveis = sorted(df_clean.index.normalize().unique())
        dia_padrao = pd.Timestamp("2008-11-23") if pd.Timestamp("2008-11-23") in dias_disponiveis else dias_disponiveis[len(dias_disponiveis)//2]
        dia_escolhido = st.date_input(
            "Dia para Inspeção ao Minuto",
            value=dia_padrao.date(),
            min_value=dias_disponiveis[0].date(),
            max_value=dias_disponiveis[-1].date()
        )
    with ctrl2:
        janela_slider = st.slider(
            "Tamanho da Janela Deslizante (minutos)",
            min_value=15, max_value=180, value=60, step=15
        )
    with ctrl3:
        limiar_slider = st.slider(
            "Limiar do Z-Score para Alarme (|Z| > θ)",
            min_value=2.0, max_value=5.5, value=3.5, step=0.25
        )

    res_stream = executar_replay_streaming(df_clean, str(dia_escolhido), janela_slider, limiar_slider)
    df_dia_st = res_stream["df_dia"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Média Global (Welford Online)", f"{res_stream['mean_welford']:.3f} kW")
    k2.metric("Variância Global (Welford)", f"{res_stream['var_welford']:.3f} kW²")
    k3.metric(
        "Alarmes na Janela de 18 Meses",
        f"{res_stream['total_alarmes_globais']} eventos",
        delta=f"Janela={janela_slider}m | θ={limiar_slider}",
        delta_color="off"
    )
    k4.metric(
        f"Minutos em Alarme ({dia_escolhido})",
        f"{int(df_dia_st['Alarme'].sum())} min"
    )

    fig_st = go.Figure()
    fig_st.add_trace(go.Scatter(
        x=df_dia_st["Timestamp"], y=df_dia_st["Bruto_kW"],
        mode="lines", name="Leitura Bruta ao Minuto (kW)",
        line=dict(color="#95a5a6", width=1.0), opacity=0.7
    ))
    fig_st.add_trace(go.Scatter(
        x=df_dia_st["Timestamp"], y=df_dia_st["Janela_kW"],
        mode="lines", name=f"Janela Deslizante ({janela_slider} min)",
        line=dict(color="#1f618d", width=2.2)
    ))
    fig_st.add_trace(go.Scatter(
        x=df_dia_st["Timestamp"], y=df_dia_st["EWMA_kW"],
        mode="lines", name="EWMA Online",
        line=dict(color="#d35400", width=2.0, dash="dash")
    ))

    df_pts_alarme = df_dia_st[df_dia_st["Alarme"]]
    if not df_pts_alarme.empty:
        fig_st.add_trace(go.Scatter(
            x=df_pts_alarme["Timestamp"], y=df_pts_alarme["Bruto_kW"],
            mode="markers", name="Alarme Disparado (|Z| > θ)",
            marker=dict(color="#c0392b", size=9, symbol="x")
        ))

    fig_st.update_layout(
        title=f"Replay em Tempo Real ({dia_escolhido}) — Efeito da Janela e do Limiar Z-Score",
        height=400, margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Potência Ativa (kW)", xaxis_title="Hora"
    )
    st.plotly_chart(fig_st, use_container_width=True)

    if not res_stream["alarmes_dia"].empty:
        with st.expander(f"Ver lista detalhada dos {len(res_stream['alarmes_dia'])} minutos com alarme neste dia"):
            st.dataframe(res_stream["alarmes_dia"], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum alarme disparado neste dia com os parâmetros atuais.")