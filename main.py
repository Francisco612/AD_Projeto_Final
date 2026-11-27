from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.streaming import (
    processar_fluxo_tempo_real,
    plot_comparacao_dia_4_1,
    plot_drift_adwin_4_2,
    inspecionar_5_alarmes,
)

SOMA_DOS_NUMEROS_DE_ALUNO = 101833


def encontrar_dataset():
    """Procura automaticamente o ficheiro ZIP ou TXT do dataset em data/raw e subpastas."""
    base = Path(__file__).resolve().parent
    for raiz in [Path.cwd(), base]:
        if raiz.exists():
            for padrao in ("*.zip", "*.txt"):
                for fich in raiz.rglob(padrao):
                    if not any(p in fich.parts for p in (".venv", "venv", "env", ".git", "__pycache__")):
                        if fich.stat().st_size > 1_000_000:
                            return str(fich)
    raise FileNotFoundError("Dataset não encontrado em data/raw/.")


def carregar_e_preparar_janela():
    caminho = encontrar_dataset()
    print(f"[0] Dataset localizado: {caminho}")

    df_raw = pd.read_csv(caminho, sep=";", na_values=["?", "nan", "NaN", ""], low_memory=False)
    df_raw.index = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"], format="%d/%m/%Y %H:%M:%S")
    df_raw = df_raw.drop(columns=["Date", "Time"]).sort_index()

    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw = df_raw.reindex(pd.date_range(df_raw.index.min(), df_raw.index.max(), freq="1min"))

    meses_unicos = df_raw.index.to_period("M").unique()
    idx_inicio = SOMA_DOS_NUMEROS_DE_ALUNO % max(1, len(meses_unicos) - 18)
    meses_grupo = meses_unicos[idx_inicio : idx_inicio + 18]

    df_grupo_bruto = df_raw.loc[df_raw.index.to_period("M").isin(meses_grupo)].copy()
    pct_missing = df_grupo_bruto["Global_active_power"].isna().mean() * 100.0

    df_clean = df_grupo_bruto.interpolate(method="time", limit=60)
    df_clean["Energia_Total_Wh"] = (df_clean["Global_active_power"] * 1000.0) / 60.0
    soma_subs = df_clean["Sub_metering_1"] + df_clean["Sub_metering_2"] + df_clean["Sub_metering_3"]
    df_clean["Sub_outros_Wh"] = (df_clean["Energia_Total_Wh"] - soma_subs).clip(lower=0)

    print(f"    Janela de 18 meses: {df_clean.index.min().date()} a {df_clean.index.max().date()} | Missing bruto: {pct_missing:.2f}%")
    return df_clean


def executar_parte1_e_parte2(df_clean):
    print("\n[Parte I & II] Estatística Descritiva, Carga Basal (P10) e PCA Diário...")
    carga_basal_kw = float(df_clean["Global_active_power"].quantile(0.10))
    energia_basal_kwh_dia = carga_basal_kw * 24.0

    q1 = float(df_clean["Global_active_power"].quantile(0.25))
    q3 = float(df_clean["Global_active_power"].quantile(0.75))
    limiar_iqr = q3 + 1.5 * (q3 - q1)

    print(f"    Carga Basal (P10): {carga_basal_kw:.3f} kW ({energia_basal_kwh_dia:.2f} kWh/dia)")
    print(f"    Limiar Superior IQR (minuto): {limiar_iqr:.3f} kW")

    # PCA 24h (Mediana Horária)
    df_pca = df_clean[["Global_active_power"]].copy()
    df_pca["Data"] = df_pca.index.date
    df_pca["Hora"] = df_pca.index.hour
    matriz_24h = df_pca.pivot_table(index="Data", columns="Hora", values="Global_active_power", aggfunc="median").dropna()

    X_scaled = StandardScaler().fit_transform(matriz_24h.values)
    pca = PCA(n_components=2, random_state=42).fit(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100.0
    print(f"    PCA Perfis Diários (24h): PC1 = {var_exp[0]:.1f}%, PC2 = {var_exp[1]:.1f}% (Total = {var_exp.sum():.1f}%)")

    return carga_basal_kw, energia_basal_kwh_dia


def executar_parte3(df_clean, energia_basal_kwh_dia):
    print("\n[Parte III] Previsão Diária Causal (TimeSeriesSplit + Embedded LassoCV)...")
    df_diario = pd.DataFrame(index=df_clean.resample("D").mean().index)
    df_diario["Energia_Total_kWh"] = df_clean["Energia_Total_Wh"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub1_Cozinha_kWh"] = df_clean["Sub_metering_1"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub2_Lavandaria_kWh"] = df_clean["Sub_metering_2"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub3_Climatizacao_kWh"] = df_clean["Sub_metering_3"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Sub_Outros_kWh"] = df_clean["Sub_outros_Wh"].resample("D").sum(min_count=1200) / 1000.0
    df_diario["Energia_Variavel_kWh"] = (df_diario["Energia_Total_kWh"] - energia_basal_kwh_dia).clip(lower=0)
    df_diario["Voltage_Media"] = df_clean["Voltage"].resample("D").mean()
    df_diario["Reactive_Media"] = df_clean["Global_reactive_power"].resample("D").mean()

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

    dados["dia_semana"] = dados.index.dayofweek
    dados["is_fim_de_semana"] = (dados.index.dayofweek >= 5).astype(int)
    dados["mes"] = dados.index.month
    dados["mes_sin"] = np.sin(2 * np.pi * dados["mes"] / 12.0)
    dados["mes_cos"] = np.cos(2 * np.pi * dados["mes"] / 12.0)

    dados_validos = dados.dropna().copy()
    X = dados_validos.drop(columns=["Target"])
    y = dados_validos["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            LassoCV(alphas=np.logspace(-2, 0.3, 30), cv=TimeSeriesSplit(n_splits=3), random_state=42, max_iter=10000)
        )),
        ("regressor", LinearRegression()),
    ])

    maes_mod, maes_base, rmses_mod, r2s_mod = [], [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)

        maes_mod.append(mean_absolute_error(y_va, y_pred))
        maes_base.append(mean_absolute_error(y_va, X_va["lag_1"]))
        rmses_mod.append(root_mean_squared_error(y_va, y_pred))
        r2s_mod.append(r2_score(y_va, y_pred))

    print(f"    MAE Médio Baseline (Ontem): {np.mean(maes_base):.2f} kWh/dia")
    print(f"    MAE Médio Embedded LassoCV: {np.mean(maes_mod):.2f} kWh/dia | RMSE: {np.mean(rmses_mod):.2f} kWh/dia | R²: {np.mean(r2s_mod):.3f}")


def executar_parte4(df_clean, mostrar_graficos=True):
    print("\n[Parte IV] Streaming em Tempo Real (Welford, Janela 60m, EWMA, ADWIN e Alarmes Z-Score)...")
    df_stream, df_15m, df_drifts, df_alarmes, resumo_welford = processar_fluxo_tempo_real(
        df_clean,
        janela_min=60,
        limiar_z=3.5,
        delta_adwin=0.002,
    )

    print(f"    Leituras processadas elemento a elemento: {resumo_welford['n_leituras']:,} ({resumo_welford['tempo_s']:.2f}s)")
    print(f"    Welford vs. Batch -> Diff Média: {resumo_welford['diff_mean']:.2e} | Diff Variância: {resumo_welford['diff_var']:.2e}")
    print(f"    Drifts detetados (ADWIN 15 min): {len(df_drifts)} | Alarmes Z-Score (|Z| > 3.5): {len(df_alarmes)}")

    if mostrar_graficos:
        plot_comparacao_dia_4_1(df_stream, dia_str=None, alpha_ewma=resumo_welford["alpha_ewma"])
        plot_drift_adwin_4_2(df_15m, df_drifts)

    df_5_alarmes = inspecionar_5_alarmes(df_clean, df_alarmes, df_stream)
    if not df_5_alarmes.empty:
        print("\n    Amostra de 5 Alarmes para Avaliação Qualitativa:")
        print(df_5_alarmes[["Alarme_ID", "Timestamp", "Potencia_kW", "Media_60m_kW", "Z_Score", "Avaliacao_Tecnica"]].to_string(index=False))


def main():
    df_clean = carregar_e_preparar_janela()
    _, energia_basal_kwh_dia = executar_parte1_e_parte2(df_clean)
    executar_parte3(df_clean, energia_basal_kwh_dia)
    executar_parte4(df_clean, mostrar_graficos=True)


if __name__ == "__main__":
    main()