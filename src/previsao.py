from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, SelectFromModel
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def preparar_dataset_diario_causal(df_clean, carga_basal_kw):
    print("\n--- [3.1] Construção de Variáveis Candidatas Causais (Sem Fuga Temporal) ---")
    energia_basal_kwh_dia = carga_basal_kw * 24.0

    # Agregação diária exigindo >= 1200 min válidos (20h) para considerar o dia completo
    df_diario = pd.DataFrame(index=df_clean.resample('D').mean().index)
    df_diario['Energia_Total_kWh'] = df_clean['Energia_Total_Wh'].resample('D').sum(min_count=1200) / 1000.0
    df_diario['Sub1_Cozinha_kWh'] = df_clean['Sub_metering_1'].resample('D').sum(min_count=1200) / 1000.0
    df_diario['Sub2_Lavandaria_kWh'] = df_clean['Sub_metering_2'].resample('D').sum(min_count=1200) / 1000.0
    df_diario['Sub3_Climatizacao_kWh'] = df_clean['Sub_metering_3'].resample('D').sum(min_count=1200) / 1000.0
    df_diario['Sub_Outros_kWh'] = df_clean['Sub_metering_outros'].resample('D').sum(min_count=1200) / 1000.0
    df_diario['Energia_Variavel_kWh'] = (df_diario['Energia_Total_kWh'] - energia_basal_kwh_dia).clip(lower=0)
    df_diario['Voltage_Media'] = df_clean['Voltage'].resample('D').mean()
    df_diario['Reactive_Media'] = df_clean['Global_reactive_power'].resample('D').mean()

    dados = pd.DataFrame(index=df_diario.index)
    # Alvo: consumo total do dia t (kWh/dia)
    dados['Target_Energia_kWh'] = df_diario['Energia_Total_kWh']

    # 1. Lags autoregressivos (estritamente <= t-1)
    dados['lag_1'] = df_diario['Energia_Total_kWh'].shift(1)
    dados['lag_2'] = df_diario['Energia_Total_kWh'].shift(2)
    dados['lag_3'] = df_diario['Energia_Total_kWh'].shift(3)
    dados['lag_7'] = df_diario['Energia_Total_kWh'].shift(7)

    # 2. Lags por sub-contador e grandezas elétricas de ontem (t-1)
    dados['sub1_cozinha_lag1'] = df_diario['Sub1_Cozinha_kWh'].shift(1)
    dados['sub2_lavandaria_lag1'] = df_diario['Sub2_Lavandaria_kWh'].shift(1)
    dados['sub3_climatizacao_lag1'] = df_diario['Sub3_Climatizacao_kWh'].shift(1)
    dados['sub_outros_lag1'] = df_diario['Sub_Outros_kWh'].shift(1)
    dados['energia_variavel_lag1'] = df_diario['Energia_Variavel_kWh'].shift(1)
    dados['voltage_lag1'] = df_diario['Voltage_Media'].shift(1)
    dados['reactive_lag1'] = df_diario['Reactive_Media'].shift(1)

    # 3. Estatísticas móveis causais (shift(1) obrigatório antes do rolling!)
    serie_passada = df_diario['Energia_Total_kWh'].shift(1)
    dados['roll_mean_3'] = serie_passada.rolling(window=3, min_periods=2).mean()
    dados['roll_mean_7'] = serie_passada.rolling(window=7, min_periods=5).mean()
    dados['roll_median_7'] = serie_passada.rolling(window=7, min_periods=5).median()
    dados['roll_std_7'] = serie_passada.rolling(window=7, min_periods=5).std()
    dados['roll_mean_14'] = serie_passada.rolling(window=14, min_periods=10).mean()

    # 4. Variáveis determinísticas de calendário para o dia t (conhecidas antecipadamente)
    anos_dataset = list(range(df_diario.index.year.min(), df_diario.index.year.max() + 1))
    feriados_fr = holidays.France(years=anos_dataset)

    dados['dia_semana'] = dados.index.dayofweek
    dados['is_fim_de_semana'] = (dados.index.dayofweek >= 5).astype(int)
    dados['mes'] = dados.index.month
    dados['is_feriado'] = dados.index.normalize().isin(feriados_fr).astype(int)
    dados['mes_sin'] = np.sin(2 * np.pi * dados['mes'] / 12.0)
    dados['mes_cos'] = np.cos(2 * np.pi * dados['mes'] / 12.0)

    dados_validos = dados.dropna().copy()

    assert dados_validos.index.is_monotonic_increasing, "Índice temporal não está ordenado."
    assert not dados_validos.isna().any().any(), "Existem NaNs residuais na matriz."

    X = dados_validos.drop(columns=['Target_Energia_kWh'])
    y = dados_validos['Target_Energia_kWh']
    assert X.shape[1] >= 12, f"Número insuficiente de variáveis ({X.shape[1]} < 12)."

    print(f"Matriz causal válida: {X.shape[0]} dias x {X.shape[1]} variáveis candidatas.")
    return X, y

def executar_selecao_comparada(X, y, carga_basal_kw):
    print("\n--- [3.2] Seleção Comparada dentro de Pipeline + TimeSeriesSplit ---")
    tscv = TimeSeriesSplit(n_splits=5)
    k_features = 6
    mi_scorer = partial(mutual_info_regression, random_state=42)

    pipelines = {
        'Todas as Variáveis (Baseline RL)': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Filter (SelectKBest - Info Mútua)': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mi_scorer, k=k_features)),
            ('regressor', LinearRegression())
        ]),
        'Wrapper (RFE - Regressão Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', RFE(estimator=LinearRegression(), n_features_to_select=k_features, step=1)),
            ('regressor', LinearRegression())
        ]),
        'Embedded (SelectFromModel - LassoCV)': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(
                LassoCV(alphas=np.logspace(-2, 0.3, 30), cv=TimeSeriesSplit(n_splits=3), random_state=42, max_iter=10000)
            )),
            ('regressor', LinearRegression())
        ])
    }

    resultados_folds = []
    historico_selecao = {
        nome: pd.DataFrame(0, index=X.columns, columns=[f'Fold_{i+1}' for i in range(5)])
        for nome in pipelines if 'Todas' not in nome
    }

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Referência ingénua de persistência (ontem)
        y_pred_naive = X_val['lag_1']
        resultados_folds.append({
            'Metodo': 'Persistência Ingénua (Ontem)',
            'Fold': fold_idx + 1,
            'N_Vars': 1,
            'MAE_kWh': mean_absolute_error(y_val, y_pred_naive),
            'RMSE_kWh': root_mean_squared_error(y_val, y_pred_naive),
            'R2': r2_score(y_val, y_pred_naive)
        })

        for nome, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)

            if 'selector' in pipe.named_steps:
                suporte = pipe.named_steps['selector'].get_support()
                historico_selecao[nome].iloc[:, fold_idx] = suporte.astype(int)
                n_vars = int(suporte.sum())
            else:
                n_vars = X.shape[1]

            resultados_folds.append({
                'Metodo': nome,
                'Fold': fold_idx + 1,
                'N_Vars': n_vars,
                'MAE_kWh': mean_absolute_error(y_val, y_pred),
                'RMSE_kWh': root_mean_squared_error(y_val, y_pred),
                'R2': r2_score(y_val, y_pred)
            })

    df_res = pd.DataFrame(resultados_folds)
    tabela_comp = df_res.groupby('Metodo', sort=False).agg(
        Vars_Medias=('N_Vars', 'mean'),
        MAE_Media=('MAE_kWh', 'mean'),
        MAE_Std=('MAE_kWh', 'std'),
        RMSE_Media=('RMSE_kWh', 'mean'),
        R2_Medio=('R2', 'mean')
    ).round(3)

    print("\nDesempenho em Validação Temporal (TimeSeriesSplit - 5 Folds):")
    print(tabela_comp)

    # Estabilidade entre folds (Jaccard)
    df_freq_geral = pd.DataFrame(index=X.columns)
    for nome, df_bin in historico_selecao.items():
        df_freq_geral[nome] = (df_bin.sum(axis=1) / df_bin.shape[1]) * 100
        jaccards = []
        for i in range(5):
            for j in range(i + 1, 5):
                s1 = set(df_bin.index[df_bin.iloc[:, i] == 1])
                s2 = set(df_bin.index[df_bin.iloc[:, j] == 1])
                jaccards.append(len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 1.0)
        print(f"Estabilidade Jaccard Médio [{nome}]: {np.mean(jaccards):.3f}")

    # Gráfico comparativo: Estabilidade + MAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), gridspec_kw={'width_ratios': [1.25, 1.0]})
    df_plot_freq = df_freq_geral.loc[df_freq_geral.sum(axis=1) > 0].copy()
    df_plot_freq.columns = ['Filter\n(Info Mútua)', 'Wrapper\n(RFE)', 'Embedded\n(LassoCV)']
    df_plot_freq = df_plot_freq.sort_values(by=list(df_plot_freq.columns), ascending=False)

    sns.heatmap(
        df_plot_freq, ax=ax1, cmap='YlGnBu', annot=True, fmt='.0f',
        vmin=0, vmax=100, linewidths=0.6, linecolor='white',
        cbar_kws={'label': 'Frequência de Seleção nos 5 Folds (%)'}
    )
    ax1.set_title('Estabilidade entre Folds: Lags de 1 e 7 dias e médias móveis dominam', fontsize=11.5, fontweight='bold', loc='left')
    ax1.set_ylabel('Variáveis Candidatas')

    metodos_plot = list(tabela_comp.index)
    labels_curtos = ['Persistência\n(1 var)', 'RL Todas\n(22 vars)', 'RL Filter\n(6 vars)', 'RL Wrapper\n(6 vars)', 'RL Embedded\n(LassoCV)']
    maes = [tabela_comp.loc[m, 'MAE_Media'] for m in metodos_plot]
    stds = [tabela_comp.loc[m, 'MAE_Std'] for m in metodos_plot]
    cores = ['#95a5a6', '#34495e', '#2980b9', '#16a085', '#d35400']

    bars = ax2.bar(labels_curtos, maes, yerr=stds, capsize=4, color=cores, width=0.58, edgecolor='black', linewidth=0.6)
    ax2.set_ylim(0, max(maes) * 1.28)
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.set_title('Regressão Linear com variáveis selecionadas iguala/supera o modelo completo', fontsize=11.5, fontweight='bold', loc='left')
    ax2.set_ylabel('MAE Médio (kWh/dia)')
    plt.tight_layout()
    plt.show()

    return tabela_comp, df_freq_geral