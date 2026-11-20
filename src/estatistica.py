import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def investigar_outliers(df, coluna='Global_active_power'):
    print(f"\n--- Deteção e Investigação de Outliers ({coluna}) ---")
    
    col = df[coluna].dropna()
    
    # 1. Regra 1.5 x IQR
    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf_iqr, lim_sup_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask_iqr = (df[coluna] < lim_inf_iqr) | (df[coluna] > lim_sup_iqr)
    
    # 2. Z-Score Robusto (Mediana e MAD)
    mediana = col.median()
    mad = stats.median_abs_deviation(col)
    z_robusto = 0.6745 * (df[coluna] - mediana) / mad
    mask_z = np.abs(z_robusto) > 3
    
    print(f"1. Regra 1.5 x IQR: {mask_iqr.sum()} minutos extremos ({mask_iqr.sum()/len(col)*100:.2f}%) | Limite superior: {lim_sup_iqr:.2f} kW")
    print(f"2. Z-Score Robusto (|Z| > 3): {mask_z.sum()} minutos extremos ({mask_z.sum()/len(col)*100:.2f}%)")
    
    # 3. Investigação da natureza dos extremos (Erros vs. Ondas de Frio / Uso Simultâneo)
    extremos = df[mask_iqr]
    erros_tensao = extremos[(extremos['Voltage'] < 195) | (extremos['Voltage'] > 265)]
    print(f"\n[Verificação de Erros] Registos extremos com tensão anómala (<195V ou >265V): {len(erros_tensao)}")
    
    print("\n[Verificação Sazonal] Distribuição dos outliers IQR por mês (1=Jan ... 12=Dez):")
    print(extremos.index.month.value_counts().sort_index().to_dict())
    
    cols_sub = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    comp = pd.DataFrame({
        'Mediana_Regime_Normal': df[~mask_iqr][cols_sub].median(),
        'Mediana_Nos_Outliers_IQR': extremos[cols_sub].median()
    })
    print("\n[Verificação de Circuitos] Mediana dos sub-contadores (Wh/min):")
    print(comp.round(2))
    
    # Mantém todas as linhas intactas e apenas adiciona flags informativas
    df_resultado = df.copy()
    df_resultado['Outlier_IQR'] = mask_iqr
    df_resultado['Outlier_ZRobusto'] = mask_z
    
    return df_resultado

def analisar_distribuicao_e_assimetria(df_vars):
    print("\n--- Análise de Distribuição e Assimetria (Média vs. Mediana) ---")
    
    resumo_forma = pd.DataFrame({
        'Media': df_vars.mean(),
        'Mediana': df_vars.median(),
        'Desvio_Padrao': df_vars.std(),
        'IQR': df_vars.quantile(0.75) - df_vars.quantile(0.25),
        'Skewness': df_vars.skew()
    })
    print(resumo_forma.round(3))

    col = df_vars['Global_active_power'].dropna()
    media, mediana, skew_val = col.mean(), col.median(), col.skew()

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True,
        gridspec_kw={'height_ratios': [0.22, 0.78]}
    )
    sns.boxplot(x=col, ax=ax_box, color='#85c1e9', fliersize=1.5)
    ax_box.set(xlabel='')
    ax_box.set_title(
        f'Assimetria positiva acentuada (Skew = {skew_val:.2f}): a média ({media:.2f} kW) sobrestima o consumo típico ({mediana:.2f} kW)',
        fontsize=12, fontweight='bold', loc='left', pad=10
    )

    sns.histplot(col, bins=80, kde=True, ax=ax_hist, color='#3498db', alpha=0.45, edgecolor='none')
    ax_hist.axvline(mediana, color='#1e8449', linestyle='-', linewidth=2.4, label=f'Mediana = {mediana:.3f} kW (Localização Robusta)')
    ax_hist.axvline(media, color='#d35400', linestyle='--', linewidth=2.2, label=f'Média = {media:.3f} kW (Distorcida pelos picos)')
    ax_hist.set_xlabel('Potência Ativa Global (kW)')
    ax_hist.set_ylabel('Frequência (n.º de minutos)')
    ax_hist.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

    return resumo_forma

def decompor_carga_basal(df):
    print("\n--- Decomposição da Carga Basal Constante (Frigorífico / Arcas / Standby) ---")
    
    carga_basal_kw = float(df['Global_active_power'].quantile(0.10))
    energia_basal_kwh_dia = carga_basal_kw * 24.0
    
    df_decomposed = df.copy()
    df_decomposed['Carga_Basal_kW'] = carga_basal_kw
    df_decomposed['Consumo_Variavel_kW'] = (df_decomposed['Global_active_power'] - carga_basal_kw).clip(lower=0)
    
    print(f"Carga Basal Constante estimada (P10): {carga_basal_kw:.3f} kW ({carga_basal_kw*1000:.0f} W contínuos)")
    print(f"Equivalente energético diário constante: {energia_basal_kwh_dia:.2f} kWh/dia")
    
    return df_decomposed, carga_basal_kw

def gerar_tabela_dinamica(df, coluna='Global_active_power'):
    print(f"\n--- Tabela Dinâmica: Mediana e Contagem (n) por Hora x Dia da Semana ({coluna}) ---")
    
    df_temp = df.copy()
    df_temp['Hora'] = df_temp.index.hour
    df_temp['Dia_Semana'] = df_temp.index.day_name()
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot_mediana = pd.pivot_table(
        df_temp, values=coluna, index='Hora', columns='Dia_Semana', aggfunc='median'
    ).reindex(columns=dias_ordem)
    
    pivot_n = pd.pivot_table(
        df_temp, values=coluna, index='Hora', columns='Dia_Semana', aggfunc='count'
    ).reindex(columns=dias_ordem)
    
    print("Mediana Horária (kW):")
    print(pivot_mediana.round(3))
    print("\nContagem de observações válidas (n):")
    print(pivot_n)
    
    return pivot_mediana, pivot_n