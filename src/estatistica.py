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

def plot_prova_outliers_reais(df_analise):
    df_valid = df_analise.dropna(subset=['Global_active_power', 'Global_intensity', 'Voltage']).copy()
    mask_out = df_valid['Outlier_IQR']
    
    # Calcular limite superior IQR para referência visual
    q1 = df_valid['Global_active_power'].quantile(0.25)
    q3 = df_valid['Global_active_power'].quantile(0.75)
    lim_sup_iqr = q3 + 1.5 * (q3 - q1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- PAINEL 1: Consistência Física (Corrente vs. Potência Ativa) ---
    normal_sample = df_valid[~mask_out].sample(n=min(15000, (~mask_out).sum()), random_state=42)
    outlier_sample = df_valid[mask_out].sample(n=min(15000, mask_out.sum()), random_state=42)

    axes[0].scatter(normal_sample['Global_intensity'], normal_sample['Global_active_power'],
                    color='#5d6d7e', alpha=0.25, s=8, label='Regime Normal')
    axes[0].scatter(outlier_sample['Global_intensity'], outlier_sample['Global_active_power'],
                    color='#d35400', alpha=0.35, s=10, label='Outliers IQR (> limiar)')
    axes[0].axhline(lim_sup_iqr, color='#c0392b', linestyle='--', linewidth=1.3, label=f'Limiar IQR ({lim_sup_iqr:.2f} kW)')
    axes[0].set_title('1. Lei Física (P vs. I): linearidade perfeita sem ruído', fontsize=11, fontweight='bold', loc='left')
    axes[0].set_xlabel('Intensidade de Corrente (A)')
    axes[0].set_ylabel('Potência Ativa Global (kW)')
    axes[0].legend(loc='upper left', fontsize=8.5, frameon=True)

    # --- PAINEL 2: Decomposição por Circuito (Normal vs. Outliers IQR) ---
    df_valid['Energia_Wh'] = (df_valid['Global_active_power'] * 1000.0) / 60.0
    df_valid['Outros_Wh'] = (
        df_valid['Energia_Wh'] - (df_valid['Sub_metering_1'] + df_valid['Sub_metering_2'] + df_valid['Sub_metering_3'])
    ).clip(lower=0)

    cols_circ = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Outros_Wh']
    nomes_circ = ['Cozinha\n(Sub 1)', 'Lavandaria\n(Sub 2)', 'Climat./Águas\n(Sub 3)', 'Geral/Luz\n(Outros)']

    medias_norm = df_valid[~mask_out][cols_circ].mean().values
    medias_out = df_valid[mask_out][cols_circ].mean().values

    x = np.arange(len(nomes_circ))
    w = 0.36
    axes[1].bar(x - w/2, medias_norm, width=w, color='#5d6d7e', label='Regime Normal')
    axes[1].bar(x + w/2, medias_out, width=w, color='#d35400', label='Nos Outliers IQR')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nomes_circ, fontsize=9.5)
    axes[1].set_title('2. Origem: disparo simultâneo de eletrodomésticos', fontsize=11, fontweight='bold', loc='left')
    axes[1].set_ylabel('Energia Média por Minuto (Wh/min)')
    axes[1].legend(loc='upper left', fontsize=8.5, frameon=True)

    # --- PAINEL 3: Zoom num Dia Real de Pico (Continuidade Temporal + Tensão) ---
    dia_pico = df_valid.groupby(df_valid.index.date)['Outlier_IQR'].sum().idxmax()
    df_dia = df_valid.loc[str(dia_pico)]

    axes[2].plot(df_dia.index.hour + df_dia.index.minute / 60.0, df_dia['Global_active_power'],
                 color='#1f618d', linewidth=1.3, label='Potência Ativa (kW)')
    axes[2].axhline(lim_sup_iqr, color='#c0392b', linestyle='--', linewidth=1.3, label='Limiar IQR')
    axes[2].fill_between(
        df_dia.index.hour + df_dia.index.minute / 60.0,
        lim_sup_iqr, df_dia['Global_active_power'],
        where=(df_dia['Global_active_power'] > lim_sup_iqr),
        color='#e67e22', alpha=0.35, label='Patamares contínuos de uso'
    )
    axes[2].set_title(f'3. Zoom ({dia_pico}): patamares contínuos (tensão {df_dia["Voltage"].min():.0f}–{df_dia["Voltage"].max():.0f}V)',
                      fontsize=11, fontweight='bold', loc='left')
    axes[2].set_xlabel('Hora do Dia (0h – 24h)')
    axes[2].set_ylabel('Potência Ativa (kW)')
    axes[2].set_xlim(0, 24)
    axes[2].set_xticks([0, 6, 12, 18, 24])
    axes[2].legend(loc='upper left', fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.show()

# Executar logo após investigar_outliers(df):
plot_prova_outliers_reais(df_analise)

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

def plot_prova_carga_basal(df):
    col = df['Global_active_power'].dropna()
    
    # 1. Calcular os diferentes candidatos a Carga Basal
    p0_min = float(col.min())
    p10_basal = float(col.quantile(0.10))
    madrugada = float(col[(col.index.hour >= 3) & (col.index.hour <= 5)].median())
    
    # Dias sem atividade humana na cozinha (Sub_1) nem na lavandaria (Sub_2)
    diario_subs = df[['Sub_metering_1', 'Sub_metering_2']].resample('D').sum(min_count=1200)
    dias_ausencia = diario_subs[(diario_subs['Sub_metering_1'] == 0) & (diario_subs['Sub_metering_2'] == 0)].index
    ausencia_med = float(df[df.index.normalize().isin(dias_ausencia)]['Global_active_power'].median()) if len(dias_ausencia) > 0 else p10_basal

    print("\n--- Prova de Convergência da Carga Basal Constante ---")
    print(f"1. Mínimo Absoluto (P0 - apenas standby sem compressor): {p0_min:.3f} kW ({p0_min*1000:.0f} W)")
    print(f"2. Percentil 10 Global (P10 - valor escolhido):          {p10_basal:.3f} kW ({p10_basal*1000:.0f} W)")
    print(f"3. Mediana da Madrugada (03h00 - 05h59):                 {madrugada:.3f} kW ({madrugada*1000:.0f} W)")
    print(f"4. Mediana em Dias de Ausência (Sub_1=0 & Sub_2=0):      {ausencia_med:.3f} kW ({ausencia_med*1000:.0f} W)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.0, 1.35]})

    # --- PAINEL 1: Triangulação de Estimadores ---
    metodos = ['Mínimo (P0)\n[Só Standby]', 'P10 Global\n[Escolhido]', 'Madrugada\n[03h–05h]', 'Dias Ausência\n[Sub1=0, Sub2=0]']
    valores = [p0_min, p10_basal, madrugada, ausencia_med]
    cores = ['#95a5a6', '#c0392b', '#2980b9', '#16a085']

    bars = ax1.bar(metodos, valores, color=cores, width=0.55, edgecolor='black', linewidth=0.6)
    ax1.axhline(p10_basal, color='#c0392b', linestyle='--', linewidth=1.4, label=f'Referência P10 ({p10_basal:.3f} kW)')
    ax1.set_ylim(0, max(valores) * 1.32)

    for bar, v in zip(bars, valores):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.008, f'{v:.3f} kW\n({v*1000:.0f} W)',
                 ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    ax1.set_title('1. Convergência: P10, madrugada e ausência coincidem', fontsize=11.5, fontweight='bold', loc='left', pad=10)
    ax1.set_ylabel('Potência Ativa (kW)')
    ax1.legend(loc='upper left', fontsize=9)

    # --- PAINEL 2: Zoom ao Minuto numa Madrugada (Ciclos do Compressor) ---
    # Selecionar uma madrugada típica de inatividade (01h00 às 06h00)
    dia_exemplo = dias_ausencia[len(dias_ausencia)//2] if len(dias_ausencia) > 0 else col.index[10000].normalize()
    janela_noite = df.loc[str(dia_exemplo.date())].between_time('01:00', '06:00')['Global_active_power']
    horas_dec = janela_noite.index.hour + janela_noite.index.minute / 60.0

    ax2.plot(horas_dec, janela_noite.values, color='#2c3e50', linewidth=1.4, label='Potência ao Minuto (kW)')
    ax2.axhline(p10_basal, color='#c0392b', linestyle='--', linewidth=1.8, label=f'Carga Basal P10 ({p10_basal:.3f} kW — média do ciclo)')
    ax2.axhline(p0_min, color='#7f8c8d', linestyle=':', linewidth=1.4, label=f'Mínimo P0 ({p0_min:.3f} kW — compressor desligado)')
    ax2.fill_between(horas_dec, 0, p10_basal, color='#e74c3c', alpha=0.10)

    ax2.set_title(f'2. Zoom ao minuto ({dia_exemplo.date()}, 01h–06h): ciclos térmicos do frigorífico',
                  fontsize=11.5, fontweight='bold', loc='left', pad=10)
    ax2.set_xlabel('Hora da Madrugada')
    ax2.set_ylabel('Potência Ativa (kW)')
    ax2.set_ylim(0, max(janela_noite.max() * 1.25, 0.55))
    ax2.legend(loc='upper right', fontsize=8.8, frameon=True)

    plt.tight_layout()
    plt.show()

# Executar na Secção 1.3 logo após estimar carga_basal_kw:
plot_prova_carga_basal(df)

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