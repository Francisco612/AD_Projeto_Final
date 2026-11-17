import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preparar_dados_temporais(df):
    # Interpolação apenas para falhas curtas (<= 60 min), mantendo dias completos em falta como NaN
    df_clean = df.interpolate(method='time', limit=60).copy()
    
    # Energia ativa por minuto em Wh: (kW * 1000) / 60
    df_clean['Energia_Total_Wh'] = (df_clean['Global_active_power'] * 1000.0) / 60.0
    df_clean['Sub_metering_outros'] = (
        df_clean['Energia_Total_Wh'] - (
            df_clean['Sub_metering_1'] + df_clean['Sub_metering_2'] + df_clean['Sub_metering_3']
        )
    ).clip(lower=0)

    def obter_estacao(mes):
        if mes in [12, 1, 2]:
            return 'Inverno'
        elif mes in [3, 4, 5]:
            return 'Primavera'
        elif mes in [6, 7, 8]:
            return 'Verão'
        return 'Outono'

    df_clean['Hora'] = df_clean.index.hour
    df_clean['Dia_Semana_Num'] = df_clean.index.dayofweek
    df_clean['Tipo_Dia'] = np.where(df_clean['Dia_Semana_Num'] >= 5, 'Fim de Semana', 'Dia Útil')
    df_clean['Estacao'] = df_clean.index.month.map(obter_estacao)
    return df_clean

def gerar_figuras_sazonalidade_e_diagnostico(df_clean):
    print("\n--- [2.1] A gerar Figuras de Sazonalidade e Diagnóstico (Drill-Down) ---")
    sns.set_theme(style="whitegrid")

    # FIGURA 1: Sazonalidade Diária
    perfil_h = df_clean.groupby('Hora')['Global_active_power'].agg(
        mediana='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        media='mean'
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(perfil_h.index, perfil_h['q25'], perfil_h['q75'], color='#3498db', alpha=0.22, label='Intervalo Interquartil (P25–P75)')
    ax.plot(perfil_h.index, perfil_h['mediana'], color='#1f618d', linewidth=2.5, marker='o', label='Mediana Horária')
    ax.plot(perfil_h.index, perfil_h['media'], color='#e67e22', linewidth=1.8, linestyle='--', label='Média Horária')

    pico_h, pico_v = perfil_h['mediana'].idxmax(), perfil_h['mediana'].max()
    ax.annotate(f'Pico Noturno ({pico_h}h: {pico_v:.2f} kW)',
                xy=(pico_h, pico_v), xytext=(pico_h - 5.5, pico_v + 0.45),
                arrowprops=dict(arrowstyle='->', color='#1f618d', lw=1.5),
                fontsize=10, fontweight='bold', color='#1f618d')
    ax.set_title('O consumo doméstico apresenta dois picos diários: manhã (07h–09h) e noite (19h–22h)',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Hora do Dia (0h – 23h)')
    ax.set_ylabel('Potência Ativa Global (kW)')
    ax.set_xticks(range(0, 24))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # FIGURA 2: Sazonalidade Semanal (Barras com origem rigorosamente a zero)
    energia_diaria = df_clean['Energia_Total_Wh'].resample('D').sum(min_count=1200) / 1000.0
    df_dias = pd.DataFrame({'Energia_kWh': energia_diaria, 'Dia_Num': energia_diaria.index.dayofweek})
    nomes_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    resumo_sem = df_dias.groupby('Dia_Num')['Energia_kWh'].mean()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(nomes_dias, resumo_sem.values, color=['#5d6d7e']*5 + ['#d35400']*2, width=0.62)
    ax.set_ylim(0, resumo_sem.max() * 1.18)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f'{h:.1f} kWh', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    media_uteis, media_fds = resumo_sem.iloc[:5].mean(), resumo_sem.iloc[5:].mean()
    aumento_pct = ((media_fds - media_uteis) / media_uteis) * 100
    ax.axhline(media_uteis, color='#2c3e50', linestyle='--', linewidth=1.3, label=f'Média Dias Úteis ({media_uteis:.1f} kWh/dia)')
    ax.set_title(f'O consumo diário aumenta {aumento_pct:.0f}% ao fim de semana, atingindo o máximo ao domingo',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Dia da Semana')
    ax.set_ylabel('Energia Média Diária (kWh/dia)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # FIGURA 3: Sazonalidade Anual Decomposta por Sub-contador
    cols_energia = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_outros']
    diario_subs = df_clean[cols_energia].resample('D').sum(min_count=1200) / 1000.0
    mensal_subs = diario_subs.resample('MS').mean()
    labels_meses = [d.strftime('%Y-%m') for d in mensal_subs.index]
    nomes_leg = ['Sub 1: Cozinha', 'Sub 2: Lavandaria', 'Sub 3: Climatização/Águas Quentes', 'Outros: Iluminação/Geral']
    cores_subs = ['#e74c3c', '#f39c12', '#2980b9', '#95a5a6']

    fig, ax = plt.subplots(figsize=(12.5, 5))
    bottom = np.zeros(len(mensal_subs))
    for col, nome, cor in zip(cols_energia, nomes_leg, cores_subs):
        vals = mensal_subs[col].fillna(0).values
        ax.bar(labels_meses, vals, bottom=bottom, label=nome, color=cor, width=0.68)
        bottom += vals
    ax.set_ylim(0, bottom.max() * 1.22)
    plt.xticks(rotation=45, ha='right')
    ax.set_title('O consumo elétrico duplica entre o verão e o inverno devido à climatização e águas quentes',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Mês de Referência')
    ax.set_ylabel('Energia Média Diária (kWh/dia)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # FIGURA 4: Small Multiples por Estação do Ano
    estacoes_ordem = ['Inverno', 'Primavera', 'Verão', 'Outono']
    perfil_est = df_clean.groupby(['Estacao', 'Tipo_Dia', 'Hora'])['Global_active_power'].mean().reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True, sharex=True)
    cores_tipo = {'Dia Útil': '#2c3e50', 'Fim de Semana': '#e67e22'}
    for ax, est in zip(axes, estacoes_ordem):
        sub = perfil_est[perfil_est['Estacao'] == est]
        for tipo in ['Dia Útil', 'Fim de Semana']:
            dt = sub[sub['Tipo_Dia'] == tipo]
            ax.plot(dt['Hora'], dt['Global_active_power'], label=tipo, color=cores_tipo[tipo], linewidth=2.1)
        ax.set_title(est, fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Hora (0–23h)')
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(0, perfil_est['Global_active_power'].max() * 1.15)
    axes[0].set_ylabel('Potência Ativa Média (kW)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.96))
    fig.suptitle('O inverno amplifica os picos matinal e noturno, enquanto o verão achata o perfil diurno',
                 fontsize=12.5, fontweight='bold', x=0.01, ha='left', y=1.02)
    plt.tight_layout()
    plt.show()

    # FIGURA 5: Drill-down Diagnóstico em Agosto de 2009
    agosto_2009 = diario_subs.loc['2009-08-01':'2009-08-31']
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    bottom_ag = np.zeros(len(agosto_2009))
    dias_str = [d.strftime('%d/%m') for d in agosto_2009.index]
    for col, nome, cor in zip(cols_energia, nomes_leg, cores_subs):
        vals = agosto_2009[col].fillna(0).values
        ax.bar(dias_str, vals, bottom=bottom_ag, label=nome, color=cor, width=0.72)
        bottom_ag += vals
    ax.set_ylim(0, max(bottom_ag.max() * 1.25, 25))
    plt.xticks(rotation=45, ha='right')
    ax.axvspan(7.5, 21.5, color='#f1c40f', alpha=0.18, label='Período de consumo basal (~4–6 kWh/dia)')
    ax.set_title('Drill-down em Agosto/2009: consumo cai para ~5 kWh/dia durante duas semanas (férias)',
                 fontsize=12, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Dia do Mês (Agosto de 2009)')
    ax.set_ylabel('Energia Diária (kWh/dia)')
    ax.legend(loc='upper right', fontsize=8.8)
    plt.tight_layout()
    plt.show()