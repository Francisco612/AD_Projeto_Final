import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preparar_dados_temporais(df):
    # Interpolação apenas para falhas curtas (<= 60 min), mantendo dias completos em falta como NaN e preservando picos reais
    cols_originais = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    df_clean = df[cols_originais].interpolate(method='time', limit=60).copy()
    
    carga_basal_kw = float(df_clean['Global_active_power'].quantile(0.10))
    
    df_clean['Carga_Basal_kW'] = carga_basal_kw
    df_clean['Energia_Total_Wh'] = (df_clean['Global_active_power'] * 1000.0) / 60.0
    df_clean['Sub_metering_outros'] = (
        df_clean['Energia_Total_Wh'] - (
            df_clean['Sub_metering_1'] + df_clean['Sub_metering_2'] + df_clean['Sub_metering_3']
        )
    ).clip(lower=0)

    def obter_estacao(mes):
        if mes in [12, 1, 2]: return 'Inverno'
        elif mes in [3, 4, 5]: return 'Primavera'
        elif mes in [6, 7, 8]: return 'Verão'
        return 'Outono'

    df_clean['Hora'] = df_clean.index.hour
    df_clean['Dia_Semana_Num'] = df_clean.index.dayofweek
    df_clean['Tipo_Dia'] = np.where(df_clean['Dia_Semana_Num'] >= 5, 'Fim de Semana', 'Dia Útil')
    df_clean['Estacao'] = df_clean.index.month.map(obter_estacao)
    
    return df_clean, carga_basal_kw

def gerar_figuras_sazonalidade_e_diagnostico(df_clean, carga_basal_kw):
    print("\n--- [2.1] A gerar Figuras de Sazonalidade (Mediana + Carga Basal) e Diagnóstico ---")
    sns.set_theme(style="whitegrid")
    energia_basal_kwh_dia = carga_basal_kw * 24.0

    # FIGURA 1: Sazonalidade Diária (Mediana + IQR + Carga Basal Constante)
    perfil_h = df_clean.groupby('Hora')['Global_active_power'].agg(
        mediana='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(perfil_h.index, perfil_h['q25'], perfil_h['q75'], color='#3498db', alpha=0.22, label='Intervalo Interquartil (P25–P75)')
    ax.plot(perfil_h.index, perfil_h['mediana'], color='#1f618d', linewidth=2.5, marker='o', label='Mediana Horária (kW)')
    ax.axhline(carga_basal_kw, color='#c0392b', linestyle=':', linewidth=1.8,
               label=f'Carga Basal Constante ({carga_basal_kw:.2f} kW — frigorífico/standby)')

    pico_h, pico_v = perfil_h['mediana'].idxmax(), perfil_h['mediana'].max()
    ax.annotate(f'Pico Noturno ({pico_h}h: {pico_v:.2f} kW)',
                xy=(pico_h, pico_v), xytext=(pico_h - 5.5, pico_v + 0.42),
                arrowprops=dict(arrowstyle='->', color='#1f618d', lw=1.5),
                fontsize=10, fontweight='bold', color='#1f618d')
    ax.set_title('A mediana horária revela dois picos ocupacionais (manhã e noite) acima da carga basal fixa',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Hora do Dia (0h – 23h)')
    ax.set_ylabel('Potência Ativa Mediana (kW)')
    ax.set_xticks(range(0, 24))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # FIGURA 2: Sazonalidade Semanal (Mediana Diária + Carga Basal Constante, eixo Y a 0)
    energia_diaria = df_clean['Energia_Total_Wh'].resample('D').sum(min_count=1200) / 1000.0
    df_dias = pd.DataFrame({'Energia_kWh': energia_diaria, 'Dia_Num': energia_diaria.index.dayofweek})
    nomes_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    resumo_sem = df_dias.groupby('Dia_Num')['Energia_kWh'].median()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(nomes_dias, resumo_sem.values, color=['#5d6d7e']*5 + ['#d35400']*2, width=0.62, edgecolor='black', linewidth=0.6)
    ax.set_ylim(0, resumo_sem.max() * 1.20)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f'{h:.1f} kWh', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.axhline(energia_basal_kwh_dia, color='#c0392b', linestyle=':', linewidth=1.8,
               label=f'Carga Basal Constante ({energia_basal_kwh_dia:.1f} kWh/dia)')
    med_uteis, med_fds = resumo_sem.iloc[:5].median(), resumo_sem.iloc[5:].median()
    aumento_pct = ((med_fds - med_uteis) / med_uteis) * 100
    ax.axhline(med_uteis, color='#2c3e50', linestyle='--', linewidth=1.3, label=f'Mediana Dias Úteis ({med_uteis:.1f} kWh/dia)')
    ax.annotate(f'Fim de semana: +{aumento_pct:.1f}% na mediana',
                xy=(5.5, med_fds), xytext=(3.1, resumo_sem.max() * 1.09),
                arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.5),
                fontsize=10, fontweight='bold', color='#d35400')
    ax.set_title(f'A mediana diária sobe {aumento_pct:.0f}% ao fim de semana, assente numa carga basal fixa de {energia_basal_kwh_dia:.1f} kWh/dia',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Dia da Semana')
    ax.set_ylabel('Energia Mediana Diária (kWh/dia)')
    ax.legend(loc='upper left', fontsize=9.5)
    plt.tight_layout()
    plt.show()

    # FIGURA 3: Sazonalidade Anual (Mediana Mensal por Circuito + Carga Basal Constante)
    cols_energia = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_outros']
    diario_subs = df_clean[cols_energia].resample('D').sum(min_count=1200) / 1000.0
    mensal_subs = diario_subs.resample('MS').median()
    labels_meses = [d.strftime('%Y-%m') for d in mensal_subs.index]
    nomes_leg = ['Sub 1: Cozinha', 'Sub 2: Lavandaria', 'Sub 3: Climatização/Águas Quentes', 'Outros: Iluminação/Geral']
    cores_subs = ['#e74c3c', '#f39c12', '#2980b9', '#95a5a6']

    fig, ax = plt.subplots(figsize=(12.5, 5))
    bottom = np.zeros(len(mensal_subs))
    for col, nome, cor in zip(cols_energia, nomes_leg, cores_subs):
        vals = mensal_subs[col].fillna(0).values
        ax.bar(labels_meses, vals, bottom=bottom, label=nome, color=cor, width=0.68)
        bottom += vals

    ax.axhline(energia_basal_kwh_dia, color='#c0392b', linestyle='--', linewidth=1.7,
               label=f'Carga Basal Constante ({energia_basal_kwh_dia:.1f} kWh/dia)')
    ax.set_ylim(0, bottom.max() * 1.24)
    plt.xticks(rotation=45, ha='right')
    idx_max, idx_min = int(np.argmax(bottom)), int(np.argmin(bottom))
    ax.annotate(f'Pico Inverno ({labels_meses[idx_max]}: {bottom[idx_max]:.1f} kWh/dia)',
                xy=(idx_max, bottom[idx_max]), xytext=(max(0, idx_max - 2), bottom[idx_max] + 4.0),
                arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.5),
                fontsize=10, fontweight='bold', color='#1a5276')
    ax.annotate(f'Vale Verão ({labels_meses[idx_min]}: {bottom[idx_min]:.1f} kWh/dia)',
                xy=(idx_min, bottom[idx_min]), xytext=(max(0, idx_min - 2), bottom[idx_min] + 6.0),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                fontsize=10, fontweight='bold', color='#c0392b')
    ax.set_title('A mediana diária duplica entre o verão e o inverno devido à climatização e águas quentes',
                 fontsize=12.5, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Mês de Referência')
    ax.set_ylabel('Energia Mediana Diária (kWh/dia)')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

    # FIGURA 4: Small Multiples por Estação (Mediana + Carga Basal Constante)
    estacoes_ordem = ['Inverno', 'Primavera', 'Verão', 'Outono']
    perfil_est = df_clean.groupby(['Estacao', 'Tipo_Dia', 'Hora'])['Global_active_power'].median().reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True, sharex=True)
    cores_tipo = {'Dia Útil': '#2c3e50', 'Fim de Semana': '#e67e22'}
    for ax, est in zip(axes, estacoes_ordem):
        sub = perfil_est[perfil_est['Estacao'] == est]
        for tipo in ['Dia Útil', 'Fim de Semana']:
            dt = sub[sub['Tipo_Dia'] == tipo]
            ax.plot(dt['Hora'], dt['Global_active_power'], label=tipo, color=cores_tipo[tipo], linewidth=2.1)
        ax.axhline(carga_basal_kw, color='#c0392b', linestyle=':', linewidth=1.5, label='Carga Basal Constante')
        ax.set_title(est, fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Hora (0–23h)')
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(0, perfil_est['Global_active_power'].max() * 1.18)

    axes[0].set_ylabel('Potência Ativa Mediana (kW)')
    axes[0].annotate('Pico noturno\nmais intenso', xy=(20, 1.8), xytext=(6, 2.15),
                     arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.3),
                     fontsize=9, fontweight='bold', color='#d35400')
    axes[2].annotate('Consumo próximo\nda carga basal', xy=(14, 0.45), xytext=(5, 1.35),
                     arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.3),
                     fontsize=9, fontweight='bold', color='#2c3e50')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.96))
    fig.suptitle('Perfis medianos por estação: o consumo variável assenta sobre uma carga basal constante',
                 fontsize=12.5, fontweight='bold', x=0.01, ha='left', y=1.02)
    plt.tight_layout()
    plt.show()

    # FIGURA 5: Drill-down Diagnóstico em Agosto de 2009 vs Carga Basal Constante
    agosto_2009 = diario_subs.loc['2009-08-01':'2009-08-31']
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    bottom_ag = np.zeros(len(agosto_2009))
    dias_str = [d.strftime('%d/%m') for d in agosto_2009.index]
    for col, nome, cor in zip(cols_energia, nomes_leg, cores_subs):
        vals = agosto_2009[col].fillna(0).values
        ax.bar(dias_str, vals, bottom=bottom_ag, label=nome, color=cor, width=0.72)
        bottom_ag += vals

    ax.axhline(energia_basal_kwh_dia, color='#c0392b', linestyle='--', linewidth=1.8,
               label=f'Carga Basal Constante ({energia_basal_kwh_dia:.1f} kWh/dia)')
    ax.axvspan(7.5, 21.5, color='#f1c40f', alpha=0.16, label='Ausência / Férias (Consumo variável ≈ 0 kWh)')
    ax.annotate('Consumo encosta na linha constante\n(apenas frigorífico e standby ativos)',
                xy=(14, energia_basal_kwh_dia), xytext=(8.5, 18),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                fontsize=10, fontweight='bold', color='#922b21')
    ax.set_ylim(0, max(bottom_ag.max() * 1.25, 25))
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Drill-down Agosto/2009: a linha basal constante evidencia duas semanas de ausência total',
                 fontsize=12, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('Dia do Mês (Agosto de 2009)')
    ax.set_ylabel('Energia Diária (kWh/dia)')
    ax.legend(loc='upper right', fontsize=8.8)
    plt.tight_layout()
    plt.show()