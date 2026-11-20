import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def analisar_missing_values(df):
    missing_abs = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df)) * 100
    resumo_missing = pd.DataFrame({
        'Total_Nulos': missing_abs,
        'Percentagem_(%)': missing_pct.round(3)
    })
    
    try:
        display(resumo_missing)
    except NameError:
        print(resumo_missing)

    # Verificar se os nulos ocorrem em simultâneo em todas as variáveis
    simultaneo = (df.isna().nunique(axis=1) == 1).all()
    print(f"Falhas simultâneas em todos os sensores/variáveis: {'Sim' if simultaneo else 'Não'}")

    # Série diária de percentagem de nulos
    pct_nulos_dia = df['Global_active_power'].isna().resample('D').mean() * 100
    dias_sem_leitura = pct_nulos_dia[pct_nulos_dia == 100]
    dias_com_falhas = pct_nulos_dia[pct_nulos_dia > 0]

    print(f"Número de dias inteiros sem qualquer medição (100% nulos): {len(dias_sem_leitura)}")
    if len(dias_sem_leitura) > 0:
        print("Datas 100% sem medição:", [d.strftime('%Y-%m-%d') for d in dias_sem_leitura.index])
    print(f"Total de dias com pelo menos 1 minuto em falta: {len(dias_com_falhas)}")

    if len(dias_com_falhas) == 0:
        print("Não existem valores em falta nesta janela temporal.")
        return resumo_missing, dias_sem_leitura

    # --- FIGURA DE 2 PAINÉIS: VISÃO GERAL + ZOOM NOS DIAS COM FALHAS ---
    fig, (ax_top, ax_zoom) = plt.subplots(
        2, 1, figsize=(14, 7.8),
        gridspec_kw={'height_ratios': [0.34, 0.66]}
    )

    # 1. Painel Superior: Timeline diária nos 18 meses
    ax_top.bar(pct_nulos_dia.index, pct_nulos_dia.values, width=2.2, color='#c0392b', alpha=0.85)
    ax_top.set_ylim(0, 112)
    ax_top.set_ylabel('Nulos / Dia (%)', fontsize=10.5)
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_top.tick_params(axis='x', rotation=30)
    ax_top.grid(axis='y', linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.set_title(
        f'Auditoria de Qualidade: {len(dias_com_falhas)} dias apresentam falhas ({len(dias_sem_leitura)} dias totalmente vazios)',
        fontsize=12.5, fontweight='bold', loc='left', pad=10
    )

    # Anotar o maior bloco de falha no gráfico superior
    dia_max = pct_nulos_dia.idxmax()
    val_max = pct_nulos_dia.max()
    ax_top.annotate(
        f'Máx: {dia_max.strftime("%d/%m/%Y")} ({val_max:.0f}%)',
        xy=(dia_max, val_max), xytext=(dia_max, min(val_max + 15, 102)),
        ha='center', fontsize=9, fontweight='bold', color='#922b21'
    )

    # 2. Painel Inferior (ZOOM): Matriz Hora (0-23h) x Apenas Dias com Falhas
    df_zoom = df[df.index.normalize().isin(dias_com_falhas.index)].copy()
    df_zoom['Data_Str'] = df_zoom.index.strftime('%Y-%m-%d')
    df_zoom['Hora'] = df_zoom.index.hour
    df_zoom['Is_NaN'] = df_zoom['Global_active_power'].isna().astype(float) * 100

    matriz_zoom = df_zoom.pivot_table(
        index='Hora', columns='Data_Str', values='Is_NaN', aggfunc='mean'
    ).reindex(index=range(24), fill_value=0)

    # Mostrar percentagem dentro das células se houver <= 24 dias com falhas (omite zeros para não poluir)
    mostrar_annot = matriz_zoom.shape[1] <= 24
    annot_labels = matriz_zoom.map(lambda v: f'{v:.0f}%' if v > 0 else '') if mostrar_annot else False

    sns.heatmap(
        matriz_zoom,
        ax=ax_zoom,
        cmap='YlOrRd',
        vmin=0, vmax=100,
        linewidths=0.6, linecolor='#ecf0f1',
        annot=annot_labels, fmt='',
        annot_kws={'size': 8, 'weight': 'bold'},
        cbar_kws={'label': 'Minutos em Falta por Hora (%)', 'pad': 0.015}
    )

    ax_zoom.set_title(
        'ZOOM Detalhado (Hora × Dias com Falhas): perfil horário exato de cada interrupção de leitura',
        fontsize=12, fontweight='bold', loc='left', pad=10
    )
    ax_zoom.set_xlabel('Dias em que ocorreram falhas de leitura', fontsize=10.5)
    ax_zoom.set_ylabel('Hora do Dia (0h – 23h)', fontsize=10.5)
    ax_zoom.set_yticks(np.arange(0, 24, 2) + 0.5)
    ax_zoom.set_yticklabels([f'{h:02d}h' for h in range(0, 24, 2)], rotation=0)
    plt.setp(ax_zoom.get_xticklabels(), rotation=45, ha='right', fontsize=9.5)

    plt.tight_layout()
    plt.show()

    return resumo_missing, dias_sem_leitura

resumo_missing, dias_sem_leitura = analisar_missing_values(df)