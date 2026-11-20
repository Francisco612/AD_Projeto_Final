import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analisar_reducao_dimensao(df_clean):
    print("\n--- [2.2] Perfis Diários com Redução de Dimensão (PCA vs. t-SNE) ---")

    # 1. Matriz de perfis diários (N_dias x 24 horas) usando a MEDIANA horária
    df_h = df_clean[['Global_active_power']].copy()
    df_h['Data'] = df_h.index.date
    df_h['Hora'] = df_h.index.hour

    matriz_perfis = df_h.pivot_table(index='Data', columns='Hora', values='Global_active_power', aggfunc='median')
    dias_antes = len(matriz_perfis)
    matriz_perfis.dropna(inplace=True)
    print(f"Matriz construída: {len(matriz_perfis)} dias completos x 24 horas (excluídos {dias_antes - len(matriz_perfis)} dias incompletos).")

    datas_idx = pd.to_datetime(matriz_perfis.index)
    def obter_estacao(mes):
        if mes in [12, 1, 2]: return 'Inverno'
        elif mes in [3, 4, 5]: return 'Primavera'
        elif mes in [6, 7, 8]: return 'Verão'
        return 'Outono'

    meta_dias = pd.DataFrame({
        'Estacao': datas_idx.month.map(obter_estacao),
        'Tipo_Dia': np.where(datas_idx.dayofweek >= 5, 'Fim de Semana', 'Dia Útil')
    }, index=matriz_perfis.index)

    # 2. Padronização e PCA
    X_scaled = StandardScaler().fit_transform(matriz_perfis)
    pca = PCA(n_components=24, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Garantir que PC1 alto representa consumo elevado
    if pca.components_[0].sum() < 0:
        pca.components_[0] *= -1
        X_pca[:, 0] *= -1

    var_exp = pca.explained_variance_ratio_ * 100
    var_cum = np.cumsum(var_exp)
    print(f"Variância explicada -> PC1: {var_exp[0]:.2f}% | PC2: {var_exp[1]:.2f}% | PC3: {var_exp[2]:.2f}% (Acumulado: {var_cum[2]:.2f}%)")

    # FIGURA 6: Scree Plot + Loadings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    comps = np.arange(1, 25)
    ax1.bar(comps, var_exp, color='#3498db', alpha=0.8, label='Variância Individual (%)')
    ax1.plot(comps, var_cum, color='#e74c3c', marker='o', linewidth=2, label='Variância Acumulada (%)')
    ax1.axhline(80, color='#2c3e50', linestyle='--', linewidth=1.2, label='Limiar 80%')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(range(1, 25, 2))
    ax1.set_title('Scree Plot: PC1 a PC3 concentram a estrutura dominante dos perfis diários', fontsize=11.5, fontweight='bold', loc='left')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.legend(loc='center right')

    horas = np.arange(24)
    loadings = pca.components_[:3]
    ax2.plot(horas, loadings[0], marker='o', linewidth=2.2, color='#1f618d', label=f'PC1 ({var_exp[0]:.1f}%) — Nível Global')
    ax2.plot(horas, loadings[1], marker='s', linewidth=2.0, color='#d35400', label=f'PC2 ({var_exp[1]:.1f}%) — Dia vs. Noite')
    ax2.plot(horas, loadings[2], marker='^', linewidth=1.8, color='#27ae60', label=f'PC3 ({var_exp[2]:.1f}%) — Manhã/Almoço')
    ax2.axhline(0, color='black', linewidth=1, linestyle=':')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_title('Loadings: PC1 mede o volume total e PC2 opõe o período diurno ao noturno', fontsize=11.5, fontweight='bold', loc='left')
    ax2.set_xlabel('Hora do Dia (0h – 23h)')
    ax2.set_ylabel('Peso (Loading)')
    ax2.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

    # 3. t-SNE com seed fixa
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    # FIGURA 7: Projeções PCA vs t-SNE
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    paleta_est = {'Inverno': '#2980b9', 'Primavera': '#27ae60', 'Verão': '#f39c12', 'Outono': '#8e44ad'}
    paleta_dias = {'Dia Útil': '#34495e', 'Fim de Semana': '#e74c3c'}

    for est in ['Inverno', 'Primavera', 'Verão', 'Outono']:
        m = meta_dias['Estacao'] == est
        axes[0, 0].scatter(X_pca[m, 0], X_pca[m, 1], label=est, color=paleta_est[est], alpha=0.72, s=30)
        axes[1, 0].scatter(X_tsne[m, 0], X_tsne[m, 1], label=est, color=paleta_est[est], alpha=0.75, s=30)

    for tipo in ['Dia Útil', 'Fim de Semana']:
        m = meta_dias['Tipo_Dia'] == tipo
        axes[0, 1].scatter(X_pca[m, 0], X_pca[m, 1], label=tipo, color=paleta_dias[tipo], alpha=0.72, s=30)
        axes[1, 1].scatter(X_tsne[m, 0], X_tsne[m, 1], label=tipo, color=paleta_dias[tipo], alpha=0.75, s=30)

    axes[0, 0].set_title('PCA (PC1 × PC2) por Estação: Gradiente Inverno → Verão ao longo de PC1', fontweight='bold', loc='left')
    axes[0, 0].set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
    axes[0, 0].set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')
    axes[0, 0].legend()

    axes[0, 1].set_title('PCA (PC1 × PC2) por Tipo de Dia: Fins de semana deslocam-se para PC1/PC2 > 0', fontweight='bold', loc='left')
    axes[0, 1].set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)')
    axes[0, 1].set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)')
    axes[0, 1].legend()

    axes[1, 0].set_title('t-SNE (seed=42) por Estação: Agrupa regimes locais mas perde distâncias globais', fontweight='bold', loc='left')
    axes[1, 0].set_xlabel('Dimensão t-SNE 1')
    axes[1, 0].set_ylabel('Dimensão t-SNE 2')
    axes[1, 0].legend()

    axes[1, 1].set_title('t-SNE (seed=42) por Tipo de Dia: Isola micro-clusters de rotinas semelhantes', fontweight='bold', loc='left')
    axes[1, 1].set_xlabel('Dimensão t-SNE 1')
    axes[1, 1].set_ylabel('Dimensão t-SNE 2')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()