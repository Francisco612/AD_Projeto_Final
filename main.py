import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.carregamento import carregar_dados
from src.auditoria import analisar_missing_values
from src.estatistica import (
    investigar_outliers,
    analisar_distribuicao_e_assimetria,
    decompor_carga_basal,
    gerar_tabela_dinamica
)
from src.visualizacao import preparar_dados_temporais, gerar_figuras_sazonalidade_e_diagnostico
from src.reducao_dimensao import analisar_reducao_dimensao

def main():
    print("=======================================")
    print("  PROJETO LUSOENERGIA - PARTES 1 e 2")
    print("=======================================")
    
    # 0. Janela temporal de 18 meses do grupo
    SOMA_DOS_NUMEROS_DE_ALUNO = 101833
    rng = np.random.default_rng(SOMA_DOS_NUMEROS_DE_ALUNO)
    inicio = pd.Timestamp(rng.choice(pd.date_range('2007-01-01', '2009-05-01', freq='MS')))
    fim = inicio + pd.DateOffset(months=18)
    print(f"Janela temporal: {inicio.strftime('%Y-%m-%d')} a {fim.strftime('%Y-%m-%d')}")

    caminho_ficheiro = 'data/raw/individual+household+electric+power+consumption.zip'
    
    # --- PARTE I ---
    print("\n[Parte 1.1] Carregamento e Tipagem...")
    df = carregar_dados(caminho_ficheiro, inicio, fim)
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas.")
    
    print("\n[Parte 1.2] Auditoria de Qualidade...")
    analisar_missing_values(df)
    
    print("\n[Parte 1.3] Investigação de Outliers, Assimetria, Carga Basal e Tabela Dinâmica...")
    df_analise = investigar_outliers(df, coluna='Global_active_power')
    analisar_distribuicao_e_assimetria(df)
    df_decomposed, carga_basal_kw = decompor_carga_basal(df_analise)
    gerar_tabela_dinamica(df, coluna='Global_active_power')

    # --- PARTE II ---
    print("\n[Parte 2.1] Sazonalidade (Mediana + Carga Basal) e Diagnóstico...")
    df_clean, carga_basal_kw = preparar_dados_temporais(df)
    gerar_figuras_sazonalidade_e_diagnostico(df_clean, carga_basal_kw)

    print("\n[Parte 2.2] Perfis Diários com Redução de Dimensão (PCA vs. t-SNE)...")
    analisar_reducao_dimensao(df_clean)
    
    print("\nExecução concluída com sucesso!")

if __name__ == "__main__":
    main()