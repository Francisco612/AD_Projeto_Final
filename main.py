import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Ocultar avisos do pandas para manter a consola limpa

from src.carregamento import carregar_dados
from src.auditoria import analisar_missing_values
from src.estatistica import gerar_sumario, calcular_outliers, gerar_tabela_dinamica

def main():
    print("=======================================")
    print("  PROJETO LUSOENERGIA - PARTE 1")
    print("=======================================")
    
    # 0. Configurações 
    SOMA_NUMEROS_ALUNO = 101833
    rng = np.random.default_rng(SOMA_NUMEROS_ALUNO)
    inicio = pd.Timestamp(rng.choice(pd.date_range('2007-01-01', '2009-05-01', freq='MS')))
    fim = inicio + pd.DateOffset(months=18)

    print(f"Janela de análise sorteada para o grupo:")
    print(f" -> Início: {inicio.strftime('%Y-%m-%d')}")
    print(f" -> Fim:    {fim.strftime('%Y-%m-%d')}\n")

    caminho_ficheiro = 'C:\\Users\\franc\\Documents\\GitHub\\AD_Projeto_Final\\data\\raw\\individual+household+electric+power+consumption.zip'
    
    # 1. Carregamento
    print("\n[Passo 1] Carregamento e Tipagem de Dados...")
    df = carregar_dados(caminho_ficheiro, inicio, fim)
    print(f"Sucesso! O dataset tem {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    # 2. Auditoria
    print("\n[Passo 2] Auditoria de Qualidade...")
    analisar_missing_values(df)
    
    # 3. Estatística e Outliers
    print("\n[Passo 3] Exploração Estatística e Anomalias...")
    gerar_sumario(df)
    calcular_outliers(df, coluna='Global_active_power')
    gerar_tabela_dinamica(df, coluna='Global_active_power')
    
    print("\nFim da análise da Parte 1!")

if __name__ == "__main__":
    main()