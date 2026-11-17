import pandas as pd
import numpy as np
from scipy import stats

def remover_outliers(df, coluna='Global_active_power', metodo='iqr'):
    print(f"\n--- Remoção de Outliers ({coluna} | Método: {metodo.upper()}) ---")
    
    col_valida = df[coluna].dropna()
    
    if metodo == 'iqr':
        Q1 = col_valida.quantile(0.25)
        Q3 = col_valida.quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        
        is_outlier = (df[coluna] < limite_inf) | (df[coluna] > limite_sup)
        print(f"Limites IQR definidos: [{limite_inf:.2f}, {limite_sup:.2f}]")
        
    elif metodo == 'zscore':
        mediana = col_valida.median()
        mad = stats.median_abs_deviation(col_valida)
        z_score_robusto = 0.6745 * (df[coluna] - mediana) / mad
        
        is_outlier = np.abs(z_score_robusto) > 3
        print("Critério aplicado: |Z-Score Robusto| > 3")
    else:
        raise ValueError("Método inválido. Escolha 'iqr' ou 'zscore'.")

    # Remove apenas os registos sinalizados como outliers (mantendo eventuais NaNs intactos)
    df_limpo = df[~is_outlier].copy()
    
    n_removidos = is_outlier.sum()
    pct_removidos = (n_removidos / len(col_valida)) * 100
    print(f"Outliers removidos: {n_removidos} linhas ({pct_removidos:.2f}% dos dados válidos)")
    print(f"Registos restantes para análise: {len(df_limpo)} linhas")
    
    return df_limpo

def gerar_sumario(df):
    print("\n--- Sumário Estatístico (Sem Outliers) ---")
    sumario = df.describe()
    print(sumario)
    return sumario

def gerar_tabela_dinamica(df, coluna='Global_active_power'):
    print(f"\n--- Tabela Dinâmica: Consumo Médio Sem Outliers ({coluna}) ---")
    
    df_temp = df.copy()
    df_temp['Hora'] = df_temp.index.hour
    df_temp['Dia_Semana'] = df_temp.index.day_name()
    
    pivot = pd.pivot_table(
        df_temp, 
        values=coluna, 
        index='Hora', 
        columns='Dia_Semana', 
        aggfunc='mean'
    )
    
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=dias_ordem)
    
    print(pivot.head())
    return pivot