import pandas as pd
import numpy as np
from scipy import stats

def gerar_sumario(df):
    print("\n--- Sumário Estatístico ---")
    sumario = df.describe()
    print(sumario)
    return sumario

def calcular_outliers(df, coluna='Global_active_power'):
    print(f"\n--- Análise de Outliers ({coluna}) ---")
    
    col = df[coluna].dropna()
    
    # Método 1: Regra 1.5 x IQR
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    limite_inf_iqr = Q1 - 1.5 * IQR
    limite_sup_iqr = Q3 + 1.5 * IQR
    outliers_iqr = col[(col < limite_inf_iqr) | (col > limite_sup_iqr)]
    
    print(f"Método IQR: {len(outliers_iqr)} outliers detetados (Limites: {limite_inf_iqr:.2f} a {limite_sup_iqr:.2f})")
    
    # Método 2: Z-Score Robusto
    mediana = col.median()
    mad = stats.median_abs_deviation(col)
    z_score_robusto = 0.6745 * (col - mediana) / mad 
    outliers_z = col[np.abs(z_score_robusto) > 3]
    
    print(f"Método Z-Score Robusto: {len(outliers_z)} outliers detetados.")

def gerar_tabela_dinamica(df, coluna='Global_active_power'):
    print(f"\n--- Tabela Dinâmica: Consumo Médio ({coluna}) ---")
    
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
    
    print(pivot.head()) # Imprime as primeiras linhas
    return pivot