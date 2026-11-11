from scipy import stats

# Resumo base
sumario = df.describe()

# Cálculo de Outliers para 'Global_active_power'
col = df['Global_active_power'].dropna()

# Regra 1.5 x IQR
Q1 = col.quantile(0.25)
Q3 = col.quantile(0.75)
IQR = Q3 - Q1
limite_inferior_iqr = Q1 - 1.5 * IQR
limite_superior_iqr = Q3 + 1.5 * IQR
outliers_iqr = col[(col < limite_inferior_iqr) | (col > limite_superior_iqr)]

# Z-Score Robusto (usando Mediana e MAD)
mediana = col.median()
mad = stats.median_abs_deviation(col)
# Constante 0.6745 faz o MAD ser comparável ao desvio padrão numa distribuição normal
z_score_robusto = 0.6745 * (col - mediana) / mad 
outliers_z = col[np.abs(z_score_robusto) > 3]

# Sumário Estratificado: Consumo médio por Hora x Dia da Semana (Pivot Table)[cite: 2]
df['Hora'] = df.index.hour
df['Dia_Semana'] = df.index.day_name() # ou dayofweek

# Criar a Tabela Dinâmica
pivot_consumo = pd.pivot_table(
    df, 
    values='Global_active_power', 
    index='Hora', 
    columns='Dia_Semana', 
    aggfunc=['mean', 'median', 'count']
)
# Reordenar colunas para ordem cronológica
dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_consumo = pivot_consumo.reindex(columns=dias_ordem, level=1)