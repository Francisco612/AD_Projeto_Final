import matplotlib.pyplot as plt
import seaborn as sns

# 1. Quantificar valores em falta por variável
missing_por_variavel = df.isna().sum()
missing_percentagem = (df.isna().sum() / len(df)) * 100

# 2. Encontrar dias inteiros sem leituras
# Reamostrar para contagem diária de valores válidos
leituras_diarias = df['Global_active_power'].resample('D').count()
dias_sem_leitura = leituras_diarias[leituras_diarias == 0]

# Visualização (Exemplo: Heatmap de missing values ao longo do tempo)
plt.figure(figsize=(15, 5))
# Reamostramos para frequência diária mostrando % de nulos para não sobrecarregar o gráfico
sns.heatmap(df.resample('D').apply(lambda x: x.isna().mean()).T, cmap="YlOrRd", cbar_kws={'label': '% Valores em Falta'})
plt.title('Mapa Térmico de Valores em Falta por Dia')
plt.show()