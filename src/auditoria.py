import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_missing_values(df):
    print("\n--- Auditoria de Qualidade dos Dados ---")
    
    # 1. Total e Percentagem
    missing_abs = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df)) * 100
    
    resumo_missing = pd.DataFrame({
        'Total_Nulos': missing_abs,
        'Percentagem_(%)': missing_pct
    })
    print("\nResumo de Valores Nulos por Variável:")
    print(resumo_missing)
    
    # 2. Lacunas Estruturais (Dias inteiros sem leituras)
    leituras_diarias = df['Global_active_power'].resample('D').count()
    dias_sem_leitura = leituras_diarias[leituras_diarias == 0]
    print(f"\nNúmero de dias inteiros sem qualquer medição: {len(dias_sem_leitura)}")
    
    if len(dias_sem_leitura) > 0:
        print("Primeiros 5 dias sem medições:")
        print(dias_sem_leitura.head().index.date)

    # 3. Visualização (Mapa Térmico Diário)
    plt.figure(figsize=(12, 5))
    heatmap_data = df.resample('D').apply(lambda x: x.isna().mean()).T
    sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': '% de Nulos'})
    plt.title('Mapa Térmico de Valores em Falta por Dia')
    plt.xlabel('Data')
    plt.ylabel('Variáveis')
    plt.tight_layout()
    plt.show()