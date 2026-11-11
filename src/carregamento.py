import pandas as pd

def carregar_dados(caminho_zip, inicio=None, fim=None):
    dtypes = {
        'Date': 'str',
        'Time': 'str',
        'Global_active_power': 'float32',
        'Global_reactive_power': 'float32',
        'Voltage': 'float32',
        'Global_intensity': 'float32',
        'Sub_metering_1': 'float32',
        'Sub_metering_2': 'float32',
        'Sub_metering_3': 'float32'
    }
    
    print(f"A ler dados do ficheiro: {caminho_zip}...")
    df = pd.read_csv(
        caminho_zip, 
        sep=';', 
        na_values=['?'], 
        dtype=dtypes,
        low_memory=False,
        compression='zip'
    )
    
    # Combinar Date e Time numa coluna datetime
    print("A converter datas e horas...")
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], 
        format='%d/%m/%Y %H:%M:%S'
    )
    
    # Remover colunas originais de texto e definir o índice temporal
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('datetime', inplace=True)
    
    # Filtrar pela janela temporal do grupo
    if inicio and fim:
        print(f"A filtrar para a janela temporal: {inicio} a {fim}...")
        df = df.loc[inicio:fim]

    # Validações de integridade dos dados
    assert isinstance(df.index, pd.DatetimeIndex), "O índice não é um DatetimeIndex."
    assert df.index.is_monotonic_increasing, "O índice temporal não está ordenado."
    assert df.index.is_unique, "Existem datas/horas duplicadas no índice."
    if not df['Global_active_power'].isna().all():
        assert (df['Global_active_power'].dropna() >= 0).all(), "Existem valores negativos de potência."
    assert df.shape[1] == 7, "O número de variáveis (colunas) não corresponde ao esperado."
    
    return df