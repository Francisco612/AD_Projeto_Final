import pandas as pd

def carregar_dados(caminho_zip, inicio=None, fim=None):
    dtypes = {
        'Global_active_power': 'float32',
        'Global_reactive_power': 'float32',
        'Voltage': 'float32',
        'Global_intensity': 'float32',
        'Sub_metering_1': 'float32',
        'Sub_metering_2': 'float32',
        'Sub_metering_3': 'float32'
    }
    
    # Lê o conteúdo do zip diretamente
    df = pd.read_csv(
        caminho_zip, 
        sep=';', 
        na_values=['?'], 
        dtype=dtypes,
        parse_dates={'datetime': ['Date', 'Time']},
        date_format='%d/%m/%Y %H:%M:%S',
        low_memory=False,
        compression='zip'  # Descompacta dinamicamente
    )
    
    df.set_index('datetime', inplace=True)
    
    if inicio and fim:
        df = df.loc[inicio:fim]

    assert isinstance(df.index, pd.DatetimeIndex), "O índice não é um DatetimeIndex."
    assert df.index.is_monotonic_increasing, "O índice temporal não está ordenado."
    assert df.index.is_unique, "Existem datas/horas duplicadas no índice."
    if not df['Global_active_power'].isna().all():
        assert (df['Global_active_power'].dropna() >= 0).all(), "Existem valores negativos de potência."
    assert df.shape[1] == 7, "O número de variáveis (colunas) não corresponde ao esperado."
    
    return df