import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from river.drift import ADWIN
except ImportError as exc:
    raise ImportError("A biblioteca 'river' não está instalada. Corre: pip install river") from exc


def processar_fluxo_tempo_real(df_min, janela_min=60, limiar_z=3.5, delta_adwin=0.002):
    """
    Processa o fluxo ao minuto estritamente elemento a elemento (O(1) por leitura):
      - Média e variância globais online via algoritmo de Welford
      - Janela deslizante de 60 minutos
      - Média móvel exponencial (EWMA)
      - Deteção de concept drift com ADWIN (River) sobre blocos online de 15 minutos
      - Alarme de anomalias via Z-score em janela deslizante
    """
    timestamps = df_min.index.to_numpy()
    valores = df_min["Global_active_power"].to_numpy(dtype=float)

    # 1. Estado do algoritmo de Welford (média e variância globais)
    n_welford = 0
    mean_welford = 0.0
    m2_welford = 0.0

    # 2. Estado da janela deslizante de 60 min
    janela = deque(maxlen=janela_min)
    soma_janela = 0.0

    # 3. Estado da EWMA (alpha equivalente a span=60 min -> 2 / (60 + 1))
    alpha_ewma = 2.0 / (janela_min + 1.0)
    ewma_atual = None

    # 4. Estado do ADWIN (blocos online de 15 minutos)
    adwin = ADWIN(delta=delta_adwin)
    buffer_15m = []
    ts_inicio_15m = None
    ts_15m_lista = []
    val_15m_lista = []
    drifts = []

    # 5. Estado do alarme Z-Score em janela
    alarmes = []
    ultimo_alarme_ts = None  # cooldown de 60 min para registar eventos independentes

    # Buffers para guardar as séries online apenas para visualização posterior
    ts_validos = []
    serie_bruta = []
    serie_janela = []
    serie_ewma = []
    serie_z = []

    t0 = time.perf_counter()

    # =========================================================================
    # LOOP ELEMENTO A ELEMENTO (PROIBIDO OLHAR PARA O ARRAY COMPLETO AQUI)
    # =========================================================================
    for ts, x in zip(timestamps, valores):
        if np.isnan(x):
            continue

        ts_pd = pd.Timestamp(ts)

        # --- A. Alarme Z-Score avaliado ANTES de atualizar a janela ---
        z_score = 0.0
        if len(janela) >= 30:
            mu_local = soma_janela / len(janela)
            var_local = sum((v - mu_local) ** 2 for v in janela) / (len(janela) - 1)
            std_local = max(np.sqrt(var_local), 0.015)  # piso numérico mínimo
            z_score = (x - mu_local) / std_local

            if abs(z_score) > limiar_z:
                if ultimo_alarme_ts is None or (ts_pd - ultimo_alarme_ts).total_seconds() >= 3600:
                    alarmes.append({
                        "Timestamp": ts_pd,
                        "Potencia_kW": float(x),
                        "Media_Janela_kW": float(mu_local),
                        "Std_Janela_kW": float(std_local),
                        "Z_Score": float(z_score),
                    })
                    ultimo_alarme_ts = ts_pd

        # --- B. Welford Online ---
        n_welford += 1
        delta = x - mean_welford
        mean_welford += delta / n_welford
        delta2 = x - mean_welford
        m2_welford += delta * delta2

        # --- C. Janela Deslizante de 60 min (O(1)) ---
        if len(janela) == janela_min:
            soma_janela -= janela[0]
        janela.append(x)
        soma_janela += x
        media_janela_atual = soma_janela / len(janela)

        # --- D. EWMA Online ---
        if ewma_atual is None:
            ewma_atual = float(x)
        else:
            ewma_atual = alpha_ewma * float(x) + (1.0 - alpha_ewma) * ewma_atual

        # --- E. Agregação Online de 15 min + ADWIN (River) ---
        if ts_inicio_15m is None:
            ts_inicio_15m = ts_pd
        buffer_15m.append(float(x))

        if len(buffer_15m) == 15:
            media_15m = sum(buffer_15m) / 15.0
            ts_15m_lista.append(ts_inicio_15m)
            val_15m_lista.append(media_15m)

            adwin.update(media_15m)
            if adwin.drift_detected:
                drifts.append({
                    "Timestamp": ts_inicio_15m,
                    "Media_15m_kW": media_15m,
                    "Estimativa_ADWIN_kW": float(adwin.estimation),
                })

            buffer_15m = []
            ts_inicio_15m = None

        # Guardar outputs do passo atual
        ts_validos.append(ts_pd)
        serie_bruta.append(float(x))
        serie_janela.append(float(media_janela_atual))
        serie_ewma.append(float(ewma_atual))
        serie_z.append(float(z_score))

    tempo_total = time.perf_counter() - t0
    var_welford = m2_welford / (n_welford - 1)

    # =========================================================================
    # VERIFICAÇÃO WELFORD VS. BATCH (ÚNICO MOMENTO EM QUE SE USA O ARRAY TODO)
    # =========================================================================
    arr_batch = np.array(serie_bruta, dtype=float)
    mean_batch = float(np.mean(arr_batch))
    var_batch = float(np.var(arr_batch, ddof=1))

    df_stream = pd.DataFrame({
        "Bruto_kW": serie_bruta,
        "Janela_60m_kW": serie_janela,
        "EWMA_kW": serie_ewma,
        "Z_Score": serie_z,
    }, index=pd.DatetimeIndex(ts_validos))

    df_15m = pd.DataFrame({
        "Media_15m_kW": val_15m_lista
    }, index=pd.DatetimeIndex(ts_15m_lista))

    resumo_welford = {
        "n_leituras": n_welford,
        "tempo_s": tempo_total,
        "mean_welford": mean_welford,
        "mean_batch": mean_batch,
        "diff_mean": abs(mean_welford - mean_batch),
        "var_welford": var_welford,
        "var_batch": var_batch,
        "diff_var": abs(var_welford - var_batch),
        "alpha_ewma": alpha_ewma,
    }

    return df_stream, df_15m, pd.DataFrame(drifts), pd.DataFrame(alarmes), resumo_welford


def plot_comparacao_dia_4_1(df_stream, dia_str="2008-11-23", alpha_ewma=0.0328):
    """Figura 4.1: Leitura Bruta vs. Janela Deslizante 60 min vs. EWMA num dia interessante."""
    df_dia = df_stream.loc[dia_str]

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(df_dia.index, df_dia["Bruto_kW"], color="#95a5a6", alpha=0.55, linewidth=0.9,
            label="Leitura Bruta ao Minuto (kW)")
    ax.plot(df_dia.index, df_dia["Janela_60m_kW"], color="#1f618d", linewidth=2.0,
            label="Janela Deslizante 60 min (Online)")
    ax.plot(df_dia.index, df_dia["EWMA_kW"], color="#d35400", linestyle="--", linewidth=2.0,
            label=f"EWMA Online (α = {alpha_ewma:.4f})")

    ax.set_title(f"Figura 10 (4.1) — Replay em Tempo Real ({dia_str}): Bruto vs. Janela 60 min vs. EWMA",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel("Potência Ativa (kW)")
    ax.set_xlabel("Hora do Dia")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()


def plot_drift_adwin_4_2(df_15m, df_drifts):
    """Figura 4.2: Deteção de Concept Drift (ADWIN) ao longo dos 18 meses."""
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(df_15m.index, df_15m["Media_15m_kW"], color="#5d6d7e", alpha=0.65, linewidth=0.7,
            label="Agregado Online 15 min (kW)")
    if not df_drifts.empty:
        ax.scatter(df_drifts["Timestamp"], df_drifts["Media_15m_kW"], color="#c0392b", s=28, zorder=5,
                   label=f"Drift Detetado pelo ADWIN (n = {len(df_drifts)})")

    ax.set_title("Figura 11 (4.2) — Deteção de Concept Drift com ADWIN (River) nos 18 Meses",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel("Potência Média 15 min (kW)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()


def inspecionar_5_alarmes(df_min, df_alarmes, df_stream, janela_min=60, limiar_z=3.5):
    """Seleciona 5 alarmes representativos, enriquece com contexto físico e desenha zoom."""
    if len(df_alarmes) == 0:
        return pd.DataFrame()

    idx_amostra = np.linspace(0, len(df_alarmes) - 1, min(5, len(df_alarmes)), dtype=int)
    amostra = df_alarmes.iloc[idx_amostra].copy()

    registos = []
    for i, (_, row) in enumerate(amostra.iterrows(), start=1):
        ts = row["Timestamp"]
        reg = df_min.loc[ts]

        # Critério objetivo de diagnóstico para apoiar a avaliação qualitativa
        if row["Potencia_kW"] < 1.5 and row["Std_Janela_kW"] < 0.08:
            diagnostico = "FALSO ALARME (ciclo normal de frio sobre janela muito calma / std colapsado)"
        elif row["Potencia_kW"] >= 3.35 and (reg["Sub_metering_1"] > 0 or reg["Sub_metering_2"] > 0 or reg["Sub_metering_3"] > 0):
            diagnostico = "ALARME REAL (pico elevado com disparo simultâneo de eletrodomésticos)"
        else:
            diagnostico = "EVENTO TRANSITÓRIO (arranque pontual de carga; verificar persistência)"

        registos.append({
            "Alarme_ID": f"A{i}",
            "Timestamp": ts,
            "Hora": ts.strftime("%H:%M"),
            "Potencia_kW": round(row["Potencia_kW"], 3),
            "Media_60m_kW": round(row["Media_Janela_kW"], 3),
            "Std_60m_kW": round(row["Std_Janela_kW"], 3),
            "Z_Score": round(row["Z_Score"], 2),
            "Sub1_Wh": reg["Sub_metering_1"],
            "Sub2_Wh": reg["Sub_metering_2"],
            "Sub3_Wh": reg["Sub_metering_3"],
            "Tensao_V": round(reg["Voltage"], 1),
            "Avaliacao_Tecnica": diagnostico,
        })

    df_5 = pd.DataFrame(registos)

    # Gráfico de zoom (2h antes até 1h depois) para os 5 alarmes
    n_plots = len(df_5)
    fig, axes = plt.subplots(1, n_plots, figsize=(3.6 * n_plots, 3.8), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (_, r) in zip(axes, df_5.iterrows()):
        ts = r["Timestamp"]
        recorte = df_stream.loc[ts - pd.Timedelta(minutes=90) : ts + pd.Timedelta(minutes=45)]
        ax.plot(recorte.index, recorte["Bruto_kW"], color="#2c3e50", linewidth=1.2, label="Bruto (kW)")
        ax.plot(recorte.index, recorte["Janela_60m_kW"], color="#1f618d", linestyle="--", linewidth=1.2, label="Média 60m")
        ax.scatter([ts], [r["Potencia_kW"]], color="#c0392b", s=60, zorder=5, label=f"Z={r['Z_Score']}")
        ax.set_title(f"{r['Alarme_ID']} ({ts.strftime('%Y-%m-%d %H:%M')})\nZ = {r['Z_Score']} | P = {r['Potencia_kW']} kW",
                     fontsize=9.5, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    axes[0].set_ylabel("Potência Ativa (kW)")
    plt.tight_layout()
    plt.show()

    return df_5