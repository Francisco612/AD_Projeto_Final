# ⚡ Projeto LusoEnergia — Ciência de Dados Aplicada ao Setor Energético

Plataforma analítica e operacional desenvolvida para apoiar a **Gestão Comercial** e a **Sala de Controlo** da **LusoEnergia**, cobrindo auditoria e qualidade de dados, segmentação de perfis de consumo, previsão causal diária e monitorização em tempo real (*data streaming*).

* **Identificador do Grupo (`SOMA_DOS_NUMEROS_DE_ALUNO`):** `101833`
* **Janela de Análise:** 18 meses consecutivos extraídos de forma determinística a partir do dataset *Individual Household Electric Power Consumption* (resolução de 1 minuto).

---

## 📁 Estrutura do Repositório

```text
projeto-lusoenergia/
├── data/
│   └── raw/
│       └── individual+household+electric+power+consumption.zip  # Dados brutos intocados
├── notebooks/
│   ├── 01_parte1_lusoenergia.ipynb      # Parte I: Qualidade, Sazonalidade e Carga Basal (P10)
│   ├── 02_parte2_lusoenergia.ipynb      # Parte II: Engenharia de Atributos, PCA e t-SNE
│   ├── 03_parte3_lusoenergia.ipynb      # Parte III: Previsão Causal (TimeSeriesSplit + LassoCV)
│   └── 04_parte4_lusoenergia.ipynb      # Parte IV & V: Streaming O(1) (Welford/ADWIN) e Síntese
├── reports/
│   ├── figura_sintese_lusoenergia.png   # Figura-síntese multi-painel do projeto
│   └── relatorio_executivo_lusoenergia.pdf
├── src/
│   ├── __init__.py
│   ├── data_utils.py                    # Carregamento determinístico e limpeza (reutilizado ≥ 3x)
│   └── streaming.py                     # Motor online O(1) (Welford, EWMA, ADWIN, Z-Score)
├── dashboard_lusoenergia.py             # Aplicação interativa Streamlit (Comercial & Controlo)
├── main.py                              # Execução integrada do pipeline (Partes I a IV)
├── requirements.txt                     # Dependências Python com versões fixas
└── README.md                            # Documentação e instruções de reprodutibilidade
```

---

## ⚙️ Instalação e Configuração

### 1. Criar e ativar o ambiente virtual
No terminal, dentro da pasta raiz do projeto:

```bash
python -m venv .venv
```

* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 3. Verificar a localização dos dados brutos
Garantir que o ficheiro original `individual+household+electric+power+consumption.zip` se encontra dentro da pasta `data/raw/`. **Nenhum script altera ou escreve nesta pasta**, preservando os dados brutos intocados.

---

## ▶️ Como Executar o Projeto

### A. Pipeline Completo via Terminal (Partes I a IV)
Executa o carregamento determinístico, estatísticas descritivas, PCA, validação cruzada temporal e simulação de *streaming* com validação numérica de Welford:
```bash
python main.py
```

### B. Dashboard Interativo (Gestão Comercial & Sala de Controlo)
Inicia a aplicação web em Streamlit com simulador tarifário (€), projeção PCA/t-SNE, previsão diária e controlos em tempo real:
```bash
streamlit run dashboard_lusoenergia.py
```

### C. Notebooks Jupyter (Análise Detalhada)
Abrir qualquer notebook na pasta `notebooks/` (ex.: `01_parte1_lusoenergia.ipynb` a `04_parte4_lusoenergia.ipynb`) e executar **Kernel $\rightarrow$ Restart & Run All**.

---

## 🧩 Arquitetura dos Módulos (`src/`)

* **`src/data_utils.py`:** Centraliza a fixação de sementes (`SEED_GLOBAL = 42`), a localização automática do ficheiro em `data/raw/`, a seleção determinística da janela de 18 meses (`SOMA_DOS_NUMEROS_DE_ALUNO = 101833`), a reindexação temporal contínua de 1 minuto, a interpolação curta (até 60 minutos) e o cálculo das variáveis energéticas derivadas (`Energia_Total_Wh` e `Sub_outros_Wh`). É reutilizado nos notebooks, no `main.py` e no `dashboard_lusoenergia.py`.
* **`src/streaming.py`:** Implementa o processamento estritamente elemento a elemento em complexidade $O(1)$ por leitura (sem operações vetorizadas sobre o histórico completo durante o *replay*), incluindo:
  * **Algoritmo de Welford:** Cálculo incremental da média e variância globais, validado contra o resultado *batch* no final do fluxo.
  * **Janela Deslizante & EWMA:** Suavização em tempo real com fila circular (`deque`) e decaimento exponencial ($\alpha = \frac{2}{W + 1}$).
  * **Deteção de Concept Drift (`ADWIN`):** Monitorização de mudanças de regime via biblioteca `river` sobre blocos *online* de 15 minutos.
  * **Alarmística Contextual (Z-Score):** Deteção de picos anómalos avaliada antes da atualização da janela para evitar contaminação estatística.

---

## 🔒 Garantias de Reprodutibilidade

1. **Dados Brutos Intocados:** Leitura exclusiva a partir de `data/raw/`.
2. **Determinismo Completo:** Semente fixa (`random_state=42` / `np.random.seed(42)`) aplicada em todos os componentes estocásticos (`PCA`, `TSNE`, `LassoCV` e geradores aleatórios).
3. **Ausência de *Data Leakage*:** Todas as transformações (`StandardScaler`), seleções de atributos (`SelectFromModel`) e médias móveis causais (aplicadas após `.shift(1)`) são ajustadas exclusivamente dentro das partições cronológicas de treino do `TimeSeriesSplit`.