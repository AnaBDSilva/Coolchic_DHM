# README — Estrutura do Projeto

Avaliação do Codec Cool-Chic na Compressão de Fase em Holografia Microscópica Digital (DHM).

## Visão Geral da Estrutura

``` 
.
├── recon_holo_comp.py                      # Programa principal que contém os dois fluxos principais
├── Lloyd_DHM_PhaseRecROI_AUTO_COMP.py      # Módulo principal que tem a lógica principal de cada etapa do processo
├── modules/                                # Módulos desenvolvidos em Python durante o projeto (ou reutlizados)
├── pyDHM/                                  # Biblioteca externa para reconstrução numérica de DHM
├── data/                                   # Dados das amostras
└── reconstructions/                        # Dados de teste e resultados de todas as amostras
``` 
---

## `modules/`

Código-fonte principal do projeto.

- **`autoOrderDetection.py`** — Deteção automática da ordem +1 no espectro de Fourier do holograma.
- **`compressionPipeline.py`** — Módulo central: normalização, preparação e interface com o codec Cool-Chic (compressão/descompressão).
- **`pda_metric.py`** — Métricas de avaliação de fase (PDA-PSNR, PDA-SSIM) e as métricas tradicinais (PSNR, SSIM).
- **`processUtils.py`** — Pré-processamento: ajuste a matriz quadrada e apodização.
- **`readResults.py`** — Leitura, agregação e geração dos gráficos de rate-distortion.

---

## `reconstructions/`

Pasta principal com os dados de teste e todos os resultados obtidos.

### `reconstructions/azo/`
Reconstruções relativas a amostras "azo" (fora do âmbito principal do estudo).

### `reconstructions/fiber/`
Dados de todas as fibras estudadas.

**Amostras individuais:**
- `cannabis_1g` a `cannabis_4g` — fibras de cannabis.
- `FiberA_1` a `FiberA_5` — Fibra A (celulose branqueada).
- `FiberB_0` a `FiberB_4` — Fibra B (celulose não branqueada).

**Dentro de cada pasta de fibra (ex.: `FiberB_4/`):**

- `3Dgraph/` — Visualizações 3D da fase/superfície reconstruída.
- `coolchic/` — Resultados da compressão, organizados por λ (`lmbda_1e-1` a `lmbda_1e-5`).
- `hologram/` — Ficheiros intermédios (amplitude, fase, filtragem, PNGs de entrada, parâmetros de normalização).
- `reference/` — Dados de referência (sem compressão).
- `results*.png` — Gráficos individuais de rate-distortion por domínio (Complexo, Wrapped, Unwrapped, Re/Im).
- `correlacoes_finais_*.json` — Correlações (Pearson/Spearman) entre domínios, por métrica.
- `resultados_media_*.json` — Médias das métricas por λ.
- `resultados_Wrapped_*.json` / `resultados_Unwrapped_*.json` — Resultados das métricas por domínio de fase.

**Dentro de cada `coolchic/lmbda_X/`:**

- `3D/` — Visualização 3D correspondente a este nível de compressão.
- `logs_imag/`, `logs_real/` — Logs de treino da rede neuronal (componentes real e imaginária).
- `phase/` — Fase (wrapped/unwrapped) obtida após descompressão.
- `holo_order_*_8bits_*.npy/.ppm/.cool` — Ficheiros intermédios da compressão (normalização 8-bit, comprimido, descomprimido).
- `amplitude.npy`, `phase.npy` — Amplitude e fase reconstruídas para este λ.

### Ficheiros agregados em `fiber/` (raiz)

Resultados combinando **todas as fibras**:

- `resultsGraph_*AllFibers.png` — Curvas de rate-distortion médias por métrica/domínio.
- `correlacoes_finais_psnr.json` / `_ssim.json` — Matrizes finais de correlação entre domínios.
- `resultados_media_*.json` — Médias globais por λ e métrica.
- `results.txt` — Registo de quais amostras já passaram pelo fluxo de reconstrução, e quais já foram introduzidas no fluxo de compressão.

---

## Convenções de Nomenclatura

- `lmbda_1e-N` — valor do parâmetro λ do Cool-Chic (`1e-1` a `1e-5`), controla a taxa de compressão.
- Sufixos `_psnr` / `_ssim` — métrica objetiva aplicada.
- Sufixos `Wrapped` / `Unwrapped` / `Complexo` — domínio de avaliação da fase/campo.
