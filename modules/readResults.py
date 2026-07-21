
#%% ****************************************************
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

colors = ['blue', 'red', 'green', 'orange', 'purple']
marker = ['o', 's', '>', '<']
linestyle = ['-']
annotDist = [(0, 8), (0, -15), (-15, 0), (15, 0)]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_root = os.path.dirname(script_dir) 

metricList = ['psnr', 'ssim']

#metric = psnr or ssim  
def constructGraph(fiber_name, metric, dfList, labelList, title, outputName, isAll=False):
    #caminho absoluto para os dados e output
    if isAll:
        output_dir = os.path.join(base_root, "reconstructions", "fiber")
    else:
        output_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name)

    if metric == 'psnr':
        quality = 'PSNR (dB)'
    elif metric == 'ssim':
        quality = 'SSIM'

    plt.figure(figsize=(10, 6))

    for i in range(0, len(dfList)):
        plt.plot(dfList[i]['rate'], dfList[i][metric], marker=marker[i], linestyle=linestyle[i], color=colors[i], label=labelList[i])
    
    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel(f'Qualidade / {quality}', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i in range(0, len(dfList)):
        for _, row in dfList[i].iterrows():
            plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row[metric]), 
                        textcoords="offset points", xytext=annotDist[i], ha='center', fontsize=8, color=colors[i])
            
    plt.tight_layout()
    output_path = os.path.join(output_dir, outputName)
    plt.savefig(output_path)
    plt.show()

#ainda não tem forma de fazer o ssim, porque está se a usar o psnr calculado do cool chic
def getRDGraphsReIm(fiberList, metrics=['psnr']):
    for fiberOpt in fiberList:
        dfList = []
        labelList = []

        base_dir = os.path.join(base_root, "reconstructions", "fiber", fiberOpt, "coolchic")
        df_real, df_imag = getStatsCoolChic(base_dir)

        dfList.append(df_real)
        dfList.append(df_imag)
        labelList.append('Componente Real')
        labelList.append('Componente Imaginária')

        for metricOpt in metrics:
            constructGraph(fiberOpt, metricOpt, dfList, labelList, f'Comparação ({metricOpt}) : Matriz Real vs Imaginária', f"resultsGraphReIm_{metricOpt}.png")

def getRDGraphComplex(fiberList, metrics=metricList):
    labelList = ['Holograma Complexo']

    for fiberOpt in fiberList:
        base_dir = os.path.join(base_root, "reconstructions", "fiber", fiberOpt, "coolchic")

        for metricOpt in metrics:
            dfList = []

            if metricOpt == 'psnr':
                df_complex = getStatsCCComplex(base_dir)
            else:
                hologram_dir = os.path.join(base_root, "reconstructions", "fiber", fiberOpt, "hologram")
                df_complex = getStatsCCComplexSSIM(base_dir, hologram_dir)
    
            dfList.append(df_complex)

            constructGraph(fiberOpt, metricOpt, dfList, labelList, f'Curva Rate-Distortion {metricOpt}: Holograma Complexo', f"resultsGraphComplex_{metricOpt}.png")

def getRDGraphPhase(fiberList, df_data, phaseType,  metrics, outputFilename, isAll=False):    
    labelList = ['Fase ' + phaseType]

    for fiberOpt in fiberList:
        for metricOpt in metrics:
            dfList = []
            dfList.append(df_data)

            constructGraph(fiberOpt, metricOpt, dfList, labelList, f'Avaliação da fidelidade da fase {phaseType} em função da taxa de compressão ({metricOpt})', f"resultsGraph_{metricOpt}_{outputFilename}.png", isAll)

def getAllFibersDataComplex(allFibers, metric='psnr'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir)

    dfs_list = []
    for fiber in allFibers:
        base_dir = os.path.join(base_root, "reconstructions", "fiber", fiber, "coolchic")

        if metric == 'psnr':
            df_complex = getStatsCCComplex(base_dir)
        elif metric == 'ssim':
            hologram_dir = os.path.join(base_root, "reconstructions", "fiber", fiber, "hologram")
            df_complex = getStatsCCComplexSSIM(base_dir, hologram_dir)
        else:
            raise ValueError(f"Métrica desconhecida: {metric}")

        if not df_complex.empty:
            dfs_list.append(df_complex)

    df_total = pd.concat(dfs_list, ignore_index=True)
    df_total['lmbda'] = df_total['lmbda'].round(10)
    df_mean = df_total.groupby('lmbda').mean().reset_index()

    return df_mean

def getRDGraphComplexAll(allFibers, metrics=metricList):
    output_dir = os.path.join(base_root, "reconstructions", "fiber")

    for metric in metrics:
        df_mediaTotal = getAllFibersDataComplex(allFibers, metric)

        #guardar os dados para avaliar mais facilmente depois
        save_path = os.path.join(output_dir, f"resultados_media_{metric}.json")
        dados_json = df_mediaTotal.to_dict(orient='records')
        save_json(save_path, dados_json)

        dfList = [df_mediaTotal]
        labelList = ['Holograma Complexo']
        constructGraph("", metric, dfList, labelList, f'Curva Rate-Distortion {metric}: Holograma Complexo Todas As Amostras', f"resultsGraphComplex_{metric}_AllFibers.png", True)

def getStatsCoolChic(base_dir):
    data_real = {'rate': [], 'psnr': [], 'lmbda': []}
    data_imag = {'rate': [], 'psnr': [], 'lmbda': []}

    lambda_dirs = glob.glob(os.path.join(base_dir, 'lmbda_*'))

    print(f"Encontradas {len(lambda_dirs)} {lambda_dirs} pastas de lmbda. A extrair dados...")

    print(base_dir)

    for l_dir in lambda_dirs:
        path_real = os.path.join(l_dir, 'logs_real', '0000-results_encoder.tsv')
        if os.path.exists(path_real):
            df_r = pd.read_csv(path_real, sep=r'\s+', engine='python')
            data_real['rate'].append(df_r['rate_bpp'].iloc[0])
            data_real['psnr'].append(df_r['psnr_db'].iloc[0])
            data_real['lmbda'].append(df_r['lmbda'].iloc[0])
            
        path_imag = os.path.join(l_dir, 'logs_imag', '0000-results_encoder.tsv')
        if os.path.exists(path_imag):
            df_i = pd.read_csv(path_imag, sep=r'\s+', engine='python')
            data_imag['rate'].append(df_i['rate_bpp'].iloc[0])
            data_imag['psnr'].append(df_i['psnr_db'].iloc[0])
            data_imag['lmbda'].append(df_i['lmbda'].iloc[0])

    df_real = pd.DataFrame(data_real).sort_values(by='rate')
    df_imag = pd.DataFrame(data_imag).sort_values(by='rate')

    return df_real, df_imag

def getStatsCCComplex(base_dir):
    df_r, df_i = getStatsCoolChic(base_dir)
    data_complex = {'rate': [], 'psnr': [], 'lmbda': []}
    MAX_VAL = 255

    for i in range(0,5):
        total_bpp = df_r['rate'].iloc[i] + df_i['rate'].iloc[i]

        mse_real = (MAX_VAL**2) / (10**(df_r['psnr'].iloc[i] / 10))
        mse_imag = (MAX_VAL**2) / (10**(df_i['psnr'].iloc[i] / 10))
        
        mse_complex = mse_real + mse_imag
        psnr_complex = 10 * np.log10((MAX_VAL**2) / mse_complex)
        
        data_complex['rate'].append(total_bpp)
        data_complex['psnr'].append(psnr_complex)
        data_complex['lmbda'].append(df_r['lmbda'].iloc[i])

    df_complex = pd.DataFrame(data_complex).sort_values(by='rate')
    
    return df_complex

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ficheiro {path} não encontrado. Execute o passo de referencia primeiro!"
        )
    with open(path, 'r') as f:
        return json.load(f)

def get_complex_ssim_quantized(orig_real, orig_imag, recon_real, recon_imag):
    '''
    computes a single SSIM score for the complex hologram by treating
    real and imaginary parts as two channels of one joint signal.
    assumes both original and reconstructed are on the same 8-bit quantized scale.
    '''
    orig_stack = np.stack([orig_real, orig_imag], axis=-1).astype(np.float64)
    recon_stack = np.stack([recon_real, recon_imag], axis=-1).astype(np.float64)

    data_range = 255  
    return ssim(orig_stack, recon_stack, data_range=data_range, channel_axis=-1)

def load_decoded_quantized(l_dir, filename="holo_order", bits=8):
    path_r = os.path.join(l_dir, f"{filename}_real_{bits}bits_decomp.ppm")
    path_i = os.path.join(l_dir, f"{filename}_imag_{bits}bits_decomp.ppm")

    img_real = cv2.imread(path_r, cv2.IMREAD_UNCHANGED)
    img_imag = cv2.imread(path_i, cv2.IMREAD_UNCHANGED)

    if len(img_real.shape) == 3:
        img_real = img_real[:, :, 0]
    if len(img_imag.shape) == 3:
        img_imag = img_imag[:, :, 0]

    return img_real, img_imag

def getStatsCCComplexSSIM(base_dir, hologram_dir):
    df_r, df_i = getStatsCoolChic(base_dir)  

    orig_real = cv2.imread(os.path.join(hologram_dir, "holo_order_real_8bits.png"), cv2.IMREAD_UNCHANGED)
    orig_imag = cv2.imread(os.path.join(hologram_dir, "holo_order_imag_8bits.png"), cv2.IMREAD_UNCHANGED)

    data = {'rate': [], 'ssim': [], 'lmbda': []}
    lambda_dirs = glob.glob(os.path.join(base_dir, 'lmbda_*'))

    for l_dir in lambda_dirs:
        recon_real_path = os.path.join(l_dir, "holo_order_real_8bits_decomp.npy")
        recon_imag_path = os.path.join(l_dir, "holo_order_imag_8bits_decomp.npy")
        if not (os.path.exists(recon_real_path) and os.path.exists(recon_imag_path)):
            continue

        folder_lmbda_str = os.path.basename(l_dir).replace("lmbda_", "")
        folder_lmbda_val = float(folder_lmbda_str)

        recon_real, recon_imag = load_decoded_quantized(l_dir)
        ssim_val = get_complex_ssim_quantized(orig_real, orig_imag, recon_real, recon_imag)

        match_r = df_r[np.isclose(df_r['lmbda'].astype(float), folder_lmbda_val)]
        match_i = df_i[np.isclose(df_i['lmbda'].astype(float), folder_lmbda_val)]

        if match_r.empty or match_i.empty:
            print(f"Aviso: não foi possível encontrar rate para {l_dir} (lmbda={folder_lmbda_val})")
            continue

        total_bpp = match_r['rate'].iloc[0] + match_i['rate'].iloc[0]

        data['rate'].append(total_bpp)
        data['ssim'].append(ssim_val)
        data['lmbda'].append(folder_lmbda_val)

    return pd.DataFrame(data).sort_values(by='rate')

#fazer os plots para o poster
def getRDGraphCombined(fiber_name, df_holo, df_phase, output_dir, metrics=metricList, isAll=False):
    for metric in metrics:
        if metric == 'psnr':
            metricName = 'PSNR'
        else:
            metricName = 'SSIM'

        dfList = []
        labelList = []

        dfList.append(df_holo)
        dfList.append(df_phase)

        labelList.append('Holograma')
        labelList.append('Fase Unwrapped')

        constructGraph(fiber_name, metric, dfList, labelList, f'Curva Rate-Distortion {metricName}: Holograma vs Fase Unwrapped', f"resultsGraph_{metricName}_combined.png", True)

def getRDGraphPDACombined(fiber_name, df_pda_psnr, df_pda_ssim, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'green'
    ax1.set_xlabel('Débito / Rate (bpp)', fontsize=12)
    ax1.set_ylabel('PDA-PSNR (dB)', color=color1, fontsize=12)
    ax1.plot(df_pda_psnr['rate'], df_pda_psnr['psnr'], marker='o', linestyle='-', color=color1, label='PDA-PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'purple'
    ax2.set_ylabel('PDA-SSIM', color=color2, fontsize=12)
    ax2.plot(df_pda_ssim['rate'], df_pda_ssim['ssim'], marker='s', linestyle='--', color=color2, label='PDA-SSIM')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)

    ax1.set_title('Fidelidade da fase wrapped: PDA-PSNR e PDA-SSIM', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "resultsGraph_PDA_combined.png")
    fig.savefig(output_path)
    plt.show()
