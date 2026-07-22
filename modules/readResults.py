
#%% ****************************************************
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import modules.processUtils as procUtils
import re
import modules.pda_metric as metricsP

colors = ['blue', 'red', 'green', 'orange', 'purple']
marker = ['o', 's', '>', '<', '^']
linestyle = ['-']
annotDist = [(0, 8), (0, -15), (-15, 0), (8, 0), (15, 0)]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_root = os.path.dirname(script_dir) 

metricList = ['psnr', 'ssim']

mainFolder = "Fibras_glicerina"
PATH_MAP = {
    "FiberA": "FiberA_glicerina/",
    "FiberB": "FiberB_glicerina/",
    "cannabis": "cannabis_glicerina/"
}


########################################################################################################################################################################
## Função genérica para a construção de gráficos                                                                                                                      ##
########################################################################################################################################################################

#metric = psnr or ssim  
def constructGraph(fiber_name, metric, dfList, labelList, title, outputName, isAll=False, isCategory=False):
    '''
    Constructs a graph where one axis is a metric, and the other is the rate (bpp)
 
    Parameters
    ----
    fiber_name : str
        name of the fiber, to save the resulting files
    metric : str
        which metric was applied
    dfList : list
        list of Dataframes, where each dataframe contains the information that is correspondent to one line in the graph \n
        ex - the first dataframe is the real component information whilst the second one is the imaginary component information
        (this was made so that can only be at most five lines in one graph)
    labelList : list 
        list of strings that are the labels each line should have, needs to be in the same order as the dataframes in the list
    title : str
        title for the graph
    outputName : str
        name of the output file with the graph
    isAll : boolean
        says if its a graph with just a single sample or more than one in one line,
        impacts mainly the output folder \n
        if False then its a single sample, so it will be saved in the specific fiber folder
        else it will be saved in the overal fiber folder 
    '''
    path = ""
    info = ""

    if not isAll and not isCategory:
        prefix = re.split(r"[_.]+", fiber_name)[0]
        path = PATH_MAP.get(prefix, "")
        info = fiber_name
    else:
        if isCategory:
            prefix = re.split(r"[_.]+", fiber_name[0])[0]
            path = PATH_MAP.get(prefix, "")
            info = f"Todas as amostras do tipo {prefix}"
        elif isAll:
            info = "Todas as amostras"

    extraInfo = f"Pasta: {mainFolder}/{path}, Amostra: {info}"

    #caminho absoluto para os dados e output
    if isAll:
        output_dir = os.path.join(base_root, "reconstructions", "fiber")
    elif isCategory:
        output_dir = os.path.join(base_root, "reconstructions", "fiber", prefix+"_glicerina")
        print("CATEGORY --- " + prefix+"_glicerina")
    else:
        output_dir = os.path.join(base_root, "reconstructions", "fiber", prefix+"_glicerina", fiber_name)

    if metric == 'psnr':
        quality = 'PSNR (dB)'
    elif metric == 'ssim':
        quality = 'SSIM'

    plt.figure(figsize=(10, 6))

    for i in range(0, len(dfList)):
        plt.plot(dfList[i]['rate'], dfList[i][metric], marker=marker[i], linestyle=linestyle[0], color=colors[i], label=labelList[i])
    
    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel(f'Qualidade / {quality}', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i in range(0, len(dfList)):
        for _, row in dfList[i].iterrows():
            plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row[metric]), 
                        textcoords="offset points", xytext=annotDist[i], ha='center', fontsize=8, color=colors[i])
                
    plt.figtext(
        0.5,           
        0.035,          
        extraInfo,   
        ha='center',   
        fontsize=10, 
        style='italic',
        color='gray'
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = os.path.join(output_dir, outputName)
    plt.savefig(output_path)
    plt.show()


########################################################################################################################################################################
## Funções que criam e guardam como ficheiros os diferentes gráficos necessário                                                                                       ##
########################################################################################################################################################################

#ainda não tem forma de fazer o ssim, porque está se a usar o psnr calculado do cool chic
def getRDGraphsReIm(fiberList, metrics=['psnr']):
    for fiberOpt in fiberList:
        dfList = []
        labelList = []

        prefix = re.split(r"[_.]+", fiberOpt)[0]
        base_dir = os.path.join(base_root, "reconstructions", "fiber", f"{prefix}_glicerina", fiberOpt, "coolchic")
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
        prefix = re.split(r"[_.]+", fiberOpt)[0]
        base_dir = os.path.join(base_root, "reconstructions", "fiber", f"{prefix}_glicerina", fiberOpt, "coolchic")

        for metricOpt in metrics:
            dfList = []

            if metricOpt == 'psnr':
                df_complex = getStatsCCComplex(base_dir)
            else:
                prefix = re.split(r"[_.]+", fiberOpt)[0]
                hologram_dir = os.path.join(base_root, "reconstructions", "fiber", f"{prefix}_glicerina", fiberOpt, "hologram")
                df_complex = getStatsCCComplexSSIM(base_dir, hologram_dir)
    
            dfList.append(df_complex)

            constructGraph(fiberOpt, metricOpt, dfList, labelList, f'Curva Rate-Distortion {metricOpt}: Holograma Complexo', f"resultsGraphComplex_{metricOpt}.png")

def getRDGraphPhase(fiberList, phaseType, metrics, outputFilename, isAll=False, isCategory=False, isPhaseWrapped=False):    
    if isPhaseWrapped:
        outputFilename = "phaseWrapped"
        phaseType = "Wrapped"
    else:
        outputFilename = "phaseUnwrapped"
        phaseType = "Unwrapped"

    labelList = ['Fase ' + phaseType]

    for fiberOpt in fiberList:
        for metricOpt in metrics:
            dfList = []
            df_data = metricsP.get_metric_fiber(fiberOpt, outputFilename, isPhaseWrapped, metricOpt)

            dfList.append(df_data)

            constructGraph(fiberOpt, metricOpt, dfList, labelList, f'Avaliação da fidelidade da fase {phaseType} em função da taxa de compressão ({metricOpt})', f"resultsGraph_{metricOpt}_{outputFilename}.png", isAll=isAll, isCategory=isCategory)

def getRDGraphPhaseAll(fiberList, phaseType, metrics, outputFilename, isAll=True, isCategory=False, isPhaseWrapped=False):    
    for metricOpt in metrics:
        dfList = []
        
        if isPhaseWrapped:
            outputFilename = "phaseWrapped"
            phaseType = "Wrapped"
        else:
            outputFilename = "phaseUnwrapped"
            phaseType = "Unwrapped"

        labelList = ['Fase ' + phaseType]

        df_data = metricsP.getAllFibersPhaseMetric(fiberList, isPhaseWrapped, metricOpt)

        base_dir = os.path.join(base_root, "reconstructions", "fiber")
            
        nome_ficheiro = f"resultados_{phaseType}_{metricOpt}.json"
        save_path = os.path.join(base_dir, nome_ficheiro)
            
        dados_json = df_data.to_dict(orient='records')
        procUtils.save_json(save_path, dados_json)

        dfList.append(df_data)

        if isCategory:
            prefix = re.split(r"[_.]+", fiberList[0])[0]
            outputName =  f"resultsGraph_{metricOpt}_{outputFilename}_All{prefix}.png"
        else:
            outputName = f"resultsGraph_{metricOpt}_{outputFilename}_AllFibers.png"

        constructGraph(fiberList, metricOpt, dfList, labelList, f'Avaliação da fidelidade da fase {phaseType} em função da taxa de compressão ({metricOpt})', outputName, isAll=isAll, isCategory=isCategory)


def getRDGraphComplexAll(allFibers, metrics=metricList, isCategory=False, isAll=True):
    output_dir = os.path.join(base_root, "reconstructions", "fiber")

    for metric in metrics:
        df_mediaTotal = getAllFibersDataComplex(allFibers, metric)

        #guardar os dados para avaliar mais facilmente depois
        save_path = os.path.join(output_dir, f"resultados_media_{metric}.json")
        dados_json = df_mediaTotal.to_dict(orient='records')
        procUtils.save_json(save_path, dados_json)

        dfList = [df_mediaTotal]
        labelList = ['Holograma Complexo']

        if isCategory:
            prefix = re.split(r"[_.]+", allFibers[0])[0]
            outputName =  f"resultsGraphComplex_{metric}_All{prefix}.png"
        else:
            outputName = f"resultsGraphComplex_{metric}_AllFibers.png"

        constructGraph(allFibers, metric, dfList, labelList, f'Curva Rate-Distortion {metric}: Holograma Complexo Todas As Amostras', outputName, isAll=isAll, isCategory=isCategory)

#fazer os plots para o poster
def getRDGraphCombined(df_holo, df_phase, df_pda, metrics=metricList, isAll=False):
    for metric in metrics:
        if metric == 'psnr':
            metricName = 'PSNR'
            pdaMetric = 'PDA-PSNR'
        else:
            metricName = 'SSIM'
            pdaMetric = 'PDA-SSIM'

        dfList = []
        labelList = []

        dfList.append(df_holo)
        dfList.append(df_phase)
        dfList.append(df_pda)

        labelList.append(f'Holograma ({metricName})')
        labelList.append(f'Fase Unwrapped ({metricName})')
        labelList.append(f'Fase Wrapped ({pdaMetric})')

        constructGraph("", metric, dfList, labelList, f'Curva Rate-Distortion {metricName}', f"resultsGraph_{metricName}_combined.png", isAll=isAll)

#faz um gráfico com as das métricas aplicadas à fase wrapped
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

########################################################################################################################################################################
## Funções que vão buscar ou calcular as informações das métricas, juntando tudo em Dataframes                                                                        ##
########################################################################################################################################################################

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

def getAllFibersDataComplex(allFibers, metric='psnr'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir)

    dfs_list = []
    for fiber in allFibers:
        prefix = re.split(r"[_.]+", fiber)[0]
        base_dir = os.path.join(base_root, "reconstructions", "fiber", prefix+"_glicerina", fiber, "coolchic")

        if metric == 'psnr':
            df_complex = getStatsCCComplex(base_dir)
        elif metric == 'ssim':
            hologram_dir = os.path.join(base_root, "reconstructions", "fiber", prefix+"_glicerina", fiber, "hologram")
            df_complex = getStatsCCComplexSSIM(base_dir, hologram_dir)
        else:
            raise ValueError(f"Métrica desconhecida: {metric}")

        if not df_complex.empty:
            dfs_list.append(df_complex)

    df_total = pd.concat(dfs_list, ignore_index=True)
    df_total['lmbda'] = df_total['lmbda'].round(10)
    df_mean = df_total.groupby('lmbda').mean().reset_index()

    return df_mean