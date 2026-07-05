
#%% ****************************************************
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def getRDGraphReIm(fiber_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir) 
    base_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name, "coolchic")
    output_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name)

    df_real, df_imag = getStatsCoolChic(base_dir)

    plt.figure(figsize=(10, 6))

    plt.plot(df_real['rate'], df_real['psnr'], marker='o', linestyle='-', color='blue', label='Componente Real')
    plt.plot(df_imag['rate'], df_imag['psnr'], marker='s', linestyle='-', color='red', label='Componente Imaginária')

    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel('Qualidade / PSNR (dB)', fontsize=12)
    plt.title('Comparação de Compressão: Matriz Real vs Imaginária', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, row in df_real.iterrows():
        plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row['psnr']), 
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='blue')
        
    for i, row in df_imag.iterrows():
        plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row['psnr']), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "resultsGraphReIm.png")
    plt.savefig(output_path)
    plt.show()

def getRDGraphComplex(fiber_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir) 
    base_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name, "coolchic")
    output_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name)

    df_complex = getStatsCCComplex(base_dir)

    plt.figure(figsize=(10, 6))

    plt.plot(df_complex['rate'], df_complex['psnr'], marker='o', linestyle='-', color='blue', label='Holograma Complexo')

    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel('Qualidade / PSNR (dB)', fontsize=12)
    plt.title('Curva Rate-Distortion: Holograma Complexo', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, row in df_complex.iterrows():
        plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row['psnr']), 
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='blue')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "resultsGraphComplex.png")
    plt.savefig(output_path)
    plt.show()


def getRDGraphPhase(fiber_name, file_for_graph, phaseType,  metric, outputFilename, isAll=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir) 
    if isAll:
        output_dir = os.path.join(base_root, "reconstructions", "fiber")
    else:    
        output_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name)

    df_data = file_for_graph

    plt.figure(figsize=(10, 6))

    plt.plot(df_data['rate'], df_data['psnr'], marker='o', linestyle='-', color='blue', label='Fase ' + phaseType)

    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    unit = " (dB)" if "SSIM" not in metric.upper() else ""
    plt.ylabel('Qualidade / ' + metric + unit, fontsize=12)
    plt.title('Avaliação da fidelidade da fase ' + phaseType + ' em função da taxa de compressão', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, row in df_data.iterrows():
        plt.annotate(f"λ={row['lmbda']}", (row['rate'], row['psnr']), 
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='blue')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "resultsGraph_" + metric + outputFilename + ".png")
    plt.savefig(output_path)
    plt.show()

def getAllFibersGraphComplexMetric(allFibers, metric='psnr'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir)
    output_dir = os.path.join(base_root, "reconstructions", "fiber")

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

    if dfs_list:
        df_total = pd.concat(dfs_list, ignore_index=True)
    else:
        print("Nenhum dado encontrado para gerar o gráfico.")
        return

    df_total['lmbda'] = df_total['lmbda'].round(10)
    df_mediaTotal = df_total.groupby('lmbda').mean().reset_index()

    #guardar os dados para avaliar mais facilmente depois
    save_path = os.path.join(output_dir, f"resultados_media_{metric}.json")
    dados_json = df_mediaTotal.to_dict(orient='records')
    save_json(save_path, dados_json)

    y_col = metric  
    y_label = 'PSNR (dB)' if metric == 'psnr' else 'SSIM'
    title_metric = metric.upper()

    plt.figure(figsize=(10, 6))
    plt.plot(df_mediaTotal['rate'], df_mediaTotal[y_col], marker='o', linestyle='-', color='blue', label='Holograma Complexo')

    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel('Qualidade / ' + y_label, fontsize=12)
    plt.title(f'Curva Rate-Distortion: Holograma Complexo De Todas as Fibras ({title_metric})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, row in df_mediaTotal.iterrows():
        plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row[y_col]),
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='blue')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"resultsGraphComplex{title_metric}AllFibers.png")
    plt.savefig(output_path)
    plt.show()

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

def getRDGraphComplexSSIM(fiber_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir)
    base_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name, "coolchic")
    hologram_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name, "hologram")
    output_dir = os.path.join(base_root, "reconstructions", "fiber", fiber_name)

    df_complex_ssim = getStatsCCComplexSSIM(base_dir, hologram_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(df_complex_ssim['rate'], df_complex_ssim['ssim'], marker='o', linestyle='-', color='blue', label='Holograma Complexo')

    plt.xlabel('Débito / Rate (bpp)', fontsize=12)
    plt.ylabel('Qualidade / SSIM', fontsize=12)
    plt.title('Curva Rate-Distortion (SSIM): Holograma Complexo', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, row in df_complex_ssim.iterrows():
        plt.annotate(f"λ={row['lmbda']:.1e}", (row['rate'], row['ssim']),
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='blue')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "resultsGraphComplexSSIM.png")
    plt.savefig(output_path)
    plt.show()

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