#%%

import importlib
from pyDHM import utils
importlib.reload(utils)

from pyDHM import numProp
importlib.reload(numProp)
import numpy as np

import cv2

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import modules.readResults as readR

from skimage.metrics import structural_similarity as ssim

########################################################################################################################################################################
## Implementação de métricas adaptadas à fase                                                                                                              ##
########################################################################################################################################################################

pi_val = np.pi

#garante que está entre -pi e pi
def wrap(mt):
    aux = np.mod((mt + pi_val), 2*pi_val)
    res = aux - pi_val

    return res

#shortest circular difference
#devolve a diferença mais pequena quando circular
def scd(mt1, mt2):
    res = wrap(mt1 - mt2)

    return res

#circular mean square error
def cmse(mt1, mt2):
    sq = np.square(scd(mt1, mt2))
    res = np.mean(sq)

    return res

#circular psnr
def cpsnr(mt1, mt2, bits):
    if bits == 8:
        max = 255
    elif bits == 16:
        max = 65535
    
    aux = max / cmse(mt1, mt2)
    res = 10 * np.log10(aux)

    return res

#normaliza uma distribuição de uma fase
def pda_1(ph1, c=0):
    aux = ph1 - np.mean(ph1)
    ph1_res = wrap(aux)

    aux2 = ph1_res - np.mean(ph1_res)
    ph1_res2 = wrap(aux2)

    cond = np.mean(np.abs(ph1_res2))
    if cond > (pi_val/2) :
        res = wrap(ph1_res2 + c + pi_val)
    else:
        res = wrap(ph1_res2 + c)

    return res

#faz o shift do 2(comprimido) para o 1(original)
def pda_2(ph1, ph2):
    ph1_norm = pda_1(ph1)
    ph2_norm = pda_1(ph2)
    aux = ph2_norm + np.mean(scd(ph1_norm, ph2_norm))

    #este res é o resultado final do ph2
    res = wrap(aux)
    return res, ph1_norm

def phase_psnr(ph1, ph2, is_wrapped=True):
    ph1 = ph1.astype(np.float64)
    ph2 = ph2.astype(np.float64)
    
    if is_wrapped:
        data_range = 2 * np.pi 
    else:
        data_range = ph1.max() - ph1.min()

    mse = np.mean((ph1 - ph2) ** 2)
    psnr = 10 * np.log10((data_range **2) / mse)
    
    return psnr

# ph1 -> fase wrapped antes de comprimir (depois do pda_1)
# ph2 -> fase wrapped depois de comprimir (depois do pda_2)
def pda_psnr(ph1, ph2, isWrapped=True):
    ph2_norm, ph1_norm = pda_2(ph1, ph2)

    #separar cada uma em real e imaginário
    ph1_real = np.cos(ph1_norm)
    ph1_imag = np.sin(ph1_norm)
    
    ph2_real = np.cos(ph2_norm)
    ph2_imag = np.sin(ph2_norm)

    #calcular a métrica no conjunto dos reais e no dos imaginarios 
    mt_real = phase_psnr(ph1_real, ph2_real, isWrapped) # ph1_real e ph2_real
    mt_imag = phase_psnr(ph1_imag, ph2_imag, isWrapped) # ph1_imag e ph2_imag

    #fazer a média dos resultados
    res = np.sqrt((mt_real**2 + mt_imag**2) / 2)
    print("PSNR WRAPPED PHASE - " + str(res))
    return res

def pda_ssim(ph1, ph2, isWrapped):
    ph2_norm, ph1_norm = pda_2(ph1, ph2)

    if isWrapped:
        ph1_real = np.cos(ph1_norm)
        ph1_imag = np.sin(ph1_norm)
        
        ph2_real = np.cos(ph2_norm)
        ph2_imag = np.sin(ph2_norm)

  
        mt_real = ssim(ph1_real, ph2_real, data_range=2.0, full=False) 
        mt_imag = ssim(ph1_imag, ph2_imag, data_range=2.0, full=False)

        res = (mt_real + mt_imag) / 2.0
        print("SSIM WRAPPED PHASE - " + str(res))
        
    else:
        dynamic_range = ph1_norm.max() - ph1_norm.min()
        
        res = ssim(ph1_norm, ph2_norm, data_range=dynamic_range, full=False)
        print("SSIM UNWRAPPED PHASE - " + str(res))

    return res

########################################################################################################################################################################
## Calcular valores das métricas e criar o gráfico para as métricas da fase                                                                                           ##
########################################################################################################################################################################

def get_metric_fiber(fiber_name, outputFilename, isPhaseWrapped, metric='psnr'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(script_dir) 
    prefix = re.split(r"[_.]+", fiber_name)[0]
    compressed_dir = os.path.join(base_root, "reconstructions", "fiber", f"{prefix}_glicerina", fiber_name, "coolchic")
    reference_dir = os.path.join(base_root, "reconstructions", "fiber", f"{prefix}_glicerina", fiber_name, "reference")

    data = {'rate': [], metric: [], 'lmbda': []}
    df_complex = readR.getStatsCCComplex(compressed_dir)

    lambda_dirs = glob.glob(os.path.join(compressed_dir, 'lmbda_*'))

    for l_dir in lambda_dirs:
        path_cp = os.path.join(l_dir, 'phase', outputFilename + '.npy')
        path_rf = os.path.join(reference_dir, outputFilename + '.npy')
            
        phase_cp = np.load(path_cp)
        phase_rf = np.load(path_rf)

        if metric == 'psnr':
            if isPhaseWrapped:
                val = pda_psnr(phase_rf, phase_cp, isPhaseWrapped)
            else:
                val = phase_psnr(phase_rf, phase_cp, False)
        elif metric == 'ssim':
            val = pda_ssim(phase_rf, phase_cp, isPhaseWrapped)
        else:
            raise ValueError(f"metric desconhecido: {metric}")
        
        pathSplit = re.split(r"[_.]+", l_dir)
        print(pathSplit)
        lambda_val = pathSplit[-1]

        lin = df_complex[df_complex['lmbda'] == float(lambda_val)]
        bpp = lin['rate'].iloc[0]

        data[metric].append(val)
        data['lmbda'].append(float(lambda_val))
        data['rate'].append(bpp)

    df_data = pd.DataFrame(data).sort_values(by='rate')
    return df_data

def getAllFibersPhaseMetric(allFibers, isPhaseWrapped, metric='psnr'):
    if isPhaseWrapped:
        outputFilename = "phaseWrapped"
    else:
        outputFilename = "phaseUnwrapped"

    dfs_list = []

    for fiber in allFibers:
        df_complex = get_metric_fiber(fiber, outputFilename, isPhaseWrapped, metric)

        if not df_complex.empty:
            dfs_list.append(df_complex)

    if dfs_list:
        df_total = pd.concat(dfs_list, ignore_index=True)
    else:
        print("Nenhum dado encontrado para gerar o gráfico.")
        return 

    df_meanTotal = df_total.groupby('lmbda').mean().reset_index()

    return df_meanTotal, df_total

########################################################################################################################################################################
## Aplicar métricas de correlação e salvar os resultados                                                                                                              ##
########################################################################################################################################################################

def applyCorrelationMetrics(base_dir=None):
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_root = os.path.dirname(script_dir) 
        base_dir = os.path.join(base_root, "reconstructions", "fiber", "results", "numbers_of_results")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "metrics_correlation")

    for metric in ['psnr', 'ssim']:
        path_complex = os.path.join(base_dir, f"resultados_media_{metric}.json")
        path_wrapped = os.path.join(base_dir, f"resultados_Wrapped_{metric}.json")
        path_unwrapped = os.path.join(base_dir, f"resultados_Unwrapped_{metric}.json")
        
        df_complex = loadAllMetricsJson(path_complex, metric, "Complexo")
        df_wrapped = loadAllMetricsJson(path_wrapped, metric, "Wrapped")
        df_unwrapped = loadAllMetricsJson(path_unwrapped, metric, "Unwrapped")
        
        dfs_val = [df for df in [df_complex, df_wrapped, df_unwrapped] if df is not None]
        
        if len(dfs_val) < 2:
            print("Não há dados suficientes para cruzar métricas. Verifique os ficheiros JSON.")
            continue
            
        #alinhar pelo valor lambda
        df_final = pd.concat(dfs_val, axis=1, join='inner')
        
        matrix_pearson = df_final.corr(method='pearson')
        matrix_spearman = df_final.corr(method='spearman')
        
        correlation_data = {
            "pearson": matrix_pearson.to_dict(),
            "spearman": matrix_spearman.to_dict()
        }
        
        correlationPath = os.path.join(output_dir, f"correlacoes_finais_{metric}.json")
                
        readR.save_json(correlationPath, correlation_data)

def loadAllMetricsJson(filepath, metric, element):
    df = pd.read_json(filepath)
    
    #garantir que os lambda estao todos no mesmo formato
    df['lmbda'] = df['lmbda'].astype(float)
    df['lmbda'] = df['lmbda'].round(10)
    
    df = df.set_index('lmbda')
    
    #só queremos as métrica, nao precisamos do rate
    if 'rate' in df.columns:
        df = df.drop(columns=['rate'])
        
    df.columns = [f"{col}_{metric}_{element}" for col in df.columns]
    return df

    