
#%%
import sys

project_root = "/home/anabs/Documents/Uni/Investigação/Coolchic_DHM"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import modules.readResults as readR
import modules.pda_metric as metricP
import modules.processUtils as procUtils
import os

script_dir = os.getcwd()
base_root = os.path.dirname(script_dir)

output_dir = os.path.join(project_root, "reconstructions", "fiber", "results", "confidence_interval")
os.makedirs(output_dir, exist_ok=True)

#%%
#todos os nomes das fibras disponiveis organizados em categorias 
fibersA = ["FiberA_1", "FiberA_1_lowerpart", "FiberA_2", "FiberA_3", "FiberA_4"]
fibersB = ["FiberB_0", "FiberB_1", "FiberB_2", "FiberB_3", "FiberB_4"]
cannabis = ["cannabis_1g", "cannabis_2g", "cannabis_3g"]
#cannabis = ["cannabis_1g", "cannabis_2g", "cannabis_3g", "cannabis_4g"]
#fibersA = ["FiberA_1", "FiberA_1_lowerpart", "FiberA_2", "FiberA_3", "FiberA_4", "FiberA_5"]

#todas as fibras
allFibers = np.concatenate((fibersA, fibersB, cannabis)).tolist()

def mean_confidence_interval(data, confidence=0.95):
    #converter para floats
    a = np.array(data, dtype=float)
    n = len(a)
    mean = np.mean(a)

    #precisa de dois ou mais elementos
    if n < 2:
        return mean, mean, mean

    #calcular para a distribuiçao t-student, por ser um tamanho de amostra pequeno (13)
    sem = stats.sem(a)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem

    return mean, mean - margin, mean + margin

def summarize_metric(df, group_col, metric_col, confidence=0.95):
    rows = []
    for gval, g in df.groupby(group_col):
        mean, low, high = mean_confidence_interval(g[metric_col], confidence)
        rows.append({
            group_col: gval,
            "metric": metric_col.upper(),
            "rate": g["rate"].mean(),
            "mean": mean,
            "low": low,
            "high": high,
            "n": len(g)
        })
    out = pd.DataFrame(rows).sort_values(group_col)
    out["yerr_low"] = out["mean"] - out["low"]
    out["yerr_high"] = out["high"] - out["mean"]
    return out

def create_combined_error_graph(psnr_dfs, ssim_dfs, labels, extraInfo, outputName):
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    metrics_data = [
        ("psnr", psnr_dfs, "PSNR (dB)"),
        ("ssim", ssim_dfs, "SSIM")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for ax, (metric, dfs, ylabel) in zip(axes, metrics_data):

        for df, label, color, marker in zip(dfs, labels, colors, markers):

            summ = summarize_metric(df, "lmbda", metric)

            ax.errorbar(
                summ["rate"],
                summ["mean"],
                yerr=[
                    summ["yerr_low"],
                    summ["yerr_high"]
                ],
                fmt=marker + "-",
                color=color,
                linewidth=2,
                capsize=5,
                label=label+f"{ylabel})"
            )

        ax.set_title(metric.upper())
        ax.set_xlabel("Débito / Rate (bpp)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.figtext(
        0.5,
        0.02,
        extraInfo,
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = os.path.join(
        output_dir,
        f"resultsGraph_confidence_{outputName}.png"
    )

    plt.savefig(output_path)
    plt.show()


def create_error_graph(df_dataP, df_dataS, extraInfo, outputName):
    psnr_sum = summarize_metric(df_dataP, "lmbda", "psnr")
    ssim_sum = summarize_metric(df_dataS, "lmbda", "ssim")

    print(psnr_sum)
    print(ssim_sum)

    procUtils.save_json(
        os.path.join(output_dir, f"resultados_confidence_{outputName}_psnr.json"),
        psnr_sum.to_dict(orient='records')
    )
    procUtils.save_json(
        os.path.join(output_dir, f"resultados_confidence_{outputName}_ssim.json"),
        ssim_sum.to_dict(orient='records')
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    prefix = "PDA-" if outputName == "wrapped" else ""

    for ax, summ, title in [
        (axes[0], psnr_sum, f"{prefix}PSNR"),
        (axes[1], ssim_sum, f"{prefix}SSIM"),
    ]:
        ax.errorbar(
            summ["rate"],
            summ["mean"],
            yerr=[summ["yerr_low"], summ["yerr_high"]],
            fmt="o-",
            capsize=5,
            linewidth=2
        )
        ax.set_title(title)
        ax.set_xlabel("Débito / Rate (bpp)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Média ± IC 95%")
    plt.tight_layout()
    plt.figtext(
            0.5,           
            0.00015,          
            extraInfo,   
            ha='center',   
            fontsize=10, 
            style='italic',
            color='gray'
        )
    
    output_path = os.path.join(output_dir, f"resultsGraph_confidence_{outputName}.png")
    plt.savefig(output_path)
    plt.show()


_, df_dataP_complex = readR.getAllFibersDataComplex(allFibers, "psnr")
_, df_dataS_complex = readR.getAllFibersDataComplex(allFibers, "ssim")

_, df_dataP_wrapped = metricP.getAllFibersPhaseMetric(allFibers, True, "psnr")
_, df_dataS_wrapped = metricP.getAllFibersPhaseMetric(allFibers, True, "ssim")

_, df_dataP_unwrapped = metricP.getAllFibersPhaseMetric(allFibers, False, "psnr")
_, df_dataS_unwrapped = metricP.getAllFibersPhaseMetric(allFibers, False, "ssim")

sampleSz = len(allFibers)

fiberTypes = "FiberA, FiberB e cannabis"

create_error_graph(df_dataP_complex, df_dataS_complex, f'{sampleSz} amostras de {fiberTypes} para o Holograma Complexo', 'complex')
create_error_graph(df_dataP_wrapped, df_dataS_wrapped, f'{sampleSz} amostras de {fiberTypes} para a Fase Wrapped', 'wrapped')
create_error_graph(df_dataP_unwrapped, df_dataS_unwrapped, f'{sampleSz} amostras de {fiberTypes} para a Fase Unwrapped', 'unwrapped')

create_combined_error_graph([df_dataP_complex, df_dataP_unwrapped, df_dataP_wrapped], [df_dataS_complex,df_dataS_unwrapped, df_dataS_wrapped], ["Holograma complexo (", "Fase Unwrapped (", "Fase wrapped (PDA-"], f'{sampleSz} amostras de {fiberTypes}', "combined")