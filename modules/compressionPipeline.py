import importlib
from pyDHM import utils
importlib.reload(utils)

from pyDHM import numProp
importlib.reload(numProp)
import numpy as np

import numpy as np
import cv2

import os
import subprocess
import re

# criar a estrutura de pastas para depois guardar os ficheiros necessários
def create_fiber_structure(fiber_name, lambdas=["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]):
    prefix = re.split(r"[_.]+", fiber_name)[0]
    
    first_base = f"reconstructions/fiber/"

    base = f"{first_base}/{prefix}_glicerina/{fiber_name}"
    
    folders = [
        f"{base}/hologram",
        f"{base}/reference",
        f"{base}/3Dgraph",
        *[f"{base}/coolchic/lmbda_{l}" for l in lambdas],
        *[f"{base}/coolchic/lmbda_{l}/logs_real" for l in lambdas],
        *[f"{base}/coolchic/lmbda_{l}/logs_imag" for l in lambdas],
        *[f"{base}/coolchic/lmbda_{l}/phase" for l in lambdas],
        *[f"{base}/coolchic/lmbda_{l}/3D" for l in lambdas],
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print(f"Estrutura criada para {fiber_name}")

def compress_hologram(fiber_name, lmbda, bits, iterations=2000, filename="holo"):
    prefix = re.split(r"[_.]+", fiber_name)[0]
    base = os.path.abspath(f"reconstructions/fiber/{prefix}_glicerina/{fiber_name}")
    ##### REAL
    input_path  = f"{base}/hologram/{filename}_real_{str(bits)}bits.png"
    output_path = f"{base}/coolchic/lmbda_{lmbda}/{filename}_real_{bits}bits_comp.cool"
    workdir_real = f"{base}/coolchic/lmbda_{lmbda}/logs_real"
    workdir_imag = f"{base}/coolchic/lmbda_{lmbda}/logs_imag"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""} 

    #remover os ficheiros já treinados para treinar de novo
    #for f in Path(workdir_real).glob("*.pt"):
    #    f.unlink()

    subprocess.run([
        "python", "Cool-Chic/cc_encode.py",
        "-i", input_path,
        "-o", output_path,
        "--lmbda", str(lmbda),
        "--n_itr", str(iterations),
        "--workdir", workdir_real 
    ], check=True, env=env)

    ##### IMAGINARIO
    input_path  = f"{base}/hologram/{filename}_imag_{bits}bits.png"
    output_path = f"{base}/coolchic/lmbda_{lmbda}/{filename}_imag_{bits}bits_comp.cool"
    
    #for f in Path(workdir_imag).glob("*.pt"):
    #    f.unlink()

    subprocess.run([
        "python", "Cool-Chic/cc_encode.py",
        "-i", input_path,
        "-o", output_path,
        "--lmbda", str(lmbda),
        "--n_itr", str(iterations),
        "--workdir", workdir_imag
    ], check=True, env=env)

def decompress_hologram(fiber_name, lmbda, bits, filename="holo"):
    prefix = re.split(r"[_.]+", fiber_name)[0]
    base = f"reconstructions/fiber/{prefix}_glicerina/{fiber_name}"
    ##### REAL
    input_path  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_real_{bits}bits_comp.cool"
    output_ppm  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_real_{bits}bits_decomp.ppm"
    
    subprocess.run([
        "python", "Cool-Chic/cc_decode.py",
        "-i", input_path,
        "-o", output_ppm,       
    ], check=True)
    
    ##### IMAGINARIO
    input_path  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_imag_{bits}bits_comp.cool"
    output_ppm  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_imag_{bits}bits_decomp.ppm"
    
    subprocess.run([
        "python", "Cool-Chic/cc_decode.py",
        "-i", input_path,
        "-o", output_ppm,       
    ], check=True)

def normalizeComplexMatrix(hologram, bits):
    if bits == 8:
        dtype = np.uint8
        max_scale = 255
    elif bits == 16:
        dtype = np.uint16
        max_scale = 65535

    #separar a parte real da imaginaria
    holo_real = np.real(hologram)
    holo_imaginary = np.imag(hologram)

    #valores de max e min de cada um para guardar
    maxR, minR = np.max(holo_real), np.min(holo_real) 
    maxI, minI = np.max(holo_imaginary), np.min(holo_imaginary) 

    #normalizar
    holo_real = ((holo_real - minR) / (maxR - minR)) * max_scale
    holo_imaginary = ((holo_imaginary - minI) / (maxI - minI)) * max_scale
    
    holo_real_b = holo_real.astype(dtype) 
    holo_imag_b = holo_imaginary.astype(dtype) 
    
    return holo_real_b, holo_imag_b, minR, maxR, minI, maxI

def denormalizeRealImag(fiber_name, lmbda, bits, minR, maxR, minI, maxI, filename="holo"):
    prefix = re.split(r"[_.]+", fiber_name)[0]
    base = f"reconstructions/fiber/{prefix}_glicerina/{fiber_name}"
    output_ppm_r  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_real_{bits}bits_decomp.ppm"
    output_ppm_i  = f"{base}/coolchic/lmbda_{lmbda}/{filename}_imag_{bits}bits_decomp.ppm"
    output_path_r = f"{base}/coolchic/lmbda_{lmbda}/{filename}_real_{bits}bits_decomp.npy"
    output_path_i = f"{base}/coolchic/lmbda_{lmbda}/{filename}_imag_{bits}bits_decomp.npy"
    output_path = f"{base}/coolchic/lmbda_{lmbda}/{filename}_{bits}bits_decomp.npy"

    if bits == 8:
        max_scale = 255.0
    elif bits == 16:
        max_scale = 65535.0

    img_real = cv2.imread(output_ppm_r, cv2.IMREAD_UNCHANGED)
    img_imag = cv2.imread(output_ppm_i, cv2.IMREAD_UNCHANGED)

    #no caso do formato ppm que e o default do cool chic output transformar em rgb
    if len(img_real.shape) == 3:
        img_real = img_real[:, :, 0]

    if len(img_imag.shape) == 3:
        img_imag = img_imag[:, :, 0]

    holo_real_float = img_real.astype(np.float64)
    holo_imag_float = img_imag.astype(np.float64)

    holo_real_rec = (holo_real_float / max_scale) * (maxR - minR) + minR
    holo_imag_rec = (holo_imag_float / max_scale) * (maxI - minI) + minI

    np.save(output_path_r, holo_real_rec)
    np.save(output_path_i, holo_imag_rec)

    #montar de novo a matrix complexa 
    holo_complexo_rec = holo_real_rec + (1j * holo_imag_rec)
    
    np.save(output_path, holo_complexo_rec)

    return holo_complexo_rec
