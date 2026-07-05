"""
Title-->            Numerical propagation examples.
Author-->           Ana Doblas, Carlos Trujillo, Raul Castaneda,
Date-->             05/09/2022
                    University of Memphis
                    Optical Imaging Research lab (OIRL)
                    EAFIT University
                    Applied Optics Group
Abstract -->        Script that implements the different methods to propagate the resulting complex field data
Links-->          - https://github.com/catrujilla/pyDHM

"""  
#%% Hologram Reconstruction Routine for Lloyd Mirror Common Path DHM
import importlib
from pyDHM import utils
importlib.reload(utils)
import time

from skimage.restoration import denoise_nl_means


from matplotlib.ticker import MultipleLocator

from pyDHM import numProp
importlib.reload(numProp)
import numpy as np

# Interactive file importing
import tkinter as tk
from tkinter import filedialog
from scipy.io import loadmat
import re

from matplotlib import pyplot as plt
import numpy as np
import skimage as ski
import cv2

import modules.autoOrderDetection as aod
import modules.compressionPipeline as cd
import modules.readResults as readRes
from modules.processUtils import select_roi_opencv, segment_fiber_grabcut_seeds, segment_fiber_grabcut, Rect_Window, pre_process, variance_norm, calc_zobj


'''
============================================================
FUNCTION INDEX
============================================================
 
STEP 1 - Loading & Pre-processing
----------------------------------
loadHolograms(path, pathRef)
    Loads the hologram and reference hologram from .mat files.
    Opens a file dialog if paths are None.
 
applyPreProcessing(imgHolo, imgHoloRef)
    Applies pre-processing to both holograms: padding, decimation,
    contrast subtraction and apodization windowing.
 
normalizeAndSaveHolo(holo_filter, filenameHolo)
    Normalizes the complex hologram to two 8-bit integer matrices
    (real + imaginary) and saves them as PNG + JSON min/max info.
 
STEP 2 - Fourier Transform & Order Selection
---------------------------------------------
manualOrderSelection(u1, u1r)
    Applies FFT and lets the user manually select the +1 order
    via spatial filter selection.
 
showInitialHologramState(ft_holo, imgHolo, procHolo, procRef, u1)
    Displays hologram, reference, spectrum and contrast hologram
    in a 2x2 subplot with spatial/frequency coordinates.
 
autoOrderSelect(filenameHolo, u1, u1r, onlyOrderPlus1=True)
    Automatically locates and isolates the +1 diffraction order
    using thresholding and geometric detection.
 
STEP 3 - Cool-Chic Compression
--------------------------------
runCoolChic(filenameHolo, lambda_now)
    Runs the Cool-Chic codec to compress and decompress the
    normalized hologram, then denormalizes back to complex form.
 
STEP 4 - Propagation
---------------------
loadPropagationInfo(holo_filter, filenameHolo, onlyOrderPlus1=True)
    Loads the previously saved order center/radius and the optimal
    z distance from JSON. Also loads or pads the hologram itself.
 
propagationOnZ(holo_filter, x_min, y_min, filenameHolo, onlyOrderPlus1=True)
    Propagates the hologram along z, computes focus metric,
    finds optimal plane and plots the focus curve with dual axis.
 
STEP 5 - Phase Unwrapping & Background Correction
---------------------------------------------------
refocusOptPlaneChooseROI(holo_filter, zextr, filenameHolo, ref_filter, applyCoolChic)
    Refocuses to optimal plane, selects ROI, extracts phase and
    intensity for hologram and reference, saves amplitude and phase.
 
showIntensityPhase(phase, intensity, phase_ref, ref_intensity)
    Displays intensity and phase of hologram and reference
    in a 2x2 subplot with spatial coordinates. Returns the spatial
    coordinates used for the plots.
 
phaseUnwrapping(phase, phase_ref, filenameHolo)
    Unwraps phase of hologram and reference, applies Gaussian
    smoothing and computes phase difference. Saves all results.
 
grabcutSegmentation(phase_dif, filenameHolo)
    Segments fiber from background using GrabCut algorithm.
    Displays fiber and background masks. Returns the background mask.
 
createAjdustedPlane(phase_dif, mask_bg)
    Fits a plane to the background pixels using least squares.
 
applyPlane(phase_dif, a, b, c)
    Subtracts the fitted plane from the full phase map.
 
showCorrectedDifference(phase_dif, phase_corr, xxc, yyc)
    Displays original and plane-corrected phase difference
    side by side with colorbars.
 
showPhaseUnwrapRes(xxc, yyc, phase_corr)
    Plots the 3D OPD (Optical Path Difference) surface profile
    from the corrected phase map. Returns the OPD conversion factor
    and meshgrids reused by the denoising plot.
 
STEP 6 - Phase Denoising
-------------------------
phaseDenoising(phase_corr, opt=1)
    Denoises the phase map using either non-local means (opt=1)
    or bilateral filter (opt=2).
 
showDenoisingRes(phase_denoised, factor_k0, X, Y, xxc, yyc, filenameHolo, lambda_now=None, applyCoolChic=False)
    Displays denoised OPD as both 3D surface and 2D map side by side.
    Saves the figure to the appropriate folder.
 


============================================================
'''

# ****************************************************
# STEP 1 - Hologram Loading and pre-processing
# ****************************************************

# .............................................
# Settings
# .............................................
wavelength = 632.8e-9       # wavelength [m]
Mtransv = 12.8    # Microscope Objective Magnification [x]
Subtract = False
AutoOrderSelect = True
saveFiles = True
folder = "fiber" #onde vão ficar guardadas as imagens

# .............................................
# Pixel subsampling in case of color camera
# .............................................
Decmt = False        # Decimate (1/2) pixel sampling?
Nout  = 2600        # Processed Hologram size (square) 
if Decmt:
    dx = 4.4e-6     # pixel pitch (4.4 um)*2 (color camera: Guppy Pro 503c)
else:
    dx = 2.2e-6     # pixel pitch (2.2 um)*2 (mono camera: Guppy Pro 503b)
# .............................................

# --- transformação zi -> zo ---
def zi_to_zo(zi):
    fl = 16
    zo = 1000*zi*fl/(Mtransv*(zi + fl*Mtransv))
    return zo

def zo_to_zi(zo):
    fl = 16
    zi = (zo*fl*Mtransv**2)/(-zo*Mtransv + fl)
    return zi

# .............................................
# 1.2 - File opennig
# .............................................
def loadHolograms(path=None, pathRef=None):
    '''
    does the loading of the hologram itself and its reference
 
    Parameters
    ----
    path : str
        path to the .mat hologram file
        if None then asks user to choose himself
    pathRef : str
        path to the .mat reference hologram file
        if None then asks user to choose himself
 
    Returns
    ----
    filenameHolo : str
        name of the fiber being anylized
    imgHolo : ndarray
        raw hologram intensity image as captured by the camera (real-valued)
    imgHoloRef : ndarray
        raw reference hologram intensity image as captured by the camera
        (real-valued)
    '''

    root = tk.Tk()
    root.withdraw()

    if path is None:
        # abrir diálogo de seleção de ficheiro do holograma
        filepath = filedialog.askopenfilename(
            title="Select .mat file",
            filetypes=[("MAT file", "*.mat")]
        )
    else:
        filepath = path

    # verificar se selecionou
    if not filepath:
        raise ValueError("No file selected")

    # extrair nome
    pathSplit = re.split(r"[/.]+", filepath)
    filename = pathSplit[-1]
    print(pathSplit)

    print("Selected file:", filepath)
    print("Filename:", filename)

    filenameHolo = filename

    # Criar a estrutura de pastas se for para guardar os dados
    if saveFiles:
        cd.create_fiber_structure(filename)

    # Loading hologram data
    dataHolo = loadmat(filepath)

    print(dataHolo.keys())

    Nframes = dataHolo["Nframes"][0][0]
    ShutterTime = dataHolo["ShutterTime"][0][0]
    imgHolo = dataHolo["I1"].squeeze()

    print(f"Nframes = {Nframes:.0f}")
    print(f"Shutter time = {ShutterTime} us")
    print(imgHolo.shape)

    if pathRef is None:
        # Authomatic Load of Reference Hologram
        # abrir diálogo de seleção de ficheiro do holograma
        filepath = filedialog.askopenfilename(
            title="Select Reference .mat file",
            filetypes=[("MAT file", "*.mat")]
        )
    else:
        filepath = pathRef

    # verificar se selecionou
    if not filepath:
        raise ValueError("No file selected")

    # extrair nome
    pathSplit = re.split(r"[/.]+", filepath)
    filename = pathSplit[-2]

    print("Selected file:", filepath)
    print("Filename:", filename)

    # Loading hologram data
    dataRef = loadmat(filepath)
    imgHoloRef = dataRef["I1"].squeeze()
    print(imgHoloRef.shape)

    return filenameHolo, imgHolo, imgHoloRef

# ........................................................
# 1.3 - Pre-processing data to prepare a square hologram
#       for reconstruction
# ........................................................
# Sample Hologram pre-processing
def applyPreProcessing(imgHolo, imgHoloRef):
    '''
    takes both holograms and apply pre processing, like padding if needed, and
    appodization
 
    Parameters
    ----
    imgHolo : ndarray
        raw hologram intensity image (real-valued)
    imgHoloRef : ndarray
        raw reference hologram intensity image (real-valued)
 
    Returns
    ----
    u1 : ndarray
        the hologram after being pre-processed (padded/decimated/apodized),
        or a contrast hologram if Subtract=True. Still real-valued at
        this stage;
    u1r : ndarray
        the reference hologram after being pre-processed, still real-valued
        at this stage 
    procHolo : ndarray
        the hologram after being pre-processed, before apodization/contrast

    '''
    procHolo = pre_process(imgHolo, Nout, Decmt)
    print(procHolo.shape)
    # utils.imageShow(procHolo, 'pre-proc. input field')
    # Reference Hologram pre-processing
    procRef = pre_process(imgHoloRef, Nout, Decmt)
    print(procRef.shape)
    # utils.imageShow(procRef, 'pre-proc. ref field')

    if Subtract:
        # Create Contrast Hologram
        u1 = procHolo - procRef   
    else:
        # Keep Original Hologram
        u1 = procHolo

    #  Reference
    u1r = procRef
    # Show Contrast or Original Hologram
    # utils.imageShow(u1, 'Contrast Hologram')   

    # Appodization filtering
    # N - number of columns - x
    # M - number of rows    - y
    M, N = u1.shape
    window = Rect_Window(N, M, eta=0.8, w=0.2)
    u1 *= window
    u1r *= window
    # Show Contrast or Original Hologram
    # utils.imageShow(u1, 'Apodized Hologram')   
    # utils.imageShow(u1r, 'Apodized Ref')   

    zCrit = 2.0e2*dx*dx*N/wavelength
    print(f"z critical = {zCrit: .3f} cm")

    return u1, u1r, procHolo


# ****************************************************
# STEP 2 - Fourier Transform
# ****************************************************
def manualOrderSelection(u1, u1r):    
    '''
    applys FT to the hologram and then asks the user to select the order manually

    Parameters
    ----
    u1 : ndarray
        the hologram in its complex form after being pre-processed, or a constrast hologram
    u1r : ndarray
        the reference hologram in its complex form after being pre-processed
    
    Returns
    ----
    ft_holo : ndarray
        the hologram in the Fourier domain
    holo_filter : ndarray
        may contain either just plus one order, or the full hologram with only the plus one order info
    ref_filter : ndarray
        the full reference hologram with only the order plus one info 
    '''
    # FFT to get the hologram spectrum
    ft_holo = utils.FT(u1)

    holo_filter, x, y, w, h = utils.sfmr(u1, True)
    # Apply Spatial filter to reference:
    ref_filter, x, y, w, h = utils.sfmr(u1r, False, (x, y, w, h))

    return ft_holo, holo_filter, ref_filter

    
def showInitialHologramState(ft_holo, imgHolo, procHolo, procRef, u1):
    '''
    show the hologram, the reference, the hologram spectrum and the contrast hologram

    Parameters
    ----
    ft_holo : ndarray
        the hologram in the Fourier domain
    imgHolo : ndarray
        the hologram in its complex form
    procHolo : ndarray
        the hologram in its complex form after being pre-processed
    procRef : ndarray
        the reference hologram in its complex form after being pre-processed
    u1 : ndarray
        the hologram in its complex form after being pre-processed, or a constrast hologram
    ----
    '''
    M0, N0 = imgHolo.shape
    M, N = u1.shape
    ft_holo_int = utils.intensity(ft_holo, True)
    
    if Decmt:
        r0 = int(np.floor((M - M0/2) / 2))
        r1 = int(np.floor((M + M0/2) / 2))
        c0 = int(np.floor((N - N0/2) / 2))
        c1 = int(np.floor((N + N0/2) / 2))
    else:
        r0 = int(np.floor((M - M0) / 2))
        r1 = int(np.floor((M + M0) / 2))
        c0 = int(np.floor((N - N0) / 2))
        c1 = int(np.floor((N + N0) / 2))
    
    # Crop to correct size for graphical representation
    procHoloc  = procHolo[r0:r1, c0:c1]
    procRefc  = procRef[r0:r1, c0:c1]
    u1c  = u1[r0:r1, c0:c1]
    ft_holo_intc = ft_holo_int[r0:r1, c0:c1]

    # Spatial Coordinates
    Mc, Nc = procHoloc.shape

    xxc = dx * (np.arange(Nc) - Nc/2)
    yyc = dx * (np.arange(Mc) - Mc/2)

    um_unit_scale = 1.0e6          # meters → micrometers
    obj_scale = um_unit_scale * (Mtransv**(-1))

    xxc *= obj_scale
    yyc *= obj_scale

    dy = dx
    Lxx = Nc * dx
    Lyy = Mc * dy

    # Frequency Coordinates:
    # fx = np.fft.fftshift(np.fft.fftfreq(Nc, d=dx))
    # fy = np.fft.fftshift(np.fft.fftfreq(Mc, d=dy))

    fx = np.arange(-1/(2*dx), 1/(2*dx), 1/Lxx)
    fy = np.arange(-1/(2*dy), 1/(2*dy), 1/Lyy)

    # Ensure exact length match (important for numerical safety)
    fx = fx[:Nc]
    fy = fy[:Mc]

    fx *= 1e-3   # cycles/m → cycles/mm
    fy *= 1e-3

    # --------------------------------
    # Show Hologram and Reference
    # --------------------------------

    plt.figure(figsize=(8,6))
    # --------------------------------
    # Hologram
    # --------------------------------
    plt.subplot(2,2,1)
    plt.imshow(np.abs(procHoloc)**0.5,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Hologram")
    plt.gca().set_aspect('equal')
    plt.gray()

    # --------------------------------
    # Reference
    # --------------------------------
    plt.subplot(2,2,2)
    plt.imshow(np.abs(procRefc)**0.5,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Reference")
    plt.gca().set_aspect('equal')
    plt.gray()

    # --------------------------------
    # Spectrum
    # --------------------------------
    plt.subplot(2,2,3)
    plt.imshow(ft_holo_intc**0.5,
            extent=[fx[0], fx[-1], fy[-1], fy[0]])
    plt.xlabel("fx (cpmm)")
    plt.ylabel("fy (cpmm)")
    plt.title("Hologram spectrum")
    plt.gca().set_aspect('equal')
    plt.gray()

    # --------------------------------
    # Contrast hologram
    # --------------------------------
    plt.subplot(2,2,4)
    plt.imshow(np.abs(u1c)**0.5,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Contrast Hologram")
    plt.gca().set_aspect('equal')
    plt.gray()

    plt.tight_layout()
    plt.show()

# =============================================================================================

#%% ****************************************************
# STEP 2.1 - Optional auto order +1 selection
#  ****************************************************
#if AutoOrderSelect:
def autoOrderSelect(filenameHolo, u1, u1r, onlyOrderPlus1=True):
    '''
    applies the Fourier Transform and utilizes the autoOrderSelection module to localize and create a mask
    on the order +1, and apply it to the hologram in the frequency domain

    Parameters
    ----
    filenameHolo : str
        name of the fiber being anylized
    u1 : ndarray
        the hologram in its complex form after being pre-processed, or a constrast hologram
    u1r : ndarray
        the reference hologram in its complex form after being pre-processed
    onlyOrderPlus1 : boolean
        default=True, if the hologram is the cut of the +1 order is True, otherwise its the full hologram

    Returns
    ----
    ft_holo : ndarray
        the hologram in the Fourier domain
    x_min : int
        starting column of the box that has the plus one order, only
        meaningful when onlyOrderPlus1=True; otherwise it's just 0, since
        no crop is applied in that case
    y_min : int
        starting row of the box that has the plus one order, only
        meaningful when onlyOrderPlus1=True; otherwise it's just 0, since
        no crop is applied in that case
    holo_filter : ndarray
        may contain either just plus one order, or the full hologram with only the plus one order info
    ref_filter : ndarray
        the full reference hologram with only the order plus one info 
    '''
    # FFT to get the hologram spectrum
    ft_holo = utils.FT(u1)
    
    M, N = ft_holo.shape

    #elimina o termo zero, e binariza a imagem
    bw = aod.threshold_FT(ft_holo, M, N)

    #utiliza a imagem binarizada para isolar as ordens
    #devolve o centro da ordem +1, altura do retangulo de ordem +1, largura do retangulo de ordem +1, distancia ate ao centro da ordem 1 vert, e horiz
    plus1_center, plus1_m, plus1_n, p, q, col_min, row_min, width, height, circle_radius = aod.get_plus1(bw)

    dy = dx

    x_vec = np.arange(-N/2, N/2) * dx
    y_vec = np.arange(-M/2, M/2) * dy
    X, Y = np.meshgrid(x_vec, y_vec)

    k = (2 * np.pi) / wavelength

    #devolve a ordem +1 e NAO compensa o tilt linear
    compensated_holo, masked = aod.filter_center_plus1(ft_holo, plus1_center, circle_radius)
   
    x_min = 0
    y_min = 0
    if onlyOrderPlus1:
        onlyOrder, x_min, y_min = aod.get_center_plus1_only(masked, plus1_center, circle_radius)
        print(type(onlyOrder))
        print(onlyOrder.dtype)
    
    holo_filter = compensated_holo
    if onlyOrderPlus1:
        holo_filter = onlyOrder

    FT_ref = np.fft.fftshift(np.fft.fft2(u1r))
    ref_filter, _ = aod.filter_center_plus1(FT_ref, plus1_center, circle_radius)

    filepath = f"reconstructions/fiber/{filenameHolo}/hologram/holo_order"
    
    np.save(filepath+"_ref_filter.npy", ref_filter)

    jsonPath = filepath + "_infoOrderCenter.json"
    readRes.save_json(jsonPath, {"orderCenterx":plus1_center[1], "orderCentery":plus1_center[0], "circleRadius":circle_radius})

    return ft_holo, x_min, y_min, holo_filter, ref_filter

def runCoolChic(filenameHolo, lambda_now):
    '''
    runs the cool-chic codec to compress the normalized hologram
    then it decompress the results from the compression, saves the max and min and denomalizes back to the full complex hologram

    Parameters
    ----
    filenameHolo : str
        name of the fiber that the hologram belongs to
    lambda_now : str
        indicates the lambda parameter that is being used in the compression

    Returns
    ----
    holo_filter : ndarray
        the hologram after undergoing compression and decompression, already denormalized
    '''

    bits = 8
    filepath = f"reconstructions/fiber/{filenameHolo}/hologram/holo_order"

    print(f"A comprimir {filenameHolo} com lambda {lambda_now}...")
    cd.compress_hologram(filenameHolo, lambda_now, bits, filename="holo_order")
    cd.decompress_hologram(filenameHolo, lambda_now, bits, filename="holo_order")

    jsonPath = filepath + "_infoMinMax.json"
    
    info = readRes.load_json(jsonPath)
    minR, maxR, minI, maxI = (info[k] for k in ("minR", "maxR", "minI", "maxI"))
        
    holo_filter = cd.denormalizeRealImag(filenameHolo, lambda_now, bits, minR, maxR, minI, maxI, filename="holo_order")

    return holo_filter

#%% ****************************************************
# STEP 3.1 - Optional compression
#  ****************************************************

def normalizeAndSaveHolo(holo_filter, filenameHolo):
    '''
    normalizes and saves the hologram so that its possible to feed it to a compression codec
    
    Parameters
    ----
    holo_filter : ndarray
        the hologram in its complex form with the order +1 only
    filenameHolo : str
        name of the fiber that the hologram belongs to
    '''
    bits = 8

    #if we arent using thee holo_order change it back to holo here
    filepath = f"reconstructions/fiber/{filenameHolo}/hologram/holo_order"

    #normalizar de complexo para duas matrizes de inteiro
    holo_real, holo_imag, minR, maxR, minI, maxI = cd.normalizeComplexMatrix(holo_filter, bits)

    #salvar o holograma bruto so com a ordem +1 para a compressão
    cv2.imwrite(f"{filepath}_real_{bits}bits.png", holo_real)
    cv2.imwrite(f"{filepath}_imag_{bits}bits.png", holo_imag)

    #salvar o min e o max
    info = {
        "minR": float(minR),
        "maxR": float(maxR),
        "minI": float(minI),
        "maxI": float(maxI)
    }
    
    jsonPath = filepath + "_infoMinMax.json"
    readRes.save_json(jsonPath, info)

    #salvar o holograma bruto so com a ordem +1 para a análise
    np.save(filepath+"PlusOne.npy", holo_filter)

# =============================================================================================
#%% ***************************************************
# STEP 4 -  Propagation along several z distances
#  ****************************************************
# 4.1 - Ciclo de propagação
# -----------------------------------------------------
def loadPropagationInfo(holo_filter, filenameHolo, onlyOrderPlus1=True):
    '''
    loads the propagation plane that was used in the original reference so that we can compare accuracty the results after the compression
    also pads the hologram in case it had only the order +1 size        

    Parameters
    ----
    holo_filter : ndarray
        the hologram in its complex form with the order +1 only
    filenameHolo : str
        name of the fiber that the hologram belongs to
    onlyOrderPlus1 : boolean
        default = True, if its True it means that the hologram contains only the order plus 1, otherwise its the full hologram but also with the information of the plus 1 order
    
    Returns 
    ----
    zextr : float
        propagation plane for the hologram
    holo_filter : ndarray
        the full hologram in its complex form, with the information of the order +1 only
    '''   
    filepath = f"reconstructions/fiber/{filenameHolo}/"
    jsonPath = filepath + "hologram/holo_order_infoOrderCenter.json"
    
    info = readRes.load_json(jsonPath)

    cx, cy, radius = int(info["orderCenterx"]), int(info["orderCentery"]), int(info["circleRadius"])
    
    y_min = max(0, cy - radius)
    x_min = max(0, cx - radius)
    
    #se for so a ordem sozinha, fazer padding
    if onlyOrderPlus1:
        holo_filter = aod.pad_with_zeros(holo_filter, 2600, y_min, x_min)

    #nao preciso do infoROI porque nao faço propagacao, vou logo buscar o z
    jsonPath = filepath + "reference/holo_order_infoProp_z.json"
            
    #buscar a distância z selecionada na propagaçao do holograma original
    info = readRes.load_json(jsonPath)
    zextr = float(info["zextr"])

    return zextr, holo_filter

def propagationOnZ(holo_filter, x_min, y_min, filenameHolo, onlyOrderPlus1=True):
    '''
    it apply propagation on a certain number of z planes to find the plane where the hologram is the most focused
    also pads the hologram in case it had only the order +1 size                
    
    Parameters
    ----
    holo_filter : ndarray
        the hologram in its complex form (with the +1 order isolated)
    x_min : int
        starting column of start of box that has the plus one order
    y_min : int
        starting row of the start of box that has the plus one order
    filenameHolo : str
        name of the fiber that the hologram belongs to
    onlyOrderPlus1 : boolean
        default = True, if its True it means that the hologram contains only the order plus 1, otherwise its the full hologram but also with the information of the plus 1 order
    

    Returns 
    ----
    zextr : float
        propagation plane for the hologram
    holo_filter : ndarray
        the full hologram in its complex form, with the information of the order +1 only
    '''   
    nzpts = 50
    fmetric = np.zeros(nzpts)

    #se for so a ordem sozinha, fazer padding
    if onlyOrderPlus1:
        holo_filter = aod.pad_with_zeros(holo_filter, 2600, y_min, x_min)

    print(holo_filter.size)

    # Inicialização
    output = numProp.angular_spectrum(holo_filter, 0, wavelength, dx, dx)
    intensity = utils.intensity(output, False)

    filepath = f"reconstructions/fiber/{filenameHolo}/reference/holo_order"
   
    rmin, rmax, cmin, cmax = select_roi_opencv(intensity, 0.3)

    #salvar os limites do recorte para aplicar o mesmo na compressão
    info = {
        "rmin": float(rmin),
        "rmax": float(rmax),
        "cmin": float(cmin),
        "cmax": float(cmax)
    }
    
    jsonPath = filepath + "_infoROI.json"
    readRes.save_json(jsonPath, info)

    # z em mm
    z = np.linspace(-4, 14, nzpts)

    start = time.time()

    for ii, zi in enumerate(z):

        # Propagação (converter mm em m)
        output = numProp.angular_spectrum(holo_filter, zi * 1e-3, wavelength, dx, dx)

        # Intensidade e métrica de foco
        intensity = utils.intensity(output, False)
        fmetric[ii] = variance_norm(intensity[rmin:rmax, cmin:cmax])

        # Object distance (from MO)
        zo = calc_zobj(zi, 160, 10, Mtransv)

        # (opcional debug)
        print(f"zi = {zi:.2f} mm, zo = {1e3*zo:.2f} um, metric = {fmetric[ii]:.4e}")

    end = time.time()
    print(f"Execution time: {end - start:.6f} seconds")
    
    # ----------------------------------
    # 4.2 - Normalização da métrica
    # ----------------------------------
    fmin = np.min(fmetric)
    imin = np.argmin(fmetric)

    fmetric1 = fmetric - fmin

    fmax = np.max(fmetric1)
    imax = np.argmax(fmetric1)

    fmetric1 = fmetric1 / fmax

    # ----------------------------------
    # 4.3 - Escolha do plano ótimo
    # ----------------------------------
    extreme = 0  # 0 = mínimo, 1 = máximo

    if extreme:
        zextr = z[imax]
    else:
        zextr = z[imin]

    #salvar o valor do z
    jsonPath = filepath + "_infoProp_z.json"
    readRes.save_json(jsonPath, {"zextr": float(zextr)})

    #Plot focus metric
    plt.figure(figsize=(5,4))
    plt.plot(z, fmetric1)

    # plt.xlabel('z_i (\u00B5m)')
    plt.xlabel(r'$z_i$ (mm)')
    plt.ylabel(r'$F_M$ (-)')

    # linha vertical
    plt.axvline(zextr, color='r', linestyle='--', linewidth=1.0)

    # marcador no ponto
    plt.plot(zextr, fmetric1[imin], 'ro')

    # anotação
    plt.text(zextr, fmetric1[imin],
            f'  z* = {zextr:.2f} mm',
            color='red',
            verticalalignment='bottom')
    # --- eixo secundário ---
    ax = plt.gca()

    # ----------------------------------
    # Controlo de ticks (eixo inferior)
    # ----------------------------------
    # ax.xaxis.set_major_locator(MultipleLocator((z.max()-z.min())/10))
    ax.xaxis.set_major_locator(MultipleLocator(1))   # de 1 mm em 1 mm
    ax.xaxis.set_minor_locator(MultipleLocator(0.5)) # ticks pequenos
    ax.tick_params(axis='x', labelsize=10)

    # ticks no eixo superior (ajustados!)
    # secax.xaxis.set_major_locator(MultipleLocator(0.01))  # ajusta conforme desejado
    secax = ax.secondary_xaxis('top', functions=(zi_to_zo, zo_to_zi))
    secax.set_xlabel(r'$z_o (\mu m)$')
    # secax.xaxis.set_major_locator(MultipleLocator(10.0))  # ajusta conforme desejado
    # secax.xaxis.set_minor_locator(MultipleLocator(5.0)) # ticks pequenos
    secax.tick_params(axis='x', labelsize=10)


    # ----------------------------------
    # Estética final
    # ----------------------------------
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Image and Object plane focus distance:
    zobjt_extr = calc_zobj(zextr, 160, 10, Mtransv)
    print(f"Zfimg = {zextr:.6f} mm")
    print(f"Zfobj = {zobjt_extr*1000:6f} \u00B5m")

    return zextr, holo_filter

# =============================================================================================
#%% **************************************************
# STEP 5 - Phase Unwrapping and Background Correction
# ****************************************************
def refocusOptPlaneChooseROI(holo_filter, zextr, filenameHolo, ref_filter=None, lambda_now=None, applyCoolChic=False):
    '''
    refocuses the hologram (and its reference) to the optimal z plane found
    in Step 4, lets the user choose a ROI (or reuses a previously saved one
    when applyCoolChic is True) and extracts phase and intensity for both
 
    Parameters
    ----
    holo_filter : ndarray
        the hologram in its complex form (with the +1 order isolated)
    zextr : float
        optimal propagation distance in mm, found in propagationOnZ
    filenameHolo : str
        name of the fiber that the hologram belongs to
    ref_filter : ndarray
        default = None, the reference hologram spectrum with only the +1
        order info; if None, it's loaded from disk (saved earlier by
        autoOrderSelect)
    lambda_now : str
        default = None, indicates the lambda parameter used in the
        compression, only relevant when applyCoolChic is True
    applyCoolChic : boolean
        default = False, if True reuses the ROI saved by a previous run
        instead of asking the user to select one again
 
    Returns
    ----
    phase : ndarray
        wrapped phase of the hologram ROI
    intensity : ndarray
        intensity of the hologram ROI
    phase_ref : ndarray
        wrapped phase of the reference ROI
    ref_intensity : ndarray
        intensity of the reference ROI
    '''

    filepath = f"reconstructions/fiber/{filenameHolo}/hologram/"
    if ref_filter is None:
        ref_filter = np.load(filepath + "holo_order_ref_filter.npy")

    # Refocusing to the optimum plane
    output = numProp.angular_spectrum(holo_filter, zextr*1.0e-3, wavelength, dx, dx)
    ref_output = numProp.angular_spectrum(ref_filter, zextr*1.0e-3, wavelength, dx, dx)

    # Mostrar intensidade para escolher ROI
    intensity_view = utils.intensity(output, False)

    filepath = f"reconstructions/fiber/{filenameHolo}/reference/"
    jsonPath = filepath + "holo_order_infoROI_view.json"

    #se for para o holograma descomprido, ir buscar medidas usadas no original
    if applyCoolChic:    
        info = readRes.load_json(jsonPath)
        rmin, rmax, cmin, cmax = (int(info[k]) for k in ("rmin", "rmax", "cmin", "cmax"))
    else:
        rmin, rmax, cmin, cmax = select_roi_opencv(intensity_view, 0.3)

        #salvar os limites do recorte para aplicar o mesmo na compressão
        readRes.save_json(jsonPath, {"rmin": float(rmin), "rmax": float(rmax),
                                    "cmin": float(cmin), "cmax": float(cmax)})

    # Extrair ROI
    holo_roi = output[rmin:rmax, cmin:cmax]
    ref_roi  = ref_output[rmin:rmax, cmin:cmax]

    # Phase
    phase = utils.phase(holo_roi)
    phase_ref = utils.phase(ref_roi)

    # Intensity
    intensity = utils.intensity(holo_roi, False)
    ref_intensity = utils.intensity(ref_roi, False)

    if applyCoolChic:
        filepath = f"reconstructions/fiber/{filenameHolo}/coolchic/lmbda_{lambda_now}/"
    else:
        filepath = f"reconstructions/fiber/{filenameHolo}/hologram/"
        
    #salvar a amplitude e a fase para possivel análise
    np.save(filepath+"amplitude.npy", intensity)
    np.save(filepath+"phase.npy", phase)

    return phase, intensity, phase_ref, ref_intensity

def showIntensityPhase(phase, intensity, phase_ref, ref_intensity):  
    '''
    shows intensity and phase of hologram and reference in a 2x2 subplot
    with spatial coordinates
 
    Parameters
    ----
    phase : ndarray
        wrapped phase of the hologram ROI
    intensity : ndarray
        intensity of the hologram ROI
    phase_ref : ndarray
        wrapped phase of the reference ROI
    ref_intensity : ndarray
        intensity of the reference ROI
 
    Returns
    ----
    xxc : ndarray
        x spatial coordinates (µm), reused by the later phase plots
    yyc : ndarray
        y spatial coordinates (µm), reused by the later phase plots
    '''

    
    # --------------------------------
    # Show Hologram and Reference
    # --------------------------------
    # Coordenadas Espaciais para o Gráfico
    Mc, Nc = phase.shape
    xxc = dx * (np.arange(Nc) - Nc/2)
    yyc = dx * (np.arange(Mc) - Mc/2)
    um_unit_scale = 1.0e6          
    obj_scale = um_unit_scale * (Mtransv**(-1))
    xxc *= obj_scale
    yyc *= obj_scale

    plt.figure(figsize=(10,8))
    # --------------------------------
    # Hologram
    # --------------------------------
    plt.subplot(2,2,1)
    plt.imshow(intensity,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Hologram Intensity")
    plt.gca().set_aspect('equal')
    plt.gray()


    plt.subplot(2,2,2)
    plt.imshow(phase,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Hologram Phase")
    plt.gca().set_aspect('equal')
    plt.gray()

    # --------------------------------
    # Reference
    # --------------------------------
    plt.subplot(2,2,3)
    plt.imshow(ref_intensity,
                extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Reference Intensity")
    plt.gca().set_aspect('equal')
    plt.gray()

    # --------------------------------
    # Contrast hologram
    # --------------------------------
    plt.subplot(2,2,4)
    plt.imshow(phase_ref,
            extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Reference Phase")
    plt.gca().set_aspect('equal')
    plt.gray()

    plt.tight_layout()
    plt.show()

    return xxc, yyc

# ------------------------------------------
# 5.1 - Phase unwrapping
# ------------------------------------------
# EN: Unwrap phase of hologram and reference
# PT: Desenrolar a fase do holograma e referência

def phaseUnwrapping(phase, phase_ref, filenameHolo, lambda_now=None, applyCoolChic=False):
    '''
    unwraps the phase of hologram and reference, applies a light Gaussian
    smoothing to each, and computes the phase difference between them.
    Saves the wrapped phase, both unwrapped versions and the difference
 
    Parameters
    ----
    phase : ndarray
        wrapped phase of the hologram ROI
    phase_ref : ndarray
        wrapped phase of the reference ROI
    filenameHolo : str
        name of the fiber that the hologram belongs to
    lambda_now : str
        default = None, indicates the lambda parameter used in the
        compression, only relevant when applyCoolChic is True
    applyCoolChic : boolean
        default = False, changes the save folder to the coolchic results
        folder for the given lambda_now
 
    Returns
    ----
    phase_dif : ndarray
        unwrapped phase difference between hologram and reference
    '''
    
    print("Starting phase unwrapping... / A iniciar unwrapping...")

    unwrp_phase_og = ski.restoration.unwrap_phase(phase)
    unwrp_phase = cv2.GaussianBlur(unwrp_phase_og, (0,0), 0.5)

    unwrp_phase_ref = ski.restoration.unwrap_phase(phase_ref)
    unwrp_phase_ref = cv2.GaussianBlur(unwrp_phase_ref, (0,0), 0.5)

    # Phase difference
    phase_dif = unwrp_phase - unwrp_phase_ref

    if applyCoolChic:
        filepath = f"reconstructions/fiber/{filenameHolo}/coolchic/lmbda_{lambda_now}/phase/"
    else:
        filepath = f"reconstructions/fiber/{filenameHolo}/reference/"
        
    #salvar a amplitude e a fase para possivel análise
    np.save(filepath+"phaseWrapped.npy", phase)
    np.save(filepath+"phaseUnwrapped.npy", unwrp_phase_og)
    np.save(filepath+"phaseUnwrappedGaussian.npy", unwrp_phase)
    np.save(filepath+"phaseUnwrappedDiff.npy", phase_dif)
  
    return phase_dif
# ------------------------------------------
# 5.2 - Background selection (manual)
# ------------------------------------------
# EN: Select background region manually
# PT: Selecionar manualmente região de fundo

print("Select BACKGROUND region / Selecione região de FUNDO")

# seleção manual de ROI de fundo
# rmin_bg, rmax_bg, cmin_bg, cmax_bg = select_roi_opencv(phase_dif, 0.4)
# bg_region = phase_dif[rmin_bg:rmax_bg, cmin_bg:cmax_bg]

# ------------------------------------------
# 5.2 - Fiber segmentation using GrabCut
# ------------------------------------------
# EN: Segment fiber and background
# PT: Segmentar fibra e fundo
def grabcutSegmentation(phase_dif, filenameHolo, applyCoolChic, saveReference=True):
    '''
    segments fiber from background using GrabCut (method 1, delegated to
    segment_fiber_grabcut) or a seeded variant (method 2, segment_fiber_grabcut_seeds),
    then displays the resulting fiber and background masks
 
    Parameters
    ----
    phase_dif : ndarray
        unwrapped phase difference map to segment
    filenameHolo : str
        name of the fiber that the hologram belongs to
    applyCoolChic : boolean
        if its True it follows the cool-chic logic, specific of the compression
        where it reuses the area selected from the original not compressed hologram
    saveReference : boolean
        default = True
 
    Returns
    ----
    mask_bg : ndarray
        background mask (complement of the fiber mask)
    '''
    
    segm_method = 1

    print("Select BACKGROUND region / Selecione região de FUNDO")

    match segm_method:
        case 1:
            mask_fiber = segment_fiber_grabcut(phase_dif, filenameHolo, applyCoolChic, saveReference)
        case 2:
            mask_fiber = segment_fiber_grabcut_seeds(phase_dif, 0.3)

    # Background mask (complement)
    # Máscara de fundo (complemento)
    mask_bg = 1 - mask_fiber

    # Visualizar máscara
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(mask_fiber, cmap='gray')
    plt.title("Fiber mask / Máscara fibra")

    plt.subplot(1,2,2)
    plt.imshow(mask_bg, cmap='gray')
    plt.title("Background mask / Máscara fundo")

    plt.tight_layout()
    plt.show()
    
    return mask_bg

# ------------------------------------------
# 5.3 - Plane fitting using background mask
# ------------------------------------------
# EN: Fit plane using ONLY background pixels
# PT: Ajustar plano usando APENAS fundo

def createAjdustedPlane(phase_dif, mask_bg):
    '''
    fits a plane z = a*x + b*y + c to the phase using only the background
    pixels (as given by mask_bg), via least squares
 
    Parameters
    ----
    phase_dif : ndarray
        unwrapped phase difference map
    mask_bg : ndarray
        binary background mask (1 = background pixel, 0 = fiber pixel)
 
    Returns
    ----
    a : float
        plane coefficient for x
    b : float
        plane coefficient for y
    c : float
        plane offset
    '''
    
    ny, nx = phase_dif.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    Xf = X[mask_bg == 1]
    Yf = Y[mask_bg == 1]
    Zf = phase_dif[mask_bg == 1]

    A = np.column_stack((Xf, Yf, np.ones_like(Xf)))

    coeffs, _, _, _ = np.linalg.lstsq(A, Zf, rcond=None)
    a, b, c = coeffs

    print(f"Plane fit / Plano ajustado: z = {a:.3e}x + {b:.3e}y + {c:.3e}")

    return a, b, c

# ------------------------------------------
# 5.4 - Apply plane correction to full image
# ------------------------------------------
# EN: Apply fitted plane to full phase map
# PT: Aplicar plano ajustado à imagem completa
def applyPlane(phase_dif, a, b, c):
    '''
    subtracts the background plane fitted in createAjdustedPlane from the
    full phase difference map, removing the linear background tilt
 
    Parameters
    ----
    phase_dif : ndarray
        unwrapped phase difference map
    a : float
        plane coefficient for x
    b : float
        plane coefficient for y
    c : float
        plane offset
 
    Returns
    ----
    phase_corr : ndarray
        plane-corrected phase difference map
    '''
    
    ny_full, nx_full = phase_dif.shape
    X_full, Y_full = np.meshgrid(np.arange(nx_full), np.arange(ny_full))

    plane = a * X_full + b * Y_full + c
    phase_corr = phase_dif - plane

    return phase_corr
# ------------------------------------------
# 5.5 - Visualization
# ------------------------------------------
# EN: Compare original and corrected phase
# PT: Comparar fase original e corrigida
def showCorrectedDifference(phase_dif, phase_corr, xxc, yyc):
    '''
    displays the original and plane-corrected phase difference maps side
    by side, each with its own colorbar
 
    Parameters
    ----
    phase_dif : ndarray
        unwrapped phase difference map, before plane correction
    phase_corr : ndarray
        plane-corrected phase difference map
    xxc : ndarray
        x spatial coordinates (µm) used for the plots
    yyc : ndarray
        y spatial coordinates (µm) used for the plots
    '''

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(phase_dif, extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]],cmap='viridis')
    plt.title("Phase difference / Diferença de fase")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.4)

    plt.subplot(1,2,2)
    plt.imshow(phase_corr, extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]],cmap='viridis')
    plt.title("Plane corrected / Correção por plano")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.gca().set_aspect('equal')
    plt.colorbar(shrink=0.4)

    plt.tight_layout()
    plt.show()

# =============================================================================================
# Save unwrapped phase
# from scipy.io import savemat

# savemat("phase_corr.mat", {
#     "phase_corr": phase_corr_plane,
#     "xxc": xxc,
#     "yyc": yyc
# })
# =============================================================================================
#%% ---------------------------------------------------
# Hologram Phase Unwrapping Results (3D Plots)
# ---------------------------------------------------
def showPhaseUnwrapRes(xxc, yyc, phase_corr):    
    '''
    plots the 3D OPD (Optical Path Difference) surface profile from the
    plane-corrected phase map, converting phase to optical path difference
    using the glycerin/cellulose refractive index contrast
 
    Parameters
    ----
    xxc : ndarray
        x spatial coordinates (µm) used for the plot
    yyc : ndarray
        y spatial coordinates (µm) used for the plot
    phase_corr : ndarray
        plane-corrected phase difference map
 
    Returns
    ----
    factor_k0 : float
        conversion factor from phase (rad) to optical path difference (µm)
    X : ndarray
        meshgrid built from xxc, reused by showDenoisingRes
    Y : ndarray
        meshgrid built from yyc, reused by showDenoisingRes
    '''

    ir_glycerin = 1.4965
    ir_celullose = 1.46869

    X, Y = np.meshgrid(xxc, yyc)

    delta_n = np.abs(ir_glycerin - ir_celullose)
    factor_k0 = (1e6*wavelength/(2*np.pi*delta_n))
    optThickness = -(phase_corr * factor_k0)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, optThickness, cmap='viridis', edgecolor='none')

    ax.set_title('3D OPD Profile (DHM)')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_zlabel('DPO (µm)')
    ax.set_box_aspect([1, 0.5, 0.3])
    ax.view_init(elev=70, azim=70)

    #  Colorbar with label
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.ax.set_title(r'$\mathrm{OPD}\ (\mu \mathrm{m})$', pad=10)
    plt.show()

    return factor_k0, X, Y

# =============================================================================================
#%% **************************************************
# STEP 6 - Phase Denoising
# ****************************************************
# ..........................................
# 6.1 - Apply one of two denoising methods
# ..........................................
def phaseDenoising(phase_corr, opt=1):
    '''
    denoises the plane-corrected phase map using either non-local means
    (opt=1) or a bilateral filter (opt=2)
 
    Parameters
    ----
    phase_corr : ndarray
        plane-corrected phase difference map
    opt : int
        default = 1, denoising method to use: 1 = non-local means,
        2 = bilateral filter
 
    Returns
    ----
    phase_denoised : ndarray
        denoised phase map
    '''

    
    match opt:
        case 1:
            phase_denoised = denoise_nl_means(
                phase_corr,
                h=0.5,
                patch_size=50,
                patch_distance=20
            )
        case 2:
            phase_denoised = cv2.bilateralFilter(
                phase_corr.astype(np.float32),
                d=51,               # tamanho da vizinhança
                sigmaColor=0.2,    # sensível à diferença de fase
                sigmaSpace=30      # suavização espacial
            )

    return phase_denoised

#%% .............................
# 6.2 - Plot Denoising results
# .............................
def showDenoisingRes(phase_denoised, factor_k0, X, Y, xxc, yyc, filenameHolo, lambda_now=None, applyCoolChic=False):
    '''
    plots the denoised OPD as a 3D surface and a 2D map side by side, and
    saves the resulting figure to disk
 
    Parameters
    ----
    phase_denoised : ndarray
        denoised phase map, from phaseDenoising
    factor_k0 : float
        conversion factor from phase (rad) to optical path difference (µm),
        from showPhaseUnwrapRes
    X : ndarray
        meshgrid used for the 3D surface plot, from showPhaseUnwrapRes
    Y : ndarray
        meshgrid used for the 3D surface plot, from showPhaseUnwrapRes
    xxc : ndarray
        x spatial coordinates (µm) used for the 2D map
    yyc : ndarray
        y spatial coordinates (µm) used for the 2D map
    filenameHolo : str
        name of the fiber that the hologram belongs to
    lambda_now : str
        default = None, indicates the lambda parameter used in the
        compression, only relevant when applyCoolChic is True
    applyCoolChic : boolean
        default = False, changes the save folder to the coolchic results
        folder for the given lambda_now
    '''

    optThickness = -(phase_denoised * factor_k0)

    # Criar figura com dois subplots
    fig = plt.figure(figsize=(12, 5))

    # --------------------------------
    # 3D Plot (à esquerda)
    # 3D gráfico (à esquerda)
    # --------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    surf = ax1.plot_surface(X, Y, optThickness,
                            cmap='viridis',
                            edgecolor='none')

    ax1.set_title('3D OPD Profile (DHM)')
    ax1.set_xlabel('x (µm)')
    ax1.set_ylabel('y (µm)')
    ax1.set_zlabel('DPO (µm)')
    ax1.set_box_aspect([1, 1, 0.2])
    ax1.view_init(elev=45, azim=45)

    # Colorbar associada só ao gráfico 3D
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
    cbar.ax.set_title(r'$\mathrm{OPD}\ (\mu \mathrm{m})$', pad=10)

    # --------------------------------
    # 2D Plot (à direita)
    # Gráfico 2D (à direita)
    # --------------------------------
    ax2 = fig.add_subplot(1, 2, 2)

    im = ax2.imshow(optThickness, cmap='viridis',
                    extent=[xxc[0], xxc[-1], yyc[-1], yyc[0]])

    ax2.set_title('2D OPD Map')
    ax2.set_xlabel('x (µm)')
    ax2.set_ylabel('y (µm)')
    ax2.set_aspect('equal')

    # Colorbar do 2D separada
    cbar2 = fig.colorbar(im, ax=ax2, shrink=0.6)
    cbar2.ax.set_title(r'$\mathrm{OPD}\ (\mu \mathrm{m})$', pad=10)

    # Ajustar layout
    plt.tight_layout()
    if applyCoolChic:
        filepath = f"reconstructions/fiber/{filenameHolo}/coolchic/lmbda_{lambda_now}/3D/"
    else:
        filepath = f"reconstructions/fiber/{filenameHolo}/3Dgraph/"
        
    plt.savefig(filepath + '3d-' + filenameHolo + '.png')
    plt.show()
