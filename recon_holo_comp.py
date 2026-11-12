# %% -------------------------------------------------------------------------------

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Lloyd_DHM_PhaseRecROI_AUTO_COMP as autoComp
import modules.readResults as readR
import modules.pda_metric as metricP
import modules.quantMethods as qm
import modules.roi as roiOps
import numpy as np
import re
from collections import defaultdict
from modules.reconstructionState import ReconstructionParams

applyCoolChic = True
onlyOrderPlus1 = True
saveReference = not applyCoolChic # para salvar as medidas do roi para ser igual na compressão 
lambdas = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
#lambda_now = lambdas[0] #escolher qual dos lambdas vai ser agora

# %% -------------------------------------------------------------------------------
#todos os nomes das fibras disponiveis organizados em categorias 
fibersA = ["FiberA_1", "FiberA_1_lowerpart", "FiberA_2", "FiberA_3", "FiberA_4"]
fibersB = ["FiberB_0", "FiberB_1", "FiberB_2", "FiberB_3", "FiberB_4"]
cannabis = ["cannabis_1g", "cannabis_2g", "cannabis_3g"]
CF_UB_Novembro = ["FiberB001", "FiberB002", "FiberB003", "FiberB004", "FiberB005", "FiberB006", "FiberB007", "FiberB008", "FiberB009", "FiberB0010"]
CMF = ["FiberA001", "FiberA002", "FiberA003", "FiberA004", "FiberA005", "FiberA006", "FiberA007", "FiberA008", "FiberA009", "FiberA0010"]
Eucalyptus = ["FiberE001", "FiberE002", "FiberE003", "FiberE004", "FiberE005", "FiberE006", "FiberE007", "FiberE008", "FiberE009", "FiberE010", "FiberE011", "FiberE012", "FiberE013", "FiberE014", "FiberE015"]
Pinus = ["FiberD001", "FiberD002", "FiberD003", "FiberD004", "FiberD005", "FiberD006", "FiberD007", "FiberD008", "FiberD009", "FiberD010", "FiberD011"]
#cannabis = ["cannabis_1g", "cannabis_2g", "cannabis_3g", "cannabis_4g"]
#fibersA = ["FiberA_1", "FiberA_1_lowerpart", "FiberA_2", "FiberA_3", "FiberA_4", "FiberA_5"]

#todas as fibras
allFibers = (
    fibersA
    + fibersB
    + cannabis
    + CF_UB_Novembro
    + CMF
    + Eucalyptus
    + Pinus
)

mainFolder = "Fibras_glicerina"
PATH_MAP = {
    "FiberA": "FiberA_glicerina/",
    "FiberB": "FiberB_glicerina/",
    "cannabis": "cannabis_glicerina/"
}

#caminho para as amostras geral
samplePath = f"data/DataIn/{mainFolder}/"

#caminho especifico para cada tipo de amostra de fibras
fApath = samplePath + "FiberA_glicerina/"
fBpath = samplePath + "FiberB_glicerina/"
cnpath = samplePath + "cannabis_glicerina/"
cfUBPath = "data/DataIn/CF-UB-Novembro/"
cmfPath = "data/DataIn/CMF/"
eucalyptusPath = "data/DataIn/Eucalyptus/"
pinusPath = "data/DataIn/Pinus/"

#caminho de cada referencia, uma por categoria
refA = fApath + "FiberA_ref.mat"
refB = fBpath + "FiberB_ref.mat"
refcn = cnpath + "cannabis_refg.mat"
refCF_UB = cfUBPath + "Ref-FiberB.mat"
refCMF = cmfPath + "Ref-FiberA.mat"
refEucalyptus = eucalyptusPath + "FiberERef.mat"
refPinus = pinusPath + "FiberDRefAir.mat"

SAMPLE_TYPES = {
    "FiberA":     (fApath,          refA),
    "FiberB":     (fBpath,          refB),
    "cannabis":   (cnpath,          refcn),
    "CF":         (cfUBPath,        refCF_UB),
    "CMF":        (cmfPath,         refCMF),
    "Eucalyptus": (eucalyptusPath,  refEucalyptus),
    "Pinus":      (pinusPath,       refPinus),
}

COMPACT_TYPE = {
    "FiberA": "CMF",
    "FiberB": "CF",
    "FiberE": "Eucalyptus",
    "FiberD": "Pinus",
    "cannabis": "cannabis",
}

recParams = ReconstructionParams(
    wavelength=632.8e-9,
    mtransv=12.8,
    decmt=False,
    nout=2600
)

def get_fiber_type(name):
    prefix = re.match(r"[A-Za-z]+", name).group()
    nxt = name[len(prefix):len(prefix) + 1]
    if prefix in COMPACT_TYPE and nxt.isdigit():
        return COMPACT_TYPE[prefix]
    if prefix in COMPACT_TYPE:
        return COMPACT_TYPE[prefix]
    return prefix

def test(fiber, path, ref):
    #carregar os hologramas do objeto e da referência
    filenameHolo, imgHolo, imgHoloRef = autoComp.loadHolograms(path + fiber, ref)
    
    #aplicar pre processamento nos dois hologramas
    u1, u1r, procHolo = autoComp.applyPreProcessing(imgHolo, imgHoloRef)

    #selecionar a ordem +1 e descartar o que é desnecessário
    ft_holo, x_min, y_min, holo_filter, ref_filter = autoComp.autoOrderSelect(filenameHolo, u1, u1r)

def numericalReconstructionAuto(fiber, path, ref):
    #carregar os hologramas do objeto e da referência
    filenameHolo, imgHolo, imgHoloRef = autoComp.loadHolograms(path + fiber, ref)
    
    #aplicar pre processamento nos dois hologramas
    u1, u1r, procHolo = autoComp.applyPreProcessing(recParams, imgHolo, imgHoloRef)

    #selecionar a ordem +1 e descartar o que é desnecessário
    ft_holo, x_min, y_min, holo_filter, ref_filter = autoComp.autoOrderSelect(recParams, filenameHolo, u1, u1r)

    #opcional - mostrar os hologramas
    autoComp.showInitialHologramState(recParams, ft_holo, imgHolo, procHolo, u1r, u1)

    #normalizar e salvar o hologramas para futura compressão
    autoComp.normalizeAndSaveHolo(holo_filter, filenameHolo)

    #realizar a prpogação em varios planos z para encontrar o melhor
    zextr, holo_filter = autoComp.propagationOnZ(recParams, holo_filter, x_min, y_min, filenameHolo, True)

    #refocagem no plano otimo z e separação em fase e amplitude
    phase, intensity, phase_ref, ref_intensity = autoComp.refocusOptPlaneChooseROI(holo_filter, zextr, filenameHolo, ref_filter)

    #opcional - mostrar a fase e a intensidade
    xxc, yyc = autoComp.showIntensityPhase(recParams, phase, intensity, phase_ref, ref_intensity)

    #fazer o phase unwrapping
    phase_dif = autoComp.phaseUnwrapping(phase, phase_ref, filenameHolo)

    mask_bg = autoComp.grabcutSegmentation(phase_dif, filenameHolo, False)

    a, b, c = autoComp.createAjdustedPlane(phase_dif, mask_bg)

    phase_corr = autoComp.applyPlane(phase_dif, a, b, c)

    mask_fiber  = 1 - mask_bg  #inverter os valores
    qm.runQuantMethods(recParams, phase_corr, mask_fiber, filenameHolo) 

    #optional - mostrar a diferença corrigida
    autoComp.showCorrectedDifference(phase_dif, phase_corr, xxc, yyc)

    factor_k0, X, Y = autoComp.showPhaseUnwrapRes(recParams, xxc, yyc, phase_corr)

    phase_denoised = autoComp.phaseDenoising(phase_corr)
    qm.runQuantMethods(recParams, phase_denoised, mask_fiber, filenameHolo, tag="denoised") 

    autoComp.showDenoisingRes(phase_denoised, factor_k0, X, Y, xxc, yyc, filenameHolo)
    

def numericalReconstructionAuto_CoolChic(fiber, lmbda):
    #ir buscar o holograma que correu sem compressão e comprimir, descomprimir e desnormalziar
    holo_filter = autoComp.runCoolChic(fiber, lmbda)
    
    #de forma a poder realizar comparações mais tarde, usar o mesmo z para a propagação que o original
    zextr, holo_filter = autoComp.loadPropagationInfo(holo_filter, fiber, True)

    #refocagem no plano otimo z e separação em fase e amplitude
    phase, intensity, phase_ref, ref_intensity = autoComp.refocusOptPlaneChooseROI(holo_filter, zextr, fiber, lambda_now=lmbda, applyCoolChic=True)

    #opcional - mostrar a fase e a intensidade
    xxc, yyc = autoComp.showIntensityPhase(phase, intensity, phase_ref, ref_intensity)

    #fazer o phase unwrapping
    phase_dif = autoComp.phaseUnwrapping(phase, phase_ref, fiber, lmbda, True)

    #utilizar o gracut para segmentação
    mask_bg = autoComp.grabcutSegmentation(phase_dif, fiber, True)

    #criar e ajustar o plano ao objeto
    a, b, c = autoComp.createAjdustedPlane(phase_dif, mask_bg)

    #aplciar o plano criado anteriormente
    phase_corr = autoComp.applyPlane(phase_dif, a, b, c)

    #opcional - mostrar a diferença corrigida
    autoComp.showCorrectedDifference(phase_dif, phase_corr, xxc, yyc)

    #opcional - mostrar o resultado final da phase apos ser corrigida
    factor_k0, X, Y = autoComp.showPhaseUnwrapRes(recParams, xxc, yyc, phase_corr)

    phase_denoised = autoComp.phaseDenoising(phase_corr)

    autoComp.showDenoisingRes(phase_denoised, factor_k0, X, Y, xxc, yyc, fiber, lmbda, True)

#made to run auto for all fibers or some specific category
def runReconstruction(fiberGroup):
    #run everythin again just to get the radius values for reference
    for fiber in fiberGroup:
        fiberType = get_fiber_type(fiber)

        if fiberType not in SAMPLE_TYPES:
            print(f"Tipo de fibra desconhecido: {fiber} ({fiberType})")
            continue                     

        path, ref = SAMPLE_TYPES[fiberType]

        #test(fiber, path, ref)
        numericalReconstructionAuto(fiber, path, ref)

def runCoolChic(fiberGroup):
    for fiber in fiberGroup:
        for lmb in lambdas:
            numericalReconstructionAuto_CoolChic(fiber, lmb)
            #numericalReconstructionAuto(fiber, path, ref)

########################################################################################################################################################################
## Correr as métricas para cada fibra                                                                                                                                 ##
########################################################################################################################################################################

def getMetricsHoloComplex(fiber, isCategory=False, isAll=False):
    if not isinstance(fiber, list):
        fiberList = [fiber]
    else:
        fiberList = fiber

    #real e imaginario
    readR.getRDGraphsReIm(fiberList)

    #complexo para cada fibra psnr e ssim
    readR.getRDGraphComplex(fiberList)

    #complexo para cada categoria
    if isCategory:
        categories = defaultdict(list)
        for fib in fiberList:
            categories[get_fiber_type(fib)].append(fib)

        for cat_prefix, group_items in categories.items():
            readR.getRDGraphComplexAll(group_items, isCategory=isCategory, isAll=False)

    #complexo para um grafico global com todas as fibras
    if isAll:
        readR.getRDGraphComplexAll(fiberList, isCategory=False)

def getMetricsPhase(fiber, isCategory=False, isAll=False):
    if not isinstance(fiber, list):
        fiberList = [fiber]
    else:
        fiberList = fiber
  
    #fases para cada fibra psnr e ssim
    readR.getRDGraphPhase(fiberList, "Wrapped", ['psnr', 'ssim'], "phaseWrapped", isPhaseWrapped=True)
    readR.getRDGraphPhase(fiberList, "Unwrapped", ['psnr', 'ssim'], "phaseUnwrapped", isPhaseWrapped=False)
 
    #fases para cada categoria
    if isCategory:
        categories = defaultdict(list)
        for fib in fiberList:
            categories[get_fiber_type(fib)].append(fib)

        for cat_prefix, group_items in categories.items():
            readR.getRDGraphPhaseAll(group_items, "Wrapped", ['psnr'], "phaseWrapped", isAll=False, isCategory=isCategory, isPhaseWrapped=True)
            readR.getRDGraphPhaseAll(group_items, "Wrapped", ['ssim'], "phaseWrapped", isAll=False, isCategory=isCategory, isPhaseWrapped=True)
            readR.getRDGraphPhaseAll(group_items, "Unwrapped", ['psnr'], "phaseUnwrapped", isAll=False, isCategory=isCategory, isPhaseWrapped=False)
            readR.getRDGraphPhaseAll(group_items, "Unwrapped", ['ssim'], "phaseUnwrapped", isAll=False, isCategory=isCategory, isPhaseWrapped=False)
 
    #fases para um grafico global com todas as fibras
    if isAll:
        readR.getRDGraphPhaseAll(allFibers, "Wrapped", ['psnr'], "phaseWrapped", isAll=isAll, isPhaseWrapped=True)
        readR.getRDGraphPhaseAll(allFibers, "Wrapped", ['ssim'], "phaseWrapped", isAll=isAll, isPhaseWrapped=True)
        readR.getRDGraphPhaseAll(allFibers, "Unwrapped", ['psnr'], "phaseUnwrapped", isAll=isAll, isPhaseWrapped=False)
        readR.getRDGraphPhaseAll(allFibers, "Unwrapped", ['ssim'], "phaseUnwrapped", isAll=isAll, isPhaseWrapped=False)

def getMetricsCombined(allFibers):
    metrics = ['psnr', 'ssim']
    for metric in metrics:
        df_holo, _ = readR.getAllFibersDataComplex(allFibers, metric)
        df_phase, _ = metricP.getAllFibersPhaseMetric(allFibers, False, metric)
        df_pda, _ = metricP.getAllFibersPhaseMetric(allFibers, True, metric)

        readR.getRDGraphCombined(df_holo, df_phase, df_pda, [metric], isAll=True)

########################################################################################################################################################################
## Choose ROI                                                                                                                                                         ##
########################################################################################################################################################################


#########
#runReconstruction(fibersA[-1])
#runCoolChic(fibersB)

arr2 = ["FiberB_0", "FiberB_1"]
arr3 = [x for x in fibersB if x not in arr2]

#getMetricsHoloComplex(allFibers, isCategory=True, isAll=True)
#getMetricsPhase(allFibers, isCategory=True, isAll=True)
#getMetricsCombined(allFibers)

#roiOps.chooseStandartROI(fibersA[0])
runReconstruction(CF_UB_Novembro)

