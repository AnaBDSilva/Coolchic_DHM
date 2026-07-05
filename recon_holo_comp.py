# %% -------------------------------------------------------------------------------

import Lloyd_DHM_PhaseRecROI_AUTO_COMP as autoComp
import modules.pda_metric as metrics
import modules.readResults as readR
import numpy as np
import re

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
#cannabis = ["cannabis_1g", "cannabis_2g", "cannabis_3g", "cannabis_4g"]
#fibersA = ["FiberA_1", "FiberA_1_lowerpart", "FiberA_2", "FiberA_3", "FiberA_4", "FiberA_5"]


#todas as fibras
allFibers = np.concatenate((fibersA, fibersB, cannabis))

#caminho para as amostras geral
samplePath = "/home/anabs/Documents/Uni/2ºSemestre/projeto/pyDHM-master/data/DataIn/Fibras_glicerina/"

#caminho especifico para cada tipo de amostra de fibras
fApath = samplePath + "FiberA_glicerina/"
fBpath = samplePath + "FiberB_glicerina/"
cnpath = samplePath + "cannabis_glicerina/"

#caminho de cada referencia, uma por categoria
refA = fApath + "FiberA_ref.mat"
refB = fBpath + "FiberB_ref.mat"
refcn = cnpath + "cannabis_refg.mat"

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
    u1, u1r, procHolo = autoComp.applyPreProcessing(imgHolo, imgHoloRef)

    #selecionar a ordem +1 e descartar o que é desnecessário
    ft_holo, x_min, y_min, holo_filter, ref_filter = autoComp.autoOrderSelect(filenameHolo, u1, u1r)

    #opcional - mostrar os hologramas
    autoComp.showInitialHologramState(ft_holo, imgHolo, procHolo, u1r, u1)

    #normalizar e salvar o hologramas para futura compressão
    autoComp.normalizeAndSaveHolo(holo_filter, filenameHolo)

    #realizar a prpogação em varios planos z para encontrar o melhor
    zextr, holo_filter = autoComp.propagationOnZ(holo_filter, x_min, y_min, filenameHolo, True)

    #refocagem no plano otimo z e separação em fase e amplitude
    phase, intensity, phase_ref, ref_intensity = autoComp.refocusOptPlaneChooseROI(holo_filter, zextr, filenameHolo, ref_filter)

    #opcional - mostrar a fase e a intensidade
    xxc, yyc = autoComp.showIntensityPhase(phase, intensity, phase_ref, ref_intensity)

    #fazer o phase unwrapping
    phase_dif = autoComp.phaseUnwrapping(phase, phase_ref, filenameHolo)

    mask_bg = autoComp.grabcutSegmentation(phase_dif, filenameHolo, False)

    a, b, c = autoComp.createAjdustedPlane(phase_dif, mask_bg)

    phase_corr = autoComp.applyPlane(phase_dif, a, b, c)

    #optional - mostrar a diferença corrigida
    autoComp.showCorrectedDifference(phase_dif, phase_corr, xxc, yyc)

    factor_k0, X, Y = autoComp.showPhaseUnwrapRes(xxc, yyc, phase_corr)

    phase_denoised = autoComp.phaseDenoising(phase_corr)

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
    factor_k0, X, Y = autoComp.showPhaseUnwrapRes(xxc, yyc, phase_corr)

    phase_denoised = autoComp.phaseDenoising(phase_corr)

    autoComp.showDenoisingRes(phase_denoised, factor_k0, X, Y, xxc, yyc, fiber, lmbda, True)

#made to run auto for all fibers or some specific category
def runReconstruction(fiberGroup):
    #run everythin again just to get the radius values for reference
    for fiber in fiberGroup:
        pathSplit = re.split(r"[_.]+", fiber)
        fiberType = pathSplit[0]

        match fiberType:
            case "FiberA":
                path = fApath
                ref = refA
            case "FiberB":
                path = fBpath
                ref = refB
            case "cannabis":
                path = cnpath
                ref = refcn
            case _:
                print("Tipo de fibra desconhecido.")
                exit

        #test(fiber, path, ref)
        numericalReconstructionAuto(fiber, path, ref)

def runCoolChic(fiberGroup):
    for fiber in fiberGroup:
        for lmb in lambdas:
            numericalReconstructionAuto_CoolChic(fiber, lmb)
            #numericalReconstructionAuto(fiber, path, ref)
        runMetric(fiber)

def runMetric(fiber):
    metrics.test_metric(fiber, isPhaseWrapped=True,  metric='psnr')  # PDA-PSNR
    metrics.test_metric(fiber, isPhaseWrapped=False, metric='psnr')  # PSNR
    metrics.test_metric(fiber, isPhaseWrapped=True,  metric='ssim')  # PDA-SSIM
    metrics.test_metric(fiber, isPhaseWrapped=False, metric='ssim')  # SSIM
    #readR.getRDGraphReIm(fiber)
    #readR.getRDGraphComplex(fiber)

def runMetricForAll(allFibers):
    metrics.test_metric(allFibers, isPhaseWrapped=True,  metric='psnr', isAll=True)
    metrics.test_metric(allFibers, isPhaseWrapped=True,  metric='ssim', isAll=True)
    metrics.test_metric(allFibers, isPhaseWrapped=False, metric='psnr', isAll=True)
    metrics.test_metric(allFibers, isPhaseWrapped=False, metric='ssim', isAll=True)

    readR.getAllFibersGraphComplexMetric(allFibers, metric='psnr')
    readR.getAllFibersGraphComplexMetric(allFibers, metric='ssim')

#########
#runReconstruction(allFibers)
#runCoolChic(fibersB)

arr2 = ["FiberB_0", "FiberB_1"]
arr3 = [x for x in fibersB if x not in arr2]

metrics.applyCorrelationMetrics()

