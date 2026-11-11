from modules.processUtils import fiber_output_root
import numpy as np
import cv2, json, os

########################################################################################################################################################################
## Segmentation                                                                                                                                                       ##
########################################################################################################################################################################

def fit_plane(x, y, z):
    """
    Ajusta plano z = ax + by + c por mínimos quadrados
    """
    X = np.column_stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
    Z = z.flatten()

    coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    a, b, c = coeffs

    return a, b, c

def segment_fiber_grabcut_seeds(image, scale=0.3):
    """
    EN: GrabCut segmentation using manual seeds (foreground + background)
    PT: Segmentação GrabCut usando seleção manual de sementes

    Returns:
        mask_fiber
    """

    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    h, w = img.shape
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    # máscara inicial (unknown)
    mask = np.full(resized.shape, cv2.GC_PR_BGD, dtype=np.uint8)

    print("Select fiber region / Seleciona região da fibra")
    roi_fg = cv2.selectROI("Foreground", resized, False)
    x, y, rw, rh = roi_fg
    mask[y:y+rh, x:x+rw] = cv2.GC_FGD

    print("Select background region / Seleciona região de fundo")
    roi_bg = cv2.selectROI("Background", resized, False)
    x, y, rw, rh = roi_bg
    mask[y:y+rh, x:x+rw] = cv2.GC_BGD

    cv2.destroyAllWindows()

    # redimensionar máscara para original
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img_rgb, mask_full, None,
                bgdModel, fgdModel, 5,
                mode=cv2.GC_INIT_WITH_MASK)

    mask_fiber = np.where(
        (mask_full == cv2.GC_FGD) | (mask_full == cv2.GC_PR_FGD),
        1, 0).astype(np.uint8)

    return mask_fiber

def segment_fiber_grabcut(image, fiber_name, applyCoolChic, saveReference, scale=0.3):
    """
    EN: GrabCut segmentation with screen scaling (large images fix)
    PT: Segmentação GrabCut com ajuste ao ecrã (imagens grandes)

    Parameters:
        image : input image
        scale : scaling factor for display

    Returns:
        mask_fiber (1 = fibra, 0 = fundo)
    """

    # ------------------------------------------
    # Normalize image for visualization
    # Normalizar imagem para visualização
    # ------------------------------------------
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)

    # ------------------------------------------
    # Resize for display
    # Redimensionar para caber no ecrã
    # ------------------------------------------
    h, w = img_norm.shape
    resized = cv2.resize(img_norm, (int(w*scale), int(h*scale)))

    print("Select ROI containing fiber / Seleciona região com a fibra")

    base = fiber_output_root(fiber_name)
    filepath = os.path.join(base, "reference")
    if applyCoolChic:
        jsonPathGrabcut = os.path.join(filepath, "holo_order_infoROI_grabcut.json")
        try:
            with open(jsonPathGrabcut, 'r') as file:
                info_grabcut = json.load(file)
            
            rect = (info_grabcut["x"], info_grabcut["y"], info_grabcut["rw"], info_grabcut["rh"])
        except FileNotFoundError:
            print(f"ERRO: Ficheiro {jsonPathGrabcut} não encontrado. Execute o passo de referencia primeiro!")
    else:
        roi = cv2.selectROI("GrabCut ROI", resized, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, rw, rh = roi

    # ------------------------------------------
    # Convert coordinates back to original image
    # Converter coordenadas para imagem original
    # ------------------------------------------
        x = int(x / scale)
        y = int(y / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

        rect = (x, y, rw, rh)

    if saveReference:
        info_grabcut = {
            "x": rect[0],
            "y": rect[1],
            "rw": rect[2],
            "rh": rect[3]
        }
        with open(os.path.join(filepath, "holo_order_infoROI_grabcut.json"), 'w') as file:
            json.dump(info_grabcut, file, indent=4)

    # ------------------------------------------
    # Prepare image for GrabCut
    # Preparar imagem para GrabCut
    # ------------------------------------------
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    mask = np.zeros(img_norm.shape, np.uint8)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # ------------------------------------------
    # Apply GrabCut on full-resolution image
    # Aplicar GrabCut na imagem original
    # ------------------------------------------
    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # ------------------------------------------
    # Extract final mask
    # Extrair máscara final
    # ------------------------------------------
    mask_fiber = np.where((mask==2) | (mask==0), 0, 1).astype(np.uint8)

    return mask_fiber
