import importlib
from pyDHM import utils
importlib.reload(utils)


from pyDHM import numProp
importlib.reload(numProp)
import numpy as np

import numpy as np
import cv2
import json


# ***********************************************
# Begin function definitions:
# ***********************************************
def select_roi_opencv(image, scale=0.3):
    """
    Manual ROI selection with scaling (fix for large images).

    Parameters:
        image : input image
        scale : scaling factor for display (e.g. 0.3)

    Returns:
        rmin, rmax, cmin, cmax (original image coords)
    """
    img = image.copy()

    # Normalizar para display
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)

    # Resize apenas para visualização
    h, w = img_norm.shape
    resized = cv2.resize(img_norm, (int(w*scale), int(h*scale)))

    # Seleção ROI
    roi = cv2.selectROI("Select ROI", resized, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi

    # Converter coords para imagem original
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)

    rmin, rmax = y, y + rh
    cmin, cmax = x, x + rw

    return rmin, rmax, cmin, cmax

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

    filepath = f"reconstructions/fiber/{fiber_name}/reference/holo_order"
    if applyCoolChic:
        jsonPathGrabcut = f"{filepath}_infoROI_grabcut.json"
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
        with open(f"{filepath}_infoROI_grabcut.json", 'w') as file:
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

def calc_zobj(zi, t, M, Mag):
    fl = t/M

    zobj = zi*fl/(Mag*(zi + fl*Mag))
    return zobj

def variance_norm(imagen):
    """
    Focusing function based on normalized variance of an image.
    """
    y, x = imagen.shape

    media = np.mean(imagen)

    resultado = np.sum((imagen - media) ** 2) / (x * y * media)

    return resultado

def Rect_Window(Nx, Ny, eta, w):
    """
    Rectangular apodization Tukey window
    
    Parameters
    ----------
    Nx, Ny : int
        Window size in x and y
    eta : float
        Internal radius (0 < eta < 1 - w)
    w : float
        Cosine decay width (w < 1 - eta)

    Returns
    -------
    window : 2D numpy array
    """

    # Equivalent of MATLAB linspace(-1,1,N)
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)

    # MATLAB meshgrid(x,y) -> NumPy default indexing='xy' matches it
    X, Y = np.meshgrid(x, y)

    # Masks
    mask1 = np.abs(X / (w + eta)) > 1
    mask2 = np.abs(X / eta) < 1
    mask3 = np.abs(Y / (w + eta)) > 1
    mask4 = np.abs(Y / eta) < 1

    # Window along X
    window1 = np.cos((np.pi / (2 * w)) * (np.abs(X) - eta))
    window1[mask2] = 1
    window1[mask1] = 0

    # Window along Y
    window2 = np.cos((np.pi / (2 * w)) * (np.abs(Y) - eta))
    window2[mask4] = 1
    window2[mask3] = 0

    # Element-wise multiplication (MATLAB: .*)
    window = window1 * window2

    return window

def pre_process(IH, Nout, isDecimate):
    """
    Parameters
    ----------
    IH : 2D numpy array
        Input hologram
    Nout : int
        Desired output size (square)
    isDecimate : bool
        If True, subsample by factor 2
    """

    # Cut original hologram (decimation)
    if isDecimate:
        IH = IH[::2, ::2]

    M, N = IH.shape
    print(IH.shape)

    Nin = min(M, N)
    Nmax = max(M, N)

    M2 = M // 2
    N2 = N // 2
    Nout2 = Nout // 2
    Nin2 = Nin // 2

    # Case 1: crop to smaller square
    if Nout <= Nin:
        print("Case 1")
        u1 = IH[M2 - Nout2 : M2 + Nout2,
                N2 - Nout2 : N2 + Nout2]

    # Case 2: crop to square then pad
    elif Nout > Nin and Nout < Nmax:
        print("Case 2")
        IH = IH[M2 - Nin2 : M2 + Nin2,
                N2 - Nin2 : N2 + Nin2]
        pad = (Nout - Nin) // 2
        u1 = np.pad(IH,
                    ((pad, pad), (pad, pad)),
                    mode='constant')

    # Case 3: pad directly
    else:
        print("Case 3")
        pad_M = (Nout - M) // 2
        pad_N = (Nout - N) // 2
        u1 = np.pad(IH,
                    ((pad_M, pad_M), (pad_N, pad_N)),
                    mode='constant')
    return u1
