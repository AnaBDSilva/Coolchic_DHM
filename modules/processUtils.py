import sys, os, re
import numpy as np
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

########################################################################################################################################################################
## Pre-processing                                                                                                                                                     ##
########################################################################################################################################################################

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

def pre_process(recParams, IH):
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
    if recParams.decmt:
        IH = IH[::2, ::2]

    M, N = IH.shape
    print(IH.shape)

    Nin = min(M, N)
    Nmax = max(M, N)

    M2 = M // 2
    N2 = N // 2
    Nout2 = recParams.nout // 2
    Nin2 = Nin // 2

    # Case 1: crop to smaller square
    if recParams.nout <= Nin:
        print("Case 1")
        u1 = IH[M2 - Nout2 : M2 + Nout2,
                N2 - Nout2 : N2 + Nout2]

    # Case 2: crop to square then pad
    elif recParams.nout > Nin and recParams.nout < Nmax:
        print("Case 2")
        IH = IH[M2 - Nin2 : M2 + Nin2,
                N2 - Nin2 : N2 + Nin2]
        pad = (recParams.nout - Nin) // 2
        u1 = np.pad(IH,
                    ((pad, pad), (pad, pad)),
                    mode='constant')

    # Case 3: pad directly
    else:
        print("Case 3")
        pad_M = (recParams.nout - M) // 2
        pad_N = (recParams.nout - N) // 2
        u1 = np.pad(IH,
                    ((pad_M, pad_M), (pad_N, pad_N)),
                    mode='constant')
    return u1

########################################################################################################################################################################
## JSON and other files operations                                                                                                                                    ##
########################################################################################################################################################################

def save_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ficheiro {path} não encontrado. Execute o passo de referencia primeiro!"
        )
    with open(path, 'r') as f:
        return json.load(f)

def fiber_output_root(filenameHolo):
    prefix = re.split(r"[_.]+", filenameHolo)[0]
    return os.path.join("reconstructions", "fiber", f"{prefix}_glicerina", filenameHolo)