
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, structure_tensor
import os
import modules.processUtils as procUtils

def compute_descriptors(phase, mask=None, wavelength=632.8e-9, dx=2.2e-6 / 12.8,
                        win=7, bins=64, glcm_levels=32, tensor_sigma=2.0):
    phase = np.asarray(phase, dtype=float)
    #if there isnt a mask create one that covers the whole image
    if mask is None:
        mask = np.ones(phase.shape, bool)
    dA = dx * dx
    out = {}

    # ---- OPD (eq. 10) ----
    opd = wavelength / (2 * np.pi) * phase
    v = opd[mask]

    # ---- 4.1 Optical content ----
    out["mean_OPD"] = v.mean()
    out["std_OPD"] = v.std()
    out["CV"] = out["std_OPD"] / out["mean_OPD"]
    out["IOPD"] = v.sum() * dA                

    # ---- 4.2 Roughness / complexity ----
    gy, gx = np.gradient(opd, dx)                       # d/dy, d/dx
    G = np.hypot(gx, gy)
    out["mean_gradient"] = G[mask].mean()

    ##
    #lap = ndi.laplace(opd)
    lap = ndi.laplace(opd) / dx ** 2
    out["laplacian_variance"] = lap[mask].var()

    ##
    sa = np.sqrt(1 + gx ** 2 + gy ** 2)
    out["surface_area_ratio"] = sa[mask].mean()

    # ---- 4.3 Texture ----
    # local variance - the variance of a small windows around each pixel != from global variance
    m1 = ndi.uniform_filter(opd, win)
    m2 = ndi.uniform_filter(opd ** 2, win)
    local_var = np.clip(m2 - m1 ** 2, 0, None)
    out["mean_local_variance"] = local_var[mask].mean()

    #turns OPD values into probabilities
    p, _ = np.histogram(v, bins=bins)
    p = p / p.sum()
    p = p[p > 0]
    out["entropy"] = -np.sum(p * np.log(p))
    #can i use this functions from scipy or does it differ from the calculations made in the paper
    out["skewness"] = skew(v)
    out["kurtosis"] = kurtosis(v, fisher=False)

    ## - still need revision below
    lo, hi = v.min(), v.max()
    q = np.zeros(phase.shape, np.uint8)
    q[mask] = 1 + np.clip(((v - lo) / (hi - lo + 1e-30) * (glcm_levels - 2)).astype(int), 0, glcm_levels - 2)
    P = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=glcm_levels, symmetric=True, normed=False)[1:, 1:]
    P = P / P.sum(axis=(0, 1), keepdims=True)
    for name in ("contrast", "homogeneity", "energy", "correlation"):
        out["glcm_" + name] = graycoprops(P, name).mean()   # averaged over 4 angles

    # ---- 4.4 Anisotropy / spatial organisation ----
    Axx, Axy, Ayy = structure_tensor(opd, sigma=tensor_sigma, order="rc")
    jxx, jxy, jyy = Axx[mask].mean(), Axy[mask].mean(), Ayy[mask].mean()
    l1, l2 = np.linalg.eigvalsh([[jxx, jxy], [jxy, jyy]])[::-1]   # l1 >= l2
    out["coherence"] = (l1 - l2) / (l1 + l2 + 1e-30)                # eq. 31

    # correlation length: autocorrelation (via FFT) of the masked, zero-mean OPD
    f = np.where(mask, opd - v.mean(), 0.0)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(f)) ** 2).real)
    ac /= ac.max()
    r_prof = _radial_mean(ac)
    below = np.where(r_prof < 1 / np.e)[0]
    out["correlation_length"] = (below[0] if below.size else len(r_prof)) * dx   # 1/e decay, metres

    # PSD: radial profile and dominant spatial frequency (DC excluded)
    psd = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    psd_prof = _radial_mean(psd)
    k = np.argmax(psd_prof[1:]) + 1
    out["dominant_spatial_freq"] = k / (min(phase.shape) * dx)      # cycles / metre
    out["psd_radial_profile"] = psd_prof                            # for plotting

    return out


def _radial_mean(a):
    """Radial average around the array centre."""
    cy, cx = np.array(a.shape) // 2
    y, x = np.indices(a.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    n = min(cy, cx)
    s = np.bincount(r.ravel(), a.ravel())
    c = np.bincount(r.ravel())
    return (s / np.maximum(c, 1))[:n]



def runQuantMethods(recParams, phase_corr, mask_fiber, filenameHolo, tag="phase_corr"):
    '''
    computes the phase descriptors for one fiber and saves them
    '''
    out_dir = os.path.join(procUtils.fiber_output_root(filenameHolo), "quant_methods")
    os.makedirs(out_dir, exist_ok=True)

    # because sometimes we change the value to see better in the 3d graphs
    phase = -phase_corr
    #only use what is true
    mask = np.asarray(mask_fiber).astype(bool)

    # pixel size at the SAMPLE plane = camera pixel / magnification
    d = compute_descriptors(phase, mask=mask,
                            wavelength=recParams.wavelength,
                            dx=recParams.dx / recParams.mtransv)

    psd = d.pop("psd_radial_profile")               
    scalars = {k: float(v) for k, v in d.items()}  

    procUtils.save_json(os.path.join(out_dir, f"descriptors_{tag}.json"), scalars)
    np.save(os.path.join(out_dir, f"psd_radial_{tag}.npy"), psd)
    np.save(os.path.join(out_dir, "mask_fiber.npy"), mask)   

    print(f"Descritores guardados em {out_dir}")
    return scalars

