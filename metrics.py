import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class Bandwise:
    def __init__(self, index_fn, data_range=1.0):
        """
        Applica una metrica banda per banda (su immagini NumPy).

        Args:
            index_fn: funzione di confronto tra due immagini 2D (es. SSIM, PSNR)
            data_range: range massimo dei valori (richiesto per float)
        """
        self.index_fn = index_fn
        self.data_range = data_range

    def __call__(self, X, Y):
        assert X.shape == Y.shape, "X e Y devono avere la stessa shape"
        assert X.ndim == 3, "Input deve avere shape [C, H, W]"

        C = X.shape[0]
        bwindex = []
        for ch in range(C):
            x = X[ch, :, :]
            y = Y[ch, :, :]
            index = self.index_fn(x, y, data_range=self.data_range)
            bwindex.append(index)
        return bwindex


def sam(X, Y, eps=1e-8, degrees=False):
    """
    Calcola la SAM (Spectral Angle Mapper) media tra due cubi spettrali.
    Input:
        - X, Y: numpy array con shape [C, H, W] o [B, C, H, W]
        - degrees: se True, restituisce il valore in gradi (altrimenti radianti)
    Output:
        - SAM medio tra X e Y
    """
    assert X.shape == Y.shape, "X e Y devono avere la stessa shape"

    if X.ndim == 3:
        dot = np.sum(X * Y, axis=0)
        norm_X = np.sqrt(np.sum(X ** 2, axis=0))
        norm_Y = np.sqrt(np.sum(Y ** 2, axis=0))
    elif X.ndim == 4:
        dot = np.sum(X * Y, axis=1)
        norm_X = np.sqrt(np.sum(X ** 2, axis=1))
        norm_Y = np.sqrt(np.sum(Y ** 2, axis=1))
    else:
        raise ValueError("Input deve avere 3 o 4 dimensioni (C,H,W) o (B,C,H,W)")

    denominator = norm_X * norm_Y + eps
    cos_sim = (dot + eps) / denominator
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle = np.arccos(cos_sim)

    if degrees:
        angle = np.degrees(angle)

    return np.mean(angle)


# === Istanze pronte delle metriche banda per banda
cal_bwssim = Bandwise(structural_similarity, data_range=1.0)
cal_bwpsnr = Bandwise(peak_signal_noise_ratio, data_range=1.0)

def psnr(X, Y):
    return np.mean(cal_bwpsnr(X, Y))

def ssim(X, Y):
    return np.mean(cal_bwssim(X, Y))


# === Funzione unica che restituisce tutte le metriche
def evaluate_metrics(X, Y, degrees=False):
    """
    Restituisce le metriche: SAM, PSNR, SSIM

    Args:
        X, Y: numpy array con shape [C, H, W]
        degrees: se True, restituisce SAM in gradi

    Returns:
        dict con chiavi 'SAM', 'PSNR', 'SSIM'
    """
    return {
        'SAM': sam(X, Y, degrees=degrees),
        'PSNR': psnr(X, Y),
        'SSIM': ssim(X, Y)
    }
