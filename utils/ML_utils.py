"""Shared helpers: error metrics, scaling, Nusselt number, reversal detection and plotting."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from scipy.stats import gaussian_kde
import torch

def Nu(data_uvt, kappa=1.0, H=1.0, deltaT=1, iy=1, iT=2):
    """
    data_uvt: [Nt, 3, Ny, Nx] with ch0=u, ch1=v, ch2=T
    kappa: thermal diffusivity; use 1.0 if already non-dimensional
    H: domain height (normally 1.0)
    deltaT: if None, estimated from the top and bottom wall temperatures
    iy: index of the vertical velocity channel
    iT: index of the temperature channel
    """
    v = data_uvt[:, iy]   # [Nt,Ny,Nx]
    T = data_uvt[:, iT]   # [Nt,Ny,Nx]
    Nt, Ny, Nx = T.shape

    dy = H / (Ny - 1)

    # estimate dT from the space- and time-averaged wall temperatures
    if deltaT is None:
        Tb = T[:, 0, :].mean()      # bottom
        Tt = T[:, -1, :].mean()     # top
        deltaT = Tb - Tt

    # dT/dy along axis=1 (the Ny dimension)
    dTdy = np.gradient(T, dy, axis=1)

    # total heat flux q = v*T - kappa*dT/dy
    q = v * T - kappa * dTdy

    # volume-averaged heat flux, per time step
    q_vol_t = q.mean(axis=(1, 2))  # [Nt]

    # pure conduction flux, used as the reference
    q_cond = kappa * deltaT / H

    Nu_t = q_vol_t / q_cond        # [Nt]
    Nu_mean = Nu_t.mean()

    # independent estimate from the wall gradients
    Nu_bottom_t = (-kappa * (T[:, 1, :] - T[:, 0, :]) / dy).mean(axis=1) / q_cond
    Nu_top_t    = (-kappa * (T[:, -1, :] - T[:, -2, :]) / dy).mean(axis=1) / q_cond

    return Nu_t, Nu_mean, Nu_bottom_t.mean(), Nu_top_t.mean()
    
# ---- reversal detection: zero crossings after smoothing ----
from scipy.signal import savgol_filter
def detect_reversals(series, win=21, poly=3, min_gap=10):
    """
    series: (Nt,) order parameter, e.g. angular momentum or a(t)
    win, poly: Savitzky-Golay smoothing window and polynomial order
    min_gap: minimum spacing between reversals, in frames, to suppress spikes
    returns: list of indices
    """
    s = savgol_filter(series, window_length=win, polyorder=poly, mode='interp') if len(series)>=win else series
    sign = np.sign(s)
    sign[sign==0] = 1  # avoid the unstable zero case
    flips = np.where(np.diff(sign)!=0)[0] + 1
    # drop crossings that are too close together
    keep = []
    last = -1e9
    for idx in flips:
        if idx - last >= min_gap:
            keep.append(idx)
            last = idx
    return keep, s

def cross_corr(l_p, Lhat, dt=1.0, path='/', plot=True):
    x = np.asarray(l_p, dtype=float)
    y = np.asarray(Lhat, dtype=float)
    n = min(len(x), len(y))
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    r = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * n)
    lags = np.arange(-n + 1, n) * dt

    if plot:
        plt.figure(dpi=300, figsize=(6, 3))
        plt.plot(lags, r, lw=1.5)
        plt.axvline(0, ls="--", lw=1)
        plt.axhline(0, ls="--", lw=1)
        peak = np.argmax(r)
        plt.xlabel("Lag (time)")
        plt.ylabel("Cross-corr")
        plt.title(f"peak r={r[peak]:.3f} at lag={lags[peak]:.3g}")
        plt.tight_layout()
        plt.savefig(path + 'Cross-corr.jpg')
    return lags, r

class MinMaxScaler:
    """
    Global min-max scaling for high-dimensional arrays, without sklearn.
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Set the target range of the scaler.
        Args:
            feature_range (tuple): target range (min, max), default (0, 1).
        """
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.original_shape = None

    def fit(self, data):
        """
        Compute the global minimum and maximum.
        Args:
            data (np.ndarray): high-dimensional array.
        Returns:
            self: this instance.
        """
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.original_shape = data.shape
        return self

    def transform(self, data):
        """
        Scale the data globally.
        Args:
            data (np.ndarray): high-dimensional array.
        Returns:
            np.ndarray: the scaled array.
        """
        if self.data_min is None or self.data_max is None:
            raise ValueError("Scaler has not been fitted yet. Call `fit` before `transform`.")

        data = data.reshape(-1)
        scaled_data = (data - self.data_min) / (self.data_max - self.data_min)
        scaled_data = scaled_data * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data.reshape(self.original_shape)

    def inverse_transform(self, data):
        """
        Invert the scaling.
        Args:
            data (np.ndarray): scaled high-dimensional array.
        Returns:
            np.ndarray: the array restored to its original scale.
        """
        if self.data_min is None or self.data_max is None:
            raise ValueError("Scaler has not been fitted yet. Call `fit` before `inverse_transform`.")

        data = data.reshape(-1)
        data = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        data = data * (self.data_max - self.data_min) + self.data_min
        return data.reshape(self.original_shape)

    def fit_transform(self, data):
        """
        Fit the scaler and scale the data.
        Args:
            data (np.ndarray): high-dimensional array.
        Returns:
            np.ndarray: the scaled array.
        """
        self.fit(data)
        return self.transform(data)
    
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    
class FieldScaler:
    """
    Scaling and inverse scaling of 4D field data, for numpy and torch.
    shape:
      - channels_last=False: [N, C, H, W]
      - channels_last=True : [N, H, W, C]
    mode: 'zscore' or 'minmax'
    """
    def __init__(self, mode='zscore', channels_last=False, eps=1e-8):
        assert mode in ('zscore', 'minmax')
        self.mode = mode
        self.channels_last = channels_last
        self.eps = eps
        self.loc = None   # per-channel mean or min, shape [C]
        self.scale = None # per-channel std or (max-min), shape [C]

    def fit(self, X):
        X = np.asarray(X) if not _is_torch(X) else X.detach().cpu().numpy()
        if self.channels_last:
            # [N, H, W, C]
            axes = (0, 1, 2)
            C = X.shape[-1]
        else:
            # [N, C, H, W]
            axes = (0, 2, 3)
            C = X.shape[1]
        if self.mode == 'zscore':
            loc = X.mean(axis=axes)                 # [C]
            scale = X.std(axis=axes)
        else:  # minmax
            loc = X.min(axis=axes)
            scale = X.max(axis=axes) - loc
        scale = np.maximum(scale, self.eps)
        self.loc = loc.astype(np.float32).reshape(C)
        self.scale = scale.astype(np.float32).reshape(C)
        return self

    def transform(self, X):
        assert self.loc is not None, "Call fit() first."
        return _affine(X, self.loc, self.scale, inv=False,
                       channels_last=self.channels_last)

    def inverse_transform(self, Xn):
        assert self.loc is not None, "Call fit() first."
        return _affine(Xn, self.loc, self.scale, inv=True,
                       channels_last=self.channels_last)


def _is_torch(x):
    return _HAS_TORCH and isinstance(x, torch.Tensor)

def _to_params_like(x, loc, scale):
    if _is_torch(x):
        device, dtype = x.device, x.dtype
        loc_t   = torch.as_tensor(loc,   device=device, dtype=dtype)
        scale_t = torch.as_tensor(scale, device=device, dtype=dtype)
        return loc_t, scale_t
    else:
        return np.asarray(loc), np.asarray(scale)

def _affine(X, loc, scale, inv=False, channels_last=False):
    loc_t, scale_t = _to_params_like(X, loc, scale)
    if channels_last:
        # [N, H, W, C]
        bshape = (1, 1, 1, -1)
    else:
        # [N, C, H, W]
        bshape = (1, -1, 1, 1)
    loc_t   = loc_t.reshape(bshape)
    scale_t = scale_t.reshape(bshape)
    if _is_torch(X):
        return X * scale_t + loc_t if inv else (X - loc_t) / scale_t
    else:
        return X * scale_t + loc_t if inv else (X - loc_t) / scale_t


def wrap_data(x_train, num_t):
    x_list = []
    lenth = len(x_train) - num_t
    for i in range(num_t):
        x_list.append(x_train[i:lenth+i+1])
    x_list = np.stack(x_list, axis=1)
    return x_list

def plot_lstm(encoded, train_size, time, rank, path, name='lstm dynamics', label=False):
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    c = ['blue','orange','green','red','purple','brown','hotpink','aqua']
    for i in range(rank):
        ax.plot(np.array(time[:train_size]), np.transpose(encoded[i,:train_size]), 
                 label='latent dynamics {}'.format(i+1), c=c[i])
        try:
            ax.plot(np.array(time[train_size:]), np.transpose(encoded[i,train_size:]), ls='-.', 
                    label='predicted dynamics {}'.format(i+1), c=c[i])
        except:
            pass
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)

    ax.annotate('Prediction', xy=(1.05*time[train_size], np.min(encoded)),size=16,)
    ax.vlines(time[train_size], np.min(encoded), np.max(encoded), colors='black', ls='-')
    if label:
        for i in range(rank):
            ax.text(time[0]-0.5, np.transpose(encoded[i,0]),str(i+1),size=16,color=c[i])
    
    plt.title(name,size=24)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + name + '.jpg')
    return
    
def compute_pdf_1d(U, grid=100, extend=0.0, renorm=True):
    """One-dimensional kernel-density PDF; returns the grid and the density."""
    data = np.asarray(U).ravel()
    kde = gaussian_kde(data)
    xmin, xmax = data.min(), data.max()
    span = xmax - xmin
    xmin -= extend*span
    xmax += extend*span
    x = np.linspace(xmin, xmax, grid)
    pdf = kde.evaluate(x)
    if renorm:
        pdf /= np.trapz(pdf, x)
    return np.stack([x, pdf])

def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return mean_squared_error(records_real, records_predict)
    else:
        return None 

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None
    
def get_NRMSE(y_true, y_pred, beta=1e-12):
    """Normalised RMSE: the relative l2 difference between prediction and reference.

    sqrt( sum (y_pred - y_true)^2 / (sum y_true^2 + beta) )
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return np.sqrt(np.sum((y_pred - y_true) ** 2) / (np.sum(y_true ** 2) + beta))

def reshape(inp, numberx, numbery):
    return np.reshape(inp, (-1, numberx, numbery, 1))

def shape(inp, numberx, numbery):
    return np.reshape(inp, (numberx*numbery, -1))

def compute_svd(X, svd_rank=0):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]
    return U, s, V

class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False