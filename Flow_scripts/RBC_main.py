"""TheFlowPG-DL: entry point for the POD, TPM and SRM stages and their evaluation."""
import sys
sys.path.append('../utils')
import numpy as np
import matplotlib.pyplot as plt
import os
import time as TT
import ML_utils as ut
import argparse
import torch
import datetime
import pandas as pd
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Configuration
# =============================================================================
# TheFlowPG-DL factorises the prediction into three built-in modules, applied
# consecutively. A substring of the method name (-m) selects which one runs:
#
#   'POD'   Proper Orthogonal Decomposition. Reduces the dimensionality of the
#           high-dimensional state and identifies the energy-dominating LSC
#           structure on a reduced-order coordinate. Writes the LSC field
#           (S_pred.npy) and the POD coefficients (V_pred.npy), which are the
#           arrays kept under data/ as *_lsc_*.npy and *_V_*.npy.
#   'DLMD'  Temporal Prediction Module. Advances the reduced-order coordinate,
#           predicting the Ra-Pr dependent reversal dynamics of the LSC, and
#           writes the predicted LSC field as its own S_pred.npy.
#   'rec'   Spatial Reconstruction Module. Projects the predicted LSC
#           coordinate back onto the full-state space, reconstructing the
#           small-scale velocity and temperature structures discarded by the
#           dimensionality reduction.
#
#   Evaluating one case takes two runs, TPM first and SRM second:
#
#   python RBC_main.py -m DLMD-CondLSTMrff-attn-Transfer -c RBC_all_Ra10e8
#   python RBC_main.py -m CondUNetrec-gn-Nu-div-Transfer -c RBC_all_Ra10e8
#
# Further substrings of the method name:
#   'Transfer'   load the pretrained weights in ../checkpoints instead of training
#   'ablation'   deactivate the FFE and TWA submodules of the TPM (need to be trained)
#   'div', 'Nu'  enable the divergence-free / heat-flux constraint of the SRM
parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='method', type=str,
                    default='DLMD-CondLSTMrff-attn-Transfer',
                    help='Stage and run mode, selected by substring; see above')
parser.add_argument('-c', dest='case', type=str, default='RBC_all_Ra10e8',
                    help="Case to evaluate: RBC_all_Ra10e8, RBC_all_Ra10e8Pr2, "
                         "RBC_all_Ra10e7, RBC_all_Ra5p10e7 or RBC_all_Ra10e8Pr3p2. "
                         "Append '_v' or '_T' to report the v or T component "
                         "instead of u")
parser.add_argument('-r', dest='rank', type=int, default=1,
                    help='POD truncation rank')
parser.add_argument('-d', dest='lambda_div', type=float, default=1e-2,
                    help='Weight of the divergence-free loss term of the SRM')
parser.add_argument('-n', dest='lambda_Nu', type=float, default=1e-2,
                    help='Weight of the heat-flux loss term of the SRM')
args = parser.parse_args()

method = args.method
case = args.case
rank = args.rank
lambda_div = args.lambda_div
lambda_Nu = args.lambda_Nu

backend = 'Torch'
activ = 'relu'
offset = 5              # length of the input window, in time steps
channel = 3             # (u, v, T)
epoch = 3000            # maximum number of epochs
batch_size = 128 if 'rec' in method else 32     # SRM / TPM
patience = 500          # early stopping patience, in epochs
test_size = 0.5         # first half reconstruction, second half prediction
load = False
normalize = False
c_fr = 0
seed = 42

normalize_lstm = True if 'DLMD' in method else False
normalize_uvt = True if 'rec' in method else False
Transfer = True if 'Transfer' in method else False
residual = True if 'res' in method else False

# Pretrained weights and output directories.
ckpt_root = '../checkpoints/'
result_root = '../results/'
A_path = ckpt_root + 'tpm/'
B_path = ckpt_root + 'srm/'

# Method name used for the TPM.
TPM_METHOD = 'DLMD-CondLSTMrff-attn-Transfer'

# The five flow regimes.
regimes = [
    dict(file='RBC_Ra10e8_50000_interpolate_128x128.npy', Ra=1e8, Pr=4.3, idx=0, marker='train',
         pod_file='RBC_Ra10e8_50000_lsc_128x128.npy',
         V_file='RBC_Ra10e8_50000_V_128x128.npy'),
    dict(file='RBC_Ra10e8Pr2_50000_interpolate_128x128.npy', Ra=1e8, Pr=2.0, idx=1, marker='train',
         pod_file='RBC_Ra10e8Pr2_50000_lsc_128x128.npy',
         V_file='RBC_Ra10e8Pr2_50000_V_128x128.npy'),
    dict(file='RBC_Ra10e7_50000_interpolate_128x128.npy', Ra=1e7, Pr=4.3, idx=2, marker='train',
         pod_file='RBC_Ra10e7_50000_lsc_128x128.npy',
         V_file='RBC_Ra10e7_50000_V_128x128.npy'),
    dict(file='RBC_Ra5p10e7_50000_interpolate_128x128.npy', Ra=5*1e7, Pr=4.3, idx=3, marker='predict',
         pod_file='RBC_Ra5p10e7_50000_lsc_128x128.npy',
         V_file='RBC_Ra5p10e7_50000_V_128x128.npy'),
    dict(file='RBC_Ra10e8Pr3p2_50000_interpolate_128x128.npy', Ra=1e8, Pr=3.2, idx=4, marker='predict',
         pod_file='RBC_Ra10e8Pr3p2_50000_lsc_128x128.npy',
         V_file='RBC_Ra10e8Pr3p2_50000_V_128x128.npy'),
]

valid = True
writeData = True
st = 0

os.environ['PYTHONHASHSEED'] = str(seed)
labels = ["$u_{x}$", "$u_{y}$", "$T$"]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

set_seed(42) 
sca = None
if 'v' in case:
    ob = 'v'
    ch_slice = slice(1, 2)
elif 'T' in case:
    ob = 'T'
    ch_slice = slice(2, 3)
else:
    ob = 'u'
    ch_slice = slice(0, 1)
    
if 'Ra10e7' in case:
    pos = next((i for i, d in enumerate(regimes) if "Ra10e7" in d["file"]), -1)
elif 'Ra5p10e7' in case:
    pos = next((i for i, d in enumerate(regimes) if "Ra5p10e7" in d["file"]), -1)
elif 'Pr2' in case:
    pos = next((i for i, d in enumerate(regimes) if "Pr2" in d["file"]), -1)
elif 'Pr3p2' in case:
    pos = next((i for i, d in enumerate(regimes) if "Pr3p2" in d["file"]), -1)
else:
    pos = next((i for i, d in enumerate(regimes) if "Ra10e8" in d["file"]), -1)
    
idx_pred = pos

idx_train = [i for i, d in enumerate(regimes) if "train" in d["marker"]]
# =============================================================================
# Load data
# =============================================================================
if 'RBC' in case:
    rank_st = 0
    numberx, numbery = 128, 128
    kappa = 4.822428e-05
    
    x = np.linspace(0, 1, numberx)
    y = np.linspace(0, 1, numbery)
    
    if 'Ra10e7' in case:
        fileName = 'RBC_Ra10e7_50000_interpolate_128x128.npy'
        kappa = 1.524986e-04
    elif 'Ra5p10e7' in case:
        fileName = 'RBC_Ra5p10e7_50000_interpolate_128x128.npy'
        kappa = 6.819943e-05
    elif 'Pr2' in case:
        fileName = 'RBC_Ra10e8Pr2_50000_interpolate_128x128.npy'
        kappa = 7.0710678e-05
    elif 'Pr3p2' in case:
        fileName = 'RBC_Ra10e8Pr3p2_50000_interpolate_128x128.npy'
        kappa = 5.59017e-05
    else:
        fileName = 'RBC_Ra10e8_50000_interpolate_128x128.npy'
        kappa = 4.822428e-05
    dltT = 50
    
    full_data = np.load('../data/' +fileName)
    startT = 0
    st = 40

    full_data = np.float32(full_data)
    data = full_data[:,ch_slice]  
            
    S_col_full = len(data)
    S_col = int((1-test_size)*S_col_full)
    
    S_full = np.reshape(data, (data.shape[0], -1))
    S_full = np.float32(np.transpose(S_full))
    S_col_full = S_full.shape[-1]

trainT = startT + S_col * dltT
endT = startT + S_col_full * dltT

train_size = int((trainT - startT)/dltT)


maxz = float('%.2f' % np.max(S_full))
minz = float('%.2f' % np.min(S_full))
levels = np.linspace(minz, maxz, 11)
for i in range(levels.size):
    levels[i] = float('%.2f' % levels[i])
 
xx, yy = x,y
x,y = np.meshgrid(x,y)

S_mean = S_full.mean(axis=-1)[:,np.newaxis]

time = []
time_str = []
for i in np.arange(startT, endT, dltT):
    i = float('%.3f' % i)
    time.append(i)
    time_str.append(str('%.3f' % i))
time = time[:S_col_full]
    
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

snapshots = list(S_full.T)
offlineStart = TT.time()

def rr(data):
    # data: [N,H,W], or a [N,1,H,W] / [N,H,W] slice
    S_full = np.reshape(data, (data.shape[0], -1))
    S_full = np.float32(np.transpose(S_full))  # [HW, N]
    return S_full

def recon_pod(USV, S_mean):
    r = USV[1].shape[0]
    fluctuation = np.dot(USV[0][:, :r], np.dot(USV[1][:r, :r], USV[2][:r, :]))
    return S_mean + fluctuation
    
# =============================================================================
# Output directory and normalisation
# =============================================================================
projName = '{}_{}_{}_{}'.format(method, backend, case, ob)


if 'POD' in method:
    normalize = False
       
if normalize:
    feature_range = (0, 1)
    scaler = ut.MinMaxScaler(feature_range=feature_range)
    scaler.fit(S_full)
    S_full = scaler.transform(S_full)

    
path = result_root + 'ML_' + projName + '_' + activ + '/'

if not os.path.exists(path):
    os.makedirs(path) 
writepath = path + '/output_data'
if not os.path.exists(writepath):
    os.makedirs(writepath) 
    

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

log_file_name = path + 'log.txt'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

print('Task starts')
print('data path: {}'.format(path))
print('method:', method, 'rank:',
    rank, 'case:',
    case, 'batch_size:',
    batch_size, 'epoch:',
    epoch, 'load:',
    load, 'activ:',
    activ, 'offset:',
    offset, 'normalize:',
    normalize, 'backend:',
    backend, 'c_fr:',
    c_fr,
)


# =============================================================================
# Proper Orthogonal Decomposition
# =============================================================================
# Reduces the dimensionality of the selected component and keeps the leading
# `rank` modes, which carry the energy-dominating LSC structure. Writes the
# reconstructed LSC field (S_pred.npy) and the POD coefficients (V_pred.npy);
# over the five regimes these form data/*_lsc_*.npy and data/*_V_*.npy, on
# which the TPM and the SRM are trained.

if 'POD' in method and 'DLMD' not in method:
    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()
    S_full_sub = np.subtract(S_full, S_mean)
    m = S_full_sub.shape[0]
    U, s, V = np.linalg.svd(S_full_sub, full_matrices=False)
    
    __, sigma, __ = ut.compute_svd(S_full_sub[:, :S_col-1], svd_rank=-1)
    cumulative_energy = np.cumsum(sigma**2 / (sigma**2).sum())
    np.savetxt(path + '/eig_cum.csv', cumulative_energy, delimiter=',')
    
    U = U[:,:rank]
    s = s[:rank]
    V = V[:rank,:]
    
    S = np.diag(s)
    V_pred = V
        
    if rank == m:
        fluctuation = np.dot(U, np.dot(S, V_pred))
        fluctuation_true = np.dot(U, np.dot(S, V))
    else:
        fluctuation = np.dot(U[:, :rank], np.dot(S[:rank, :rank], V_pred[:rank, :]))
        fluctuation_true = np.dot(U[:, :rank], np.dot(S[:rank, :rank], V[:rank, :]))
    
    plot_r = min(rank, 8)
    
    ut.plot_lstm(V[:plot_r], train_size, time, plot_r, path, name='POD Coe true', label=True)
    ut.plot_lstm(V_pred[:plot_r], train_size, time, plot_r, path, name='POD Coe pred', label=True)

    S_pred = S_mean + fluctuation
    S_pod = S_mean + fluctuation_true  
    
    np.save(path+'/S_pred', S_pred)
    np.save(path+'/V_pred', V_pred)
    
    snapshots2 = list(np.transpose(S_pred))
    onlineEnd = TT.perf_counter()
    
# =============================================================================
# Spatial Reconstruction Module
# =============================================================================
# Projects the predicted LSC coordinate back onto the full-state space,
# recovering the small-scale velocity and temperature structures that the
# dimensionality reduction discards.
elif 'rec' in method and 'all' in case:
    # uvt: DNS reference; pod: POD reconstruction; lsc: LSC field predicted by the TPM
    data_uvt_list = []
    data_pod_list = []
    V_list      = []

    Ra_list = []
    Pr_list = []

    # ---------------------------
    # read each case: DNS fields, stored LSC fields and stored POD coefficients
    # ---------------------------
    for i_reg, reg in enumerate(regimes):
        # DNS snapshots
        full_data = np.load('../data/' + reg["file"]).astype(np.float32)   # [N,3,H,W]
        data_uvt_reg = full_data[:, :channel]                              # [N,channel,H,W]

        if i_reg == idx_pred:
            # built exactly as the TPM builds its own output directory, so the
            # two stages agree for every case and component
            tpm_out = result_root + 'ML_{}_{}_{}_{}_{}/'.format(
                TPM_METHOD, backend, case, ob, activ)
            if not os.path.exists(tpm_out + 'S_pred.npy'):
                raise FileNotFoundError(
                    "{}S_pred.npy is missing. Run the TPM stage for this case "
                    "first: python RBC_main.py -m {} -c {}".format(
                        tpm_out, TPM_METHOD, case))
            data_lsc_pred = np.load(tpm_out + 'S_pred.npy').astype(np.float32)
            data_lsc_pred = data_lsc_pred[:, :channel]

        Ra_list.append(reg['Ra'])
        Pr_list.append(reg['Pr'])
        
        pod_file = reg.get("pod_file", None)
        if pod_file is None:
            raise ValueError("Each regime must provide 'pod_file' in regimes dict.")
        data_pod_reg = np.load('../data/' + pod_file).astype(np.float32)
        if data_pod_reg.ndim == 4:
            data_pod_reg = data_pod_reg[:, :channel]                       # [N,channel,H,W]
        else:
            raise ValueError(f"Unexpected lsc shape: {data_pod_reg.shape}")

        V_file = reg.get("V_file", None)
        if V_file is None:
            raise ValueError("Each regime must provide 'V_file' in regimes dict.")
        V_reg = np.load('../data/' + V_file).astype(np.float32)

        # normalise to coe_reg: [N, channel*rank]
        if V_reg.shape[0] == channel * rank:
            coe_reg = V_reg.T                                              # [N, dims]
        elif V_reg.shape[1] == channel * rank:
            coe_reg = V_reg                                                # [N, dims]
        else:
            raise ValueError(f"Unexpected V shape: {V_reg.shape}, expect [dims,N] or [N,dims], dims={channel*rank}")

        Nmin = min(data_uvt_reg.shape[0], data_pod_reg.shape[0], coe_reg.shape[0])
        if i_reg == idx_pred:
            Nmin = min(Nmin, data_lsc_pred.shape[0])
            data_lsc_pred = data_lsc_pred[:Nmin]
        data_uvt_reg = data_uvt_reg[:Nmin]
        data_pod_reg = data_pod_reg[:Nmin]
        coe_reg      = coe_reg[:Nmin]

        data_uvt_list.append(data_uvt_reg)
        data_pod_list.append(data_pod_reg)
        V_list.append(coe_reg)

    data_uvt_all = np.concatenate(data_uvt_list, axis=0).astype(np.float32)  # [Ntot,channel,H,W]
    data_pod_all = np.concatenate(data_pod_list, axis=0).astype(np.float32)  # [Ntot,channel,H,W]
    V_all        = np.concatenate(V_list, axis=0).astype(np.float32)       # [Ntot,channel*rank]

    del data_uvt_reg, data_pod_reg, coe_reg, V_reg
    # ---------------------------
    # normalise all cases with a single scaler
    # ---------------------------
    if normalize_uvt:
        sca = ut.FieldScaler(mode='minmax', channels_last=False).fit(data_uvt_all)
        data_uvt_all = sca.transform(data_uvt_all)
        data_pod_all = sca.transform(data_pod_all)
        data_lsc_pred = sca.transform(data_lsc_pred)
    
    if normalize_lstm:
        scal = ut.MinMaxScaler(feature_range=(0, 1))
        scal.fit(V_all)
   
    data_uvt_list = np.split(data_uvt_all, len(regimes))
    data_pod_list = np.split(data_pod_all, len(regimes))
    V_list = np.split(V_all, len(regimes))
    
    # normalise the flow parameters
    Ra_list = np.log10(np.array(Ra_list, dtype=np.float32))
    Pr_list = np.array(Pr_list, dtype=np.float32)
    cond = [i for i in zip(Ra_list, Pr_list)]

    del data_uvt_all, data_pod_all

    offlineStart = TT.time()

    import ML_CondRBCrec as ML_KPM
    if Transfer:
        load = True
    model = ML_KPM.ML_train(
        [data_uvt_list[i] for i in idx_train],
        B_path if Transfer else path, method,
        valid=valid, epoch=epoch, batch_size=batch_size,
        load=load, patience=patience,
        lsc=[data_pod_list[i] for i in idx_train],
        coe=[V_list[i] for i in idx_train],
        test_size=test_size,
        residual=residual, normalize_uvt=normalize_uvt,
        scaler=sca,
        cond=[cond[i] for i in idx_train],
        lambda_div=lambda_div, lambda_Nu=lambda_Nu)

    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()
    
    # online prediction
    data_lsc = data_lsc_pred
    data_uvt = data_uvt_list[idx_pred]
    data_pod = data_pod_list[idx_pred]
    Ra = np.full_like(data_lsc[:,0:1], cond[idx_pred][0])
    Pr = np.full_like(data_lsc[:,0:1], cond[idx_pred][1])
    V = V_list[idx_pred]
    
    data_lsc_in = np.concatenate([data_lsc, Ra, Pr], axis=1)
    
    data_p = ML_KPM.ML_pred(data_lsc_in, model, residual)
            
    data_p = data_p[:,:channel]

    if normalize_uvt:
        data_p = sca.inverse_transform(data_p)
        data_uvt = sca.inverse_transform(data_uvt)
        data_lsc = sca.inverse_transform(data_lsc)
        data_pod = sca.inverse_transform(data_pod)

    data_full = data_uvt
    
    S_pred = data_p[:,ch_slice][:,0]
    S_pod = data_pod[:,ch_slice][:,0]
    S_lsc = data_lsc[:,ch_slice][:,0]
    S_full = data_full[:,ch_slice][:,0]
    S_pred = rr(S_pred)
    S_pod = rr(S_pod)
    S_lsc = rr(S_lsc)
    S_full = rr(S_full)


    onlineEnd = TT.perf_counter()
    snapshots2 = list(np.transpose(S_pred))   
    
    if writeData:    
        np.save(path+'/UVT_pred' , data_p) 


# =============================================================================
# Temporal Prediction Module
# =============================================================================
# Advances the reduced-order coordinate, predicting the Ra-Pr dependent
# reversal dynamics of the LSC.
elif 'DLMD' in method and 'all' in case:  
    
    full_data_list = []
    Ra_list = []
    Pr_list = []
    data_uvt_list = []
    data_lsc_list = []
    V_list = []
    USV_list = []
    data_mean_list = []

    for reg in regimes:
        S_col_full_reg = full_data.shape[0]
        
        fileName = reg['file']
        Ra_list.append(reg['Ra'])
        Pr_list.append(reg['Pr'])

        full_data = np.load('../data/' + fileName).astype(np.float32)
        full_data_list.append(full_data)
        
        data_full_flat = full_data.reshape((S_col_full_reg, 3, -1))
        data_mean = np.mean(data_full_flat, axis=0)[:channel, :, np.newaxis]  # [channel,HW,1]

        data_uvt = [rr(full_data[:, i]) for i in range(channel)]  # list of [HW,N]

        # per-component SVD: (X - mean) = U S Vh
        data_usv = []
        for i in range(channel):
            X = data_uvt[i] - data_mean[i]   # [HW,N]
            U, s, Vh = np.linalg.svd(X, full_matrices=False)
            # keep only the leading `rank` modes; Sigma is truncated to avoid a full diagonal matrix
            U_r = U[:, :rank]
            S_r = np.diag(s[:rank])          # [rank,rank]
            V_r = Vh[:rank, :]               # [rank,N]
            data_usv.append([U_r, S_r, V_r])
            
        # low-rank POD reconstruction (the LSC field)
        data_lsc = [recon_pod(data_usv[i], data_mean[i]) for i in range(channel)]  # list of [HW,N]
        data_lsc = np.stack(data_lsc, axis=1)                # [HW,channel,N]
        data_lsc = data_lsc.transpose((2, 1, 0))             # [N,channel,HW]
        data_lsc = data_lsc.reshape((S_col_full_reg, channel, numberx, numbery))

        # raw uvt fields
        data_uvt_reg = np.stack(data_uvt, axis=1)            # [HW,channel,N]
        data_uvt_reg = data_uvt_reg.transpose((2, 1, 0))     # [N,channel,HW]
        data_uvt_reg = data_uvt_reg.reshape((S_col_full_reg, channel, numberx, numbery))
            
         # POD coefficients, stacked per component -> [channel*rank, N]
        V_reg = np.vstack([data_usv[i][2] for i in range(channel)]).astype(np.float32)

        data_uvt_list.append(data_uvt_reg.astype(np.float32))
        data_lsc_list.append(data_lsc.astype(np.float32))
        V_list.append(V_reg)
        USV_list.append(data_usv)
        data_mean_list.append(data_mean)
        
    V = np.vstack(V_list)
    
    del data_uvt_reg, V_reg, U_r, S_r, V_r, U, s, Vh
    del data_usv, data_lsc, data_mean, data_uvt
            
    if normalize_lstm:
        scal = ut.MinMaxScaler(feature_range=(0, 1))
        scal.fit(V)
        V = scal.transform(V)

    # ---------- Ra/Pr conditioning vector (log10 of Ra) ----------
    Ra_list = np.log10(np.array(Ra_list, dtype=np.float32))
    Pr_list = np.array(Pr_list, dtype=np.float32)
    cond = [i for i in zip(Ra_list, Pr_list)]
    
    V_in = np.split(V, len(regimes), axis=0) 
    
    offlineStart = TT.time()

    import ML_CondLSTM as ML_KPM
    if Transfer:
        load = True
    model = ML_KPM.ML_train(
        [V_in[i] for i in idx_train],
        A_path if Transfer else path, method,
        offset=offset, valid=valid, epoch=epoch, batch_size=batch_size,
        patience=patience, t=channel*rank, load=load,
        test_size=test_size,
        cond=[cond[i] for i in idx_train])
        
    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()
    
    # reconstruction and temporal prediction
    Ra, Pr = cond[idx_pred]
    
    # the first 50% of the series is assumed known
    initial = S_col # or offset

    V_pred = np.zeros((len(V), S_col_full + initial - 1), dtype=np.float32)
    idx = 1
    
    encoded = V_in[idx_pred][:,:initial]
    V_pred[:channel*rank,:initial] = V_in[idx_pred][:channel*rank,:initial]
    
    encoded = np.transpose(encoded)
    encoded = ut.wrap_data(encoded, initial)
    
    encoded_true = V_in[idx_pred][:,:S_col_full]
    encoded_true = np.transpose(encoded_true)
    encoded_true = ut.wrap_data(encoded_true, offset + S_col_full - initial)
    encoded_true = torch.from_numpy(encoded_true).to(device)
    
    t_idx = np.linspace(0, 1, S_col_full, dtype=np.float32)
    t_idx = np.arange(0, S_col_full, step=t_idx[1], dtype=np.float32)[:S_col_full]
    t_idx = ut.wrap_data(t_idx, offset + S_col_full - initial)        
    t_idx = torch.from_numpy(t_idx).to(device).unsqueeze(-1)
    
    Ra_idx = np.broadcast_to(Ra, t_idx.shape)
    Ra_idx = torch.tensor(Ra_idx, dtype=torch.float32).to(device)
    
    Pr_idx = np.broadcast_to(Pr, t_idx.shape)
    Pr_idx = torch.tensor(Pr_idx, dtype=torch.float32).to(device)
    
    cond_idx = torch.cat([Ra_idx, Pr_idx], axis=-1).to(device)
    
    encoded = torch.from_numpy(encoded).to(device)
    with torch.no_grad():
        for mm, i in enumerate(range(initial, S_col_full, idx), 1): 
            y1, ___ = model(encoded_true[:,mm:mm+offset], t_idx[:,mm:mm+offset], cond_idx[:,mm:mm+offset])
            encoded = torch.cat((encoded_true[:,1:], y1[:,-1:]), axis=1)
            if encoded.shape[1] > 1:
                encoded[:-idx,-1,:] = encoded[idx:,-2,:]

            V_pred[:channel*rank, i] = y1[-1, -1].detach().cpu().numpy().astype(np.float32)
    V_pred = V_pred[:,:S_col_full]  
 
    onlineEnd = TT.perf_counter()
    
    if normalize_lstm:
        V = scal.inverse_transform(V)
        V_pred = scal.inverse_transform(V_pred)
        
    V_pred = V_pred[:channel*rank]
      
    pred_ch = []
    for i in range(channel):
        U_r = USV_list[idx_pred][i][0]   # [HW, rank]
        S_r = USV_list[idx_pred][i][1]   # [rank, rank]                  
        pred = recon_pod([U_r, S_r, V_pred[i*rank:(i+1)*rank]], data_mean_list[idx_pred][i])  # [rank, N]
        pred_ch.append(pred)
        
    # rec_ch: list of [HW, N] -> stack -> [HW, channel, N]
    data_p = np.stack(pred_ch, axis=1)                 # [HW, channel, N]
    S_pred = data_p.transpose(2, 1, 0).astype(np.float32)  # [N, channel, HW]
    S_pred = S_pred.reshape(S_pred.shape[0], channel, numberx, numbery)
        
    data_pod = S_pred
    data_p = S_pred
    data_full = data_uvt = data_uvt_list[idx_pred]
    data_lsc = data_lsc_list[idx_pred]
    V = V_list[idx_pred]
    
    plot_r = min(channel*rank, 6)
    
    ut.plot_lstm(V, train_size, time, plot_r, path, name='POD Coe true', label=True)
    ut.plot_lstm(V_pred, train_size, time, plot_r, path, name='POD Coe pred', label=True)
    

    np.save(path+'/S_pred', S_pred)
    np.save(path+'/V_pred', V_pred)
    
    if len(S_pred.shape) > 2:
        S_pred = S_pred[:,ch_slice][:,0]
        S_pod = data_pod[:,ch_slice][:,0]
        S_lsc = data_lsc[:,ch_slice][:,0]
        S_full = data_full[:,ch_slice][:,0]
    S_pred = rr(S_pred)
    S_pod = rr(S_pod)
    S_lsc = rr(S_lsc)
    S_full = rr(S_full)
    
    snapshots2 = list(np.transpose(S_pred))
    
# =============================================================================
# Offline (training) and online (prediction) timings
# =============================================================================
try:    
    total_epoch = pd.read_csv(path+'training_log.csv')
    total_epoch = len(total_epoch)
except:
    total_epoch = epoch

time_path = os.path.join(path, "time.txt")
mode = "a" if os.path.exists(time_path) else "w"
with open(time_path, mode) as f:
    offlineTime = offlineEnd - offlineStart
    onlineTime = onlineEnd - onlineStart
    time_per_epoch = offlineTime / max(total_epoch, 1)

    print('offline time: ' + str('%.5f' % offlineTime))
    print('online time: ' + str('%.5f' % onlineTime))
    print('time per epoch: ' + str('%.5f' % time_per_epoch) + 's')

    if mode == "a":
        f.write("\n")  # separate consecutive runs

    f.write('offline time: ' + str('%.5f' % offlineTime) + 's' + '\n')
    f.write('online time: ' + str('%.5f' % onlineTime) + 's' + '\n')
    f.write('time per epoch: ' + str('%.5f' % time_per_epoch) + 's' + '\n')

# =============================================================================
# Error estimation
# =============================================================================
minT = int(min(len(snapshots), len(snapshots2)))
snapshots = snapshots[st:minT]
snapshots2 = snapshots2[st:minT]

RMSE_train = ut.get_rmse(snapshots[:S_col-st], snapshots2[:S_col-st])
NRMSE_train = ut.get_NRMSE(snapshots[:S_col-st], snapshots2[:S_col-st])

if minT > S_col:
    RMSE_pred = ut.get_rmse(snapshots[S_col-st:], snapshots2[S_col-st:])
    RMSE_all = ut.get_rmse(snapshots, snapshots2)
    NRMSE_pred = ut.get_NRMSE(snapshots[S_col-st:], snapshots2[S_col-st:])
    NRMSE_all = ut.get_NRMSE(snapshots, snapshots2)
else:
    RMSE_pred = np.nan
    RMSE_all = np.nan
    NRMSE_pred = np.nan
    NRMSE_all = np.nan
RMSE_list = [RMSE_train, RMSE_pred, RMSE_all]
NRMSE_list = [NRMSE_train, NRMSE_pred, NRMSE_all]

loc = locals()
def get_variable_name(variable):
    for k, v in loc.items():
        if loc[k] is variable:
            return k

def write_NME(RMSE_list, writepath, ob):
    fileObject = open(writepath+'/{}_list.txt'.format(ob), 'w')
    for i in range(len(RMSE_list)):
        fileObject.write(get_variable_name(RMSE_list[i]) + '\r')
        fileObject.write(str(RMSE_list[i]))
        fileObject.write('\n')
    fileObject.close()

write_NME(RMSE_list, path, 'RMSE')
write_NME(NRMSE_list, path, 'NRMSE')

if writeData:
    np.save(writepath+'/{}_reconstructed'.format(method), np.array(snapshots2, dtype=np.float32).T)

print('Task completes')
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=========="*8 + "%s"%nowtime)
print('data path: {}'.format(path))
print('method:', method, 'rank:',
    rank, 'case:',
    case, 'batch_size:',
    batch_size, 'epoch:',
    epoch, 'load:',
    load, 'activ:',
    activ, 'offset:',
    offset, 'normalize:',
    normalize, 'backend:',
    backend, 'c_fr:',
    c_fr,
)

# =============================================================================
# Reversal diagnostics: relative angular momentum, Nusselt number
# =============================================================================
l = xx*data_full[:,1] - yy*data_full[:,0]
l = np.sum(l, axis=(1,2))
Lhat = l / np.max(np.abs(l))  

l_p = xx*data_p[:,1] - yy*data_p[:,0]
l_p = np.sum(l_p, axis=(1,2))
l_p = l_p / np.max(np.abs(l_p)) 

if 'rec' in path:
    
    l_pod = xx*data_pod[:,1] - yy*data_pod[:,0]
    l_pod = np.sum(l_pod, axis=(1,2)) 
    l_pod = l_pod / np.max(np.abs(l_pod)) 
    
    for plt_idx in [int(S_col_full)-1]:
        for field_idx, ii in enumerate(labels):
            fig, ax = plt.subplots(1, 4, dpi=300, figsize=(10, 3))
    
            vmin = min(np.nanmin(data_uvt[plt_idx][field_idx]),
                       np.nanmin(data_pod[plt_idx][field_idx]),
                       np.nanmin(data_p[plt_idx][field_idx]))
            vmax = max(np.nanmax(data_uvt[plt_idx][field_idx]),
                       np.nanmax(data_pod[plt_idx][field_idx]),
                       np.nanmax(data_p[plt_idx][field_idx]))
            levels = np.linspace(vmin, vmax, 51)
    
            cf0 = ax[0].contourf(x, y, data_uvt[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[0].set_title('Ground truth')
    
            ax[1].contourf(x, y, data_pod[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[1].set_title('POD')
    
            ax[2].contourf(x, y, data_lsc[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[2].set_title('POD + TPM')
    
            ax[3].contourf(x, y, data_p[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[3].set_title('POD + TPM + SRM')
    
            # --- streamlines (white) ---
            if ("$u_{x}$" in labels) and ("$u_{y}$" in labels):
                iu, iv = labels.index("$u_{x}$"), labels.index("$u_{y}$")
                ax[0].streamplot(x, y, data_uvt[plt_idx][iu], data_uvt[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
                ax[1].streamplot(x, y, data_pod[plt_idx][iu], data_pod[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
                ax[2].streamplot(x, y, data_lsc[plt_idx][iu], data_lsc[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
                ax[3].streamplot(x, y, data_p[plt_idx][iu], data_p[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
    
            for axes in ax:
                axes.set_xlim(0, 1)
                axes.set_ylim(0, 1)
                axes.set_xticks([])
                axes.set_yticks([])
                axes.set_aspect("equal")
    
            # --- colorbar aligned with subplots (top/bottom match) ---
            fig.subplots_adjust(right=0.88)
            pos = [a.get_position() for a in ax]
            y0 = min(p.y0 for p in pos); y1 = max(p.y1 for p in pos)
            x1 = max(p.x1 for p in pos)
            cax = fig.add_axes([x1 + 0.01, y0, 0.02, y1 - y0])
            cbar = fig.colorbar(cf0, cax=cax)
            
            # --- clean ticks: 1 decimal + no duplicates (avoid "blank" from formatter/locator issues) ---
            nticks = 6
            step = (vmax - vmin) / (nticks - 1) if vmax > vmin else 0.1
            step = max(0.1, np.round(step, 1))  # ensure 0.1 resolution
            t0 = np.floor(vmin / step) * step
            ticks = t0 + step * np.arange(0, nticks + 6)  # overshoot then clip
            ticks = ticks[(ticks >= vmin - 1e-12) & (ticks <= vmax + 1e-12)]
            ticks = np.unique(np.round(ticks, 1))
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tt:.1f}" for tt in ticks])
    
            fig.suptitle(f'{ii} field, $t$ = {time[plt_idx]:.0f}')
            plt.savefig(path + 'Field_compare_{}_{}.jpg'.format(ii, time[plt_idx]), bbox_inches="tight")
            plt.show()
            plt.close()
            
elif 'DLMD' in path:
    
    for plt_idx in [int(S_col_full)-1]:
        for field_idx, ii in enumerate(labels):
            fig, ax = plt.subplots(1, 3, dpi=300, figsize=(8, 3))
    
            vmin = min(np.nanmin(data_uvt[plt_idx][field_idx]),
                       np.nanmin(data_pod[plt_idx][field_idx]),
                       np.nanmin(data_p[plt_idx][field_idx]))
            vmax = max(np.nanmax(data_uvt[plt_idx][field_idx]),
                       np.nanmax(data_pod[plt_idx][field_idx]),
                       np.nanmax(data_p[plt_idx][field_idx]))
            levels = np.linspace(vmin, vmax, 51)
    
            cf0 = ax[0].contourf(x, y, data_uvt[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[0].set_title('Ground truth')
    
            ax[1].contourf(x, y, data_lsc[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[1].set_title('POD')
    
            ax[2].contourf(x, y, data_p[plt_idx][field_idx], levels=levels, cmap=plt.cm.inferno)
            ax[2].set_title('POD + TPM')
    
            # --- streamlines (white) ---
            if ("$u_{x}$" in labels) and ("$u_{y}$" in labels):
                iu, iv = labels.index("$u_{x}$"), labels.index("$u_{y}$")
                ax[0].streamplot(x, y, data_uvt[plt_idx][iu], data_uvt[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
                ax[1].streamplot(x, y, data_lsc[plt_idx][iu], data_lsc[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
                ax[2].streamplot(x, y, data_p[plt_idx][iu], data_p[plt_idx][iv],
                                 density=1.0, linewidth=0.5, arrowsize=0.8, color="tab:blue")
    
            for axes in ax:
                axes.set_xlim(0, 1)
                axes.set_ylim(0, 1)
                axes.set_xticks([])
                axes.set_yticks([])
                axes.set_aspect("equal")
    
            # --- colorbar aligned with subplots (top/bottom match) ---
            fig.subplots_adjust(right=0.88)
            pos = [a.get_position() for a in ax]
            y0 = min(p.y0 for p in pos); y1 = max(p.y1 for p in pos)
            x1 = max(p.x1 for p in pos)
            cax = fig.add_axes([x1 + 0.01, y0, 0.02, y1 - y0])
            cbar = fig.colorbar(cf0, cax=cax)
            
            # --- clean ticks: 1 decimal + no duplicates (avoid "blank" from formatter/locator issues) ---
            nticks = 6
            step = (vmax - vmin) / (nticks - 1) if vmax > vmin else 0.1
            step = max(0.1, np.round(step, 1))  # ensure 0.1 resolution
            t0 = np.floor(vmin / step) * step
            ticks = t0 + step * np.arange(0, nticks + 6)  # overshoot then clip
            ticks = ticks[(ticks >= vmin - 1e-12) & (ticks <= vmax + 1e-12)]
            ticks = np.unique(np.round(ticks, 1))
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tt:.1f}" for tt in ticks])
    
            fig.suptitle(f'{ii} field, $t$ = {time[plt_idx]:.0f}')
            plt.savefig(path + 'Field_compare_{}_{}.jpg'.format(ii, time[plt_idx]), bbox_inches="tight")
            plt.show()
            plt.close()
            
    # POD truncation error, TPM forecasting error, and the two combined
    NRMSE_model = ut.get_NRMSE(S_lsc[:,st:], S_pred[:,st:])
    NRMSE_pod = ut.get_NRMSE(S_full[:,st:], S_lsc[:,st:])
    NRMSE_total = ut.get_NRMSE(S_full[:,st:], S_pred[:,st:])
    
    write_NME([NRMSE_pod, NRMSE_model, NRMSE_total], path, 'NRMSE_POD')
            
    df = pd.DataFrame({'A': V[0], 'B': V[1], 'C': V[2]})
    # pairwise Pearson correlation
    corr = np.asarray(df.corr(method='pearson'))   # 3x3
    print("Pairwise correlation:\n", corr)
    # heat map and trajectories
    fig, ax = plt.subplots(
        3, 2, figsize=(10, 4), dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.8]},
        constrained_layout=True
    )

    # --- 3x3 heatmap (left, share one axis across 3 rows) ---
    ax_hm = ax[0, 0]
    ax[1, 0].remove()
    ax[2, 0].remove()
    ax_hm = fig.add_subplot(3, 2, (1, 5))  # span rows 1..3 in col 1

    ax_hm.imshow(corr, vmax=1, vmin=0, cmap=plt.cm.coolwarm)
    ax_hm.set_xticks([0, 1, 2], labels)
    ax_hm.set_yticks([0, 1, 2], labels)
    ax_hm.set_title("Pearson Coefficient")
    ax[0, 0].remove()

    for i in range(3):
        for j in range(3):
            ax_hm.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

    ax[0, 1].plot(time, V[0], lw=1.2); ax[0, 1].set_ylabel("$POD_{1}(u_{x})$"); ax[0, 1].set_title("POD Mode-1 coeff")
    ax[1, 1].plot(time, V[1], lw=1.2); ax[1, 1].set_ylabel("$POD_{1}(u_{y})$")
    ax[2, 1].plot(time, V[2], lw=1.2); ax[2, 1].set_ylabel("$POD_{1}(T)$"); ax[2, 1].set_xlabel("Time")

    for k in range(2):
        ax[k, 1].tick_params(labelbottom=False)
        
    fig.tight_layout()
    plt.savefig(path + 'Correlation3x3.jpg')

    r = np.corrcoef(V[0+rank_st], Lhat)[0, 1]
    print("Pearson correlation r =", r)

plt.figure(figsize=(8,3), dpi = 300)
plt.plot(time[st:], Lhat[st:], label='Ground truth', ls='-', c='navy', alpha=0.8)
plt.plot(time[st:], l_p[st:], label='Prediction', ls='-.', marker='x', markersize=3, c='tab:orange', alpha=0.8)
plt.vlines(time[S_col], np.min(Lhat), np.max(Lhat), colors='gray')
plt.legend()
l_err = ut.get_NRMSE(Lhat[st:], l_p[st:])
plt.title(f'NRMSE = {l_err:.3f}')
plt.savefig(path + 'Angular.jpg')
plt.savefig(path + 'Angular.svg')

L_pdf = ut.compute_pdf_1d(Lhat[st:])
L2_pdf = ut.compute_pdf_1d(l_p[st:])

plt.figure(figsize=(8,3), dpi = 300)
plt.plot(*L_pdf, label='Ground truth', ls='-', c='navy', alpha=0.8)
plt.plot(*L2_pdf, label='Prediction', ls=':', marker='x', markersize=3, c='tab:orange', alpha=0.8)
if 'rec' in path:
    try:
        L3_pdf = ut.compute_pdf_1d(l_pod)
        plt.plot(*L3_pdf, label='POD', ls=':', marker='^', markersize=3, c='tab:green', alpha=0.8)
    except:
        pass
plt.xlabel('Angular momentum')
plt.ylabel('PDF')
pdf_err = ut.get_NRMSE(L_pdf, L2_pdf)
plt.title(f'NRMSE = {pdf_err:.3f}')
plt.legend()
plt.savefig(path + 'Angular PDF.jpg')
plt.savefig(path + 'Angular PDF.svg')

lags, r = ut.cross_corr(l_p[st:], Lhat[st:], dltT, path, plot=True)
print("cross-correlation =", np.max(r), "at lag =", lags[np.argmax(r)])

l_pdf = ut.compute_pdf_1d(Lhat)
print("integral of the angular-momentum pdf =", np.trapz(l_pdf[-1,:], l_pdf[0,:]))

# Reversal time
idx_rev_Lr, __ = ut.detect_reversals(l[st:], win=3, poly=2, min_gap=1)
idx_rev_Lp, __ = ut.detect_reversals(l_p[st:], win=3, poly=2, min_gap=1)

txt_path = os.path.join(path, "reversals_time.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"total reversals, ground truth: {len(idx_rev_Lr)}\n")
    f.write(f"total reversals, prediction: {len(idx_rev_Lp)}\n\n")
    f.write("reversal times, ground truth:\n")
    f.write(", ".join(map(lambda x: f"{float(x):.6f}", idx_rev_Lr)) + "\n\n")
    f.write("reversal times, prediction:\n")
    f.write(", ".join(map(lambda x: f"{float(x):.6f}", idx_rev_Lp)) + "\n\n")
print(f"[saved] text written to: {txt_path}")

csv_path = os.path.join(path, "reversals_time.csv")
df = pd.DataFrame({"t_true": pd.Series(idx_rev_Lr, dtype=float),
                    "t_pred": pd.Series(idx_rev_Lp, dtype=float)})
df.to_csv(csv_path, index_label="index")
print(f"[saved] CSV written to: {csv_path}")

# Nusselt Number
Nut_r, Nu_mean_r, Nu_bot_r, Nu_top_r = ut.Nu(data_full[st:], kappa, H=1.0)
__, Nu_mean_l, Nu_bot_l, Nu_top_l = ut.Nu(data_lsc[st:], kappa, H=1.0)
try:
    Nut_p, Nu_mean_p, Nu_bot_p, Nu_top_p = ut.Nu(data_p[st:], kappa, H=1.0)
except:
    Nut_p, Nu_mean_p, Nu_bot_p, Nu_top_p = ut.Nu(data_pod[st:], kappa, H=1.0)

plt.figure(figsize=(6,2), dpi = 300)
plt.plot(time[st:], Nut_r, label='Ground truth', ls='-', c='navy', alpha=0.8)
plt.plot(time[st:], Nut_p, label='Prediction', ls=':', marker='x', markersize=3, c='tab:orange', alpha=0.8)
plt.vlines(time[S_col], np.min(Nut_r), np.max(Nut_r), colors='gray')
plt.legend()
Nu_NRMSE = ut.get_NRMSE(Nut_r, Nut_p)
plt.title(f'Nu NRMSE = {Nu_NRMSE:.3f}')
plt.savefig(path + 'Nu.jpg')

txt_path = os.path.join(path, "Nu.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Nu_mean_r \n")
    f.write(f"True: {Nu_mean_r}\n")
    f.write(f"POD: {Nu_mean_l} \n")
    f.write(f"Prediction: {Nu_mean_p} \n")
print(f"[saved] text written to: {txt_path}")

import csv
csv_path = os.path.join(path, "paper_metrics.csv")

def _to_float(x):
    return float(np.asarray(x).ravel()[0])

def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

def _stats_row(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean="", std="", min="", max="")
    return dict(
        n=int(x.size),
        mean=float(x.mean()),
        std=float(x.std()),
        min=float(x.min()),
        max=float(x.max()),
    )

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)

    # header of the long-format table
    w.writerow(["section", "name", "value", "note"])

    # -------------------------
    # Nusselt
    # -------------------------
    w.writerow(["Nusselt", "Nu_mean_true", _to_float(Nu_mean_r), ""])
    w.writerow(["Nusselt", "Nu_mean_pod", _to_float(Nu_mean_l), ""])
    w.writerow(["Nusselt", "Nu_mean_pred", _to_float(Nu_mean_p), ""])

    # Nu NRMSE: prefer Nu_NRMSE, fall back to l_err
    nu_nrmse_val = _to_float(Nu_NRMSE) if "Nu_NRMSE" in globals() else _to_float(l_err)
    w.writerow(["Nusselt", "Nu_NRMSE", nu_nrmse_val, "prefer Nu_NRMSE; fallback l_err"])
    w.writerow(["Angular", "L_NRMSE", _to_float(l_err), ""])

    # -------------------------
    # Reversals counts
    # -------------------------
    w.writerow(["Reversals", "count_true_total", _safe_len(idx_rev_Lr), ""])
    w.writerow(["Reversals", "count_pred_total", _safe_len(idx_rev_Lp), ""])

    # -------------------------
    # Cross-correlation
    # -------------------------
    rmax = float(np.max(r))
    lag_at = float(np.asarray(lags)[np.argmax(r)])
    w.writerow(["CrossCorr", "max_r", rmax, ""])
    w.writerow(["CrossCorr", "lag_at_max", lag_at, ""])

    # -------------------------
    # NRMSE POD / model / total
    # -------------------------
    if "NRMSE_pod" in globals():
        w.writerow(["NRMSE", "NRMSE_pod",   float(NRMSE_pod),   ""])
        w.writerow(["NRMSE", "NRMSE_model", float(NRMSE_model), ""])
        w.writerow(["NRMSE", "NRMSE_total", float(NRMSE_total), ""])

    # -------------------------
    # RMSE_list / NRMSE_list
    # -------------------------
    def dump_list(section, name, arr):
        st = _stats_row(arr)
        w.writerow([section, f"{name}_stats_n",   st["n"],   ""])
        w.writerow([section, f"{name}_stats_mean", st["mean"], ""])
        w.writerow([section, f"{name}_stats_std",  st["std"],  ""])
        w.writerow([section, f"{name}_stats_min",  st["min"],  ""])
        w.writerow([section, f"{name}_stats_max",  st["max"],  ""])

        # values (one per row)
        w.writerow([section + "_values", "series", "index", "value"])
        arr = np.asarray(arr, dtype=np.float64).ravel()
        for i, v in enumerate(arr):
            w.writerow([section + "_values", name, i, float(v)])

    if "RMSE_list" in globals():
        dump_list("RMSE", "RMSE_list", RMSE_list)
    if "NRMSE_list" in globals():
        dump_list("NRMSE_list", "NRMSE_list", NRMSE_list)

print(f"[saved] CSV written to: {csv_path}")
