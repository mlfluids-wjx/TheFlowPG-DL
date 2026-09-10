"""Training, inference and physics constraints of the Spatial Reconstruction Module (SRM)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd 
import datetime
import sys
sys.path.append('../utils')
from ML_utils import EarlyStopper

torch.backends.cudnn.benchmark = True

def ML_train(inp, path, method='', valid=None, epoch=1000, batch_size=32,
             patience=200, lsc=None, coe=None, load=False, test_size=0.205,
             residual=False, normalize_uvt=True, scaler=None, cond=None,
             lambda_div=1e-2, lambda_Nu=1e-2):
    """Train or load the Spatial Reconstruction Module.

    A convolutional encoder-decoder with skip connections, constrained by a
    weighted loss that leverages both data and physics: a data-driven term,
    a divergence-free constraint weighted by `lambda_div` and a heat-flux
    constraint weighted by `lambda_Nu`.

    `method` selects the backbone and the active constraints by substring;
    `path` is only the directory that weights are written to and read from.
    """

    loss_monitor = 'all' #(loss / val_loss / all)
    save_steps = 10
    saveat = 0
    lr_max = 1e-3

    loss_fn = nn.MSELoss()
    
    if not isinstance(inp, (list, tuple)):
        inp = [inp]
        lsc = [lsc]
        coe = [coe]
    channel = inp[0].shape[1] + 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'UNet' not in method:
        raise NotImplementedError(
            "The U-Net is the only released SRM backbone; "
            "the method name must contain 'UNet'.")

    # Inputs are the LSC field together with the two conditioning maps (Ra, Pr).
    # The output has the same width; its last two channels are discarded,
    # leaving the reconstructed (u, v, T).
    import ML_UNet_TorchModel as m
    model = m.UNet(channel, norm="gn").to(device)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {total_params}")
       
    if not load:
        if cond is None:
            cond = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(len(inp))]
        elif not isinstance(cond, (list, tuple)):
            cond = [cond for _ in range(len(inp))]
        
        train_sets, test_sets = [], []
        
        for k in range(len(inp)):
            HR_k  = torch.as_tensor(inp[k], dtype=torch.float32)  # [N,C,H,W]
            lsc_k = torch.as_tensor(lsc[k], dtype=torch.float32)  # [N,C,H,W]
            coe_k = torch.as_tensor(coe[k], dtype=torch.float32)  # [N,D] or [N,channel*rank]
        
            # cond_k: [N,2]
            c = np.asarray(cond[k], dtype=np.float32).ravel()
            if c.size >= 2:
                c = c[:2]
            else:
                raise ValueError(f"cond[{k}] must have at least 2 numbers, got {c.size}")
            cond_k = np.concatenate([np.full_like(HR_k[:,0:1], c[0]), np.full_like(HR_k[:,0:1], c[1])], axis=1)
            cond_k = torch.as_tensor(cond_k, dtype=torch.float32)

            N = HR_k.shape[0]
            n_tr = int((1 - test_size) * N)
        
            train_sets.append(TensorDataset(HR_k[:n_tr], coe_k[:n_tr], lsc_k[:n_tr], cond_k[:n_tr]))
            if valid:
                test_sets.append(TensorDataset(HR_k[n_tr:], coe_k[n_tr:], lsc_k[n_tr:], cond_k[:n_tr]))
        
        X_train = DataLoader(ConcatDataset(train_sets),
                             batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        
        if valid:
            X_test = DataLoader(ConcatDataset(test_sets),
                                batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        else:
            X_test = None
         
        l_pred = torch.tensor(.0, requires_grad=False).to(device)
        l_pod = torch.tensor(.0, requires_grad=False).to(device)
        l_Ang = torch.tensor(.0, requires_grad=False).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=lr_max)
        scheduler = None
        
        epochs = epoch
        
        # Training
        best_steps = 0
        best_loss = float(np.inf)
        if not (patience==None or patience==0):
            early_stopper = EarlyStopper(patience=patience, min_delta=0)
    
        dfhistory = pd.DataFrame(columns = ["epoch","loss","loss_data",
                                            "loss_pod","loss_Ang","val_loss","lr"]) 
        
        print("Start Training...")
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=========="*8 + "%s"%nowtime)
        
        for epoch in tqdm(range(1,epochs+1)):
            model.train()
            loss_total = .0
            loss_pred_total = .0
            loss_pod_total = .0
            loss_Ang_total = .0
            for step, data in enumerate(X_train, 1):  
                HR = data[0].to(device)
                coe = data[1].to(device)
                lsc = data[2].to(device)
                cond_b = data[3].to(device) 
                
                lsc = torch.cat([lsc, cond_b], dim=1)
                   
                if residual:
                    delta = model(lsc)[:,:-2]
                    y1 = lsc[:,:-2] + delta
                else:
                    y1 = model(lsc)[:,:-2]
                
                l_pred = loss_fn(HR, y1)

                Nu_p = nusselt_vol_2d(y1)  # [B]
                Nu_t = nusselt_vol_2d(HR)

                if 'div' in method:
                    l_Ang  = lambda_div*div_free_loss_minmax(y1, normalize_uvt, scaler, iu=0, iv=1)
                if 'Nu' in method:
                    l_pod = lambda_Nu*loss_fn(torch.ones_like(Nu_t), Nu_p/Nu_t)
                
                loss = l_pred + l_Ang + l_pod
                loss.backward()
    
                loss_total = loss_total + loss.detach().item()
                loss_pred_total += l_pred.detach().item()
                loss_pod_total += l_pod.detach().item()
                loss_Ang_total += l_Ang.detach().item()
                optimizer.step()
                optimizer.zero_grad()
                
                lr = optimizer.param_groups[0]["lr"]
    
                print('\n')
                print(("[step = %d] loss: %.5f, loss_pred: %.5f, loss_pod: %.5f, loss_Ang: %.5f,") 
                      % (step, loss_total/step, loss_pred_total/step, 
                         loss_pod_total/step, loss_Ang_total/step))
        
            if valid:
                model.eval()
                print('------validation------')
                val_loss_total = .0
                for val_step, val_data in enumerate(X_test, 1):
                    with torch.no_grad():
                        val_HR = val_data[0].to(device)
                        val_lsc = val_data[2].to(device)
                        val_cond = val_data[3].to(device)
                        
                        val_lsc = torch.cat([val_lsc, val_cond], dim=1)
                        
                        if residual:
                            val_delta = model(val_lsc)[:,:-2]
                            val_y1 = val_lsc[:,:-2] + val_delta
                        else:
                            val_y1 = model(val_lsc)[:,:-2]

                        val_l_pred = loss_fn(val_y1, val_HR)
                        
                        val_loss_total += val_l_pred.detach().item()
                        
                    print(("[val_step = %d] val_loss: %.3f,") % (val_step, val_loss_total/val_step))

            else:
                val_step = 1
                val_loss_total = 0

            info = (epoch, loss_total/step, loss_pred_total/step, loss_pod_total/step,
                    loss_Ang_total/step, val_loss_total/val_step, lr)
    
            dfhistory.loc[epoch-1] = info
            
            if epoch % 1 == 0:   
                print(("\nEPOCH = %d, loss = %.5f," " loss_data = %.5f, " " loss_pod = %.5f, "
                        " loss_Ang = %.5f, "  " val_loss = %.5f, "
                        " lr = %.5f, ") %info)
                print(f'Nu_p: {Nu_p.mean().item():.3f}, Nu_t: {Nu_t.mean().item():.3f}')
    
            if not (scheduler is None):
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print('Epoch {:d}; Learning-rate: {:0.05f}'.format(epoch, param_group['lr']))
        
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if np.isnan(loss_total):
                break
            
            if valid:
                if loss_monitor == 'val_loss':
                    monitor = val_loss_total
                elif loss_monitor == 'all':
                    monitor = test_size*val_loss_total + (1 - test_size)*loss_pred_total
                else:
                    monitor = loss_total      
            else:
                monitor = loss_total

            if epoch > (1/5)*epochs:
                if monitor < best_loss:
                    best_steps += 1
                    if best_steps % save_steps == 0:
                        best_loss = monitor
                        print('------Save model:best loss is %.3f------' %best_loss)
                        saveat = epoch
                        torch.save(model.state_dict(), path+'ckpt.pth')
                    
                # early stopping
                if not (patience==None or patience==0):
                    if early_stopper.early_stop(monitor):       
                        print('------Early stopping------')
                        break
    
            print("\n"+"=========="*6 + "%s"%nowtime)
        print('Finished Training...')
        dfhistory.to_csv(path+'training_log.csv', sep=',')
        print('save at {} epoch'.format(saveat))
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(path+'ckpt.pth', weights_only=True))
    model.eval()
    return model

class TimeDistributed(nn.Module):
    def __init__(self, model, time_steps):
        super(TimeDistributed, self).__init__()
        self.model = model
        self.time_steps = time_steps

    def forward(self, x):
        x_split = torch.split(x, 1, dim=1)  
        outputs = []
        for t in range(self.time_steps):
            x_t = x_split[t].squeeze(1) 
            y_t = self.model(x_t)  
            outputs.append(y_t.unsqueeze(1))  
        return torch.cat(outputs, dim=1)  


def ML_pred(ori, model, residual=False, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    x = torch.as_tensor(ori, dtype=torch.float32, device=device)

    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].contiguous()
            yb = model(xb)
            if residual:
                yb = yb + xb
            outs.append(yb.detach().cpu())
    return torch.cat(outs, dim=0).numpy()


def div_free_loss_minmax(y_pred_n, normalize_lstm=True, scaler=None, iu=0, iv=1, dx=None, dy=None):
    B, C, H, W = y_pred_n.shape
    if dx is None: dx = 1.0 / (W - 1)
    if dy is None: dy = 1.0 / (H - 1)

    u_n = y_pred_n[:, iu:iu+1]
    v_n = y_pred_n[:, iv:iv+1]

    du_dx_n = (u_n[..., 1:-1, 2:] - u_n[..., 1:-1, :-2]) / (2.0 * dx)
    dv_dy_n = (v_n[..., 2:, 1:-1] - v_n[..., :-2, 1:-1]) / (2.0 * dy)

    if normalize_lstm:
        assert scaler is not None, "normalize_lstm=True requires scaler (FieldScaler)."
        su = torch.as_tensor(scaler.scale[iu], device=y_pred_n.device, dtype=y_pred_n.dtype)
        sv = torch.as_tensor(scaler.scale[iv], device=y_pred_n.device, dtype=y_pred_n.dtype)
        div = su * du_dx_n + sv * dv_dy_n
    else:
        div = du_dx_n + dv_dy_n

    return (div ** 2).mean()

def nusselt_vol_2d(field, dy=np.float32(1/128), kappa=4.822428e-05, deltaT=50.0, H=1.0, iv=1, iT=2):
    """
    field: [B,C,H,W]  (2D RBC: v=field[:,iv], T=field[:,iT])
    dy: physical grid spacing in y
    returns: Nu_t [B], one value per sample/frame
    """
    v = field[:, iv:iv+1]   # [B,1,H,W]
    T = field[:, iT:iT+1]   # [B,1,H,W]

    # interior central difference: dT/dy -> [B,1,H-2,W]
    dTdy = (T[:, :, 2:, :] - T[:, :, :-2, :]) / (2.0 * dy)

    v_mid = v[:, :, 1:-1, :]
    T_mid = T[:, :, 1:-1, :]

    q = v_mid * T_mid - kappa * dTdy      # [B,1,H-2,W]
    q_bar = q.mean(dim=(2, 3)).squeeze(1) # [B]

    q_cond = kappa * deltaT / H
    Nu_t = q_bar / (q_cond + 1e-12)
    return Nu_t

