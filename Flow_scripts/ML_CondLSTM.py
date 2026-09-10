"""Training and inference of the Temporal Prediction Module (TPM)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd 
import datetime
import sys
sys.path.append('../utils')
import ML_utils as ut


def ML_train(inp, path, method='', offset=1, valid=None, epoch=1000, batch_size=32,
             patience=200, t=4, load=False, test_size=0.20, cond=None):
    """Train or load the Temporal Prediction Module.

    `method` selects the architecture variant by substring; `path` is only the
    directory that weights are written to and read from.
    """
    
    
    if isinstance(inp, (list, tuple)):
        inp_list = list(inp)
    else:
        inp_list = [inp]
        
    if cond is None:
        cond_list = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(len(inp_list))]
    elif isinstance(cond, (list, tuple)):
        cond_list = [np.asarray(c, dtype=np.float32).reshape(2) for c in cond]
        assert len(cond_list) == len(inp_list), "len(cond) must match len(inp)"
    else:
        # a single cond is broadcast to every case
        c0 = np.asarray(cond, dtype=np.float32).reshape(2)
        cond_list = [c0 for _ in range(len(inp_list))]
    
    loss_monitor = 'val_loss' #(loss / val_loss / all)
    log_step_freq = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import ML_CondLatent_TorchModel as m
    
    # Feature augmentation blocks -- Fourier Feature Embedding (FFE) of the
    # time index and Temporal Window Attention (TWA) -- followed by a
    # three-layer stacked LSTM. 'ablation' in the method name deactivates both
    # submodules, leaving a canonical LSTM of identical depth, width and
    # training settings. The input and output size t is channel * rank.
    feature_augmentation = 'ablation' not in method
    lstm = m.LSTM_Attention(
        input_size=t, output_size=t,
        hidden_size=192,
        num_layers=3,
        rff_switch=feature_augmentation,            # FFE
        rff_features=53,                            # number of Fourier modes
        rff_sigma=0.01925377989026111,              # Fourier bandwidth
        rff_learnable=True,
        channel_attn_switch=feature_augmentation,   # TWA
        attn_r=2,                                   # TWA squeeze ratio
        temp_attn_switch=False,
    ).to(device)

    model = lstm
       
    if not load:
        num_t = offset + 1

        Xtrain_all = []
        Xtest_all = []

        # -------------------------------
        # per case: wrap into windows, append time + cond, then concatenate
        # -------------------------------
        for inp_i, cond_i in zip(inp_list, cond_list):
            inp_i = np.transpose(inp_i)

            # split in time order, no shuffling
            X_train, X_test = train_test_split(inp_i, test_size=test_size, shuffle=False)
            N = len(inp_i)
            train_idx = int((1 - test_size) * N)

            # ---- normalised time index, 0..1 within each case ----
            T_full = np.linspace(0, 1, N, dtype=np.float32)
            T_train = ut.wrap_data(T_full[:train_idx], num_t)
            T_train = torch.tensor(T_train).unsqueeze(-1)  # [Ns,T,1]

            X_train = ut.wrap_data(X_train, num_t)
            X_train = torch.tensor(X_train)                # [Ns,T,dims]

            # ---- cond: the same (Ra, Pr) repeated for every sample and time step ----
            C_train = torch.tensor(cond_i, dtype=torch.float32).view(1, 1, 2)
            C_train = C_train.repeat(X_train.shape[0], num_t, 1)         # [Ns,T,2]

            # layout: [..., data(dims), time(1), cond(2)]
            Xtrain_all.append(torch.cat([X_train, T_train, C_train], dim=-1))  # [Ns,T,dims+3]

            if valid:
                T_val = ut.wrap_data(T_full[train_idx:], num_t)
                T_val = torch.tensor(T_val).unsqueeze(-1)

                X_test = ut.wrap_data(X_test, num_t)
                X_test = torch.tensor(X_test)

                C_val = torch.tensor(cond_i, dtype=torch.float32).view(1, 1, 2)
                C_val = C_val.repeat(X_test.shape[0], num_t, 1)

                Xtest_all.append(torch.cat([X_test, T_val, C_val], dim=-1))

        # all cases trained jointly; shuffle=True mixes samples across cases
        X_train = torch.cat(Xtrain_all, dim=0)
        X_train = DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        if valid:
            X_test = torch.cat(Xtest_all, dim=0)
            X_test = DataLoader(X_test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        else:
            X_test = None

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
        epochs = epoch
        
        # Training
        best_loss = 10e8
        if not (patience==None or patience==0):
            early_stopper = m.EarlyStopper(patience=patience, min_delta=0)
    
        dfhistory = pd.DataFrame(columns = ["epoch","loss","loss_rec","loss_lin",
                                            "loss_pred","val_loss","val_rec",
                                            "val_lin","val_pred","lr"]) 
    
        print("Start Training...")
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=========="*8 + "%s"%nowtime)
        
        
        val_loss_total = 0
        
        for epoch in tqdm(range(1, epochs+1), disable=True):
            model.train()
            loss_total = loss_data_total = loss_rev_total = loss_A_total = 0.0

            for step, batch in enumerate(X_train, 1):
                t_hist = batch[..., -3:-2].to(device)     # [B,T,1]
                cond_b = batch[..., -2:].to(device)
                data   = batch[..., :-3].to(device)       # [B,T,dims]

                optimizer.zero_grad(set_to_none=True)

                y_pred, ___ = model(data, t_hist, cond_b)
                loss_data = loss_fn(data[:, -1], y_pred[:, -1])

                loss = loss_data

                loss.backward()
                optimizer.step()
                
                loss_total += loss.detach().item()
                loss_data_total += loss_data.detach().item()
                
                lr = optimizer.param_groups[0]["lr"]
                if step % log_step_freq == 0:   
                    print(("[step = %d] loss: %.6f, data: %.6f, rev: %.6f, A: %.6f,") 
                          % (step, loss_total/step, loss_data_total/step, loss_rev_total/step,
                             loss_A_total/step))
        
            if valid:
                model.eval()
                val_loss_total = .0
                val_loss_data_total = .0
                val_loss_rev_total = .0
                val_loss_A_total = .0
                for val_step, val_data in enumerate(X_test, 1):
                    with torch.no_grad():
                        t_hist = val_data[...,-3:-2].to(device)
                        val_cond = val_data[..., -2:].to(device)
                        val_data = val_data[...,:-3].to(device)

                        y_val, ___ = model(val_data, t_hist, val_cond)
                        
                        val_l_data = loss_fn(val_data[:,-1], y_val[:,-1])
                        
                        val_l_total = val_l_data
                        
                        val_loss_total = val_l_total.detach().item()
                        val_loss_data_total += val_l_data.detach().item()
                        
            if valid:
                val_loss_total = val_l_total.detach().item()
            else:
                val_step = 1
    
            info = (epoch, loss_total/step, loss_data_total/step, loss_rev_total/step,
                    loss_A_total/step, val_loss_total/val_step, val_loss_data_total/val_step,
                    val_loss_rev_total/val_step, val_loss_A_total/val_step, lr)
    
            dfhistory.loc[epoch-1] = info
            
            if epoch % log_step_freq == 0:   
    
                print(("\nEPOCH = %d, loss = %.5f," " loss_data = %.5f, " " loss_rev = %.5f, "
                        " loss_A = %.5f, "  " val_loss = %.5f, "  " val_data = %.5f, "
                        " loss_rev = %.5f, "  " val_A = %.5f, "
                        " lr = %.5f, ") %info)
    
            if(not scheduler is None):
                scheduler.step()
                if(epoch%10 == 0):
                    for param_group in optimizer.param_groups:
                        print('Epoch {:d}; Learning-rate: {:0.05f}'.format(epoch, param_group['lr']))
            
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            loss_total /= max(1, len(X_train))
            val_loss_total /= max(1, len(X_test))
            
            if valid:
                if loss_monitor == 'val_loss':
                    monitor = val_loss_total
                elif loss_monitor == 'all':
                    monitor = test_size*val_loss_total + (1 - test_size)*loss_total
                else:
                    monitor = loss_total      
            else:
                monitor = loss_total
                
            if epoch > epochs // 10 and monitor < best_loss:
                best_loss = monitor
                print('------Save model:best loss is %.8f------' %best_loss)
                saveat = epoch
                torch.save(model.state_dict(), path+'/ckpt.pth')
                
            # early stopping
            if not (patience==None or patience==0):
                if early_stopper.early_stop(monitor):       
                    print('------Early stopping------')
                    break
    
            print("\n"+"=========="*6 + "%s"%nowtime)
        print('Finished Training...')
        dfhistory.to_csv(path+'training_log.csv', sep=',')
        if epochs != 0:
            print('save at {} epoch'.format(saveat)) 
    model.load_state_dict(torch.load(path+'ckpt.pth', weights_only=True))
    model.eval()
    torch.cuda.empty_cache()
    return model
 
    
def ML_pred(ori, model):
    ori = np.transpose(ori)
    try:
        pred = model(torch.tensor(ori)).detach().numpy()
    except:
        pred = model(torch.tensor(ori).to('cuda')).detach()
        pred = pred.cpu().numpy()
    pred = np.transpose(pred)
    return pred
