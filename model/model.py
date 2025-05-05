import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import mne 
from typing import List, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score
import warnings
from sklearn.model_selection import train_test_split
import os
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import iirnotch, butter, filtfilt
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
#import the helper functions for the data handling and postprocessing
import dataHandler
import postProcessing
#for the csv file generation
np.set_printoptions(threshold=np.inf)


def compute_fold_stats(loader, batch_size=32, device="cpu"):
    """
    Given a TensorDataset of raw EEG windows (no labels) for one fold:
      - train_dataset: Dataset yielding (X_window, y) but we'll ignore y
    Computes per‐channel mean & std over ALL windows in train_dataset
    in a streaming (batch‐by‐batch) way.
    Returns (mu, sigma) each torch.Tensor of shape (1, C, 1).
    """
    # We'll accumulate sum and sum of squares
    sum_c   = None   # will become torch.Tensor(C,)
    sumsq_c = None
    total   = 0      # total samples = sum over batches of (B * L)
    
    for Xb, _ in loader:
        # Xb: (B, C, L)
        B, C, L = Xb.shape
        xb = Xb.reshape(B, C, L).to(torch.float64)  # promote for numeric stability
        
        # sum over batch & time dims:
        s  = xb.sum(dim=(0,2))         # shape (C,)
        ss = (xb * xb).sum(dim=(0,2))  # shape (C,)
        
        if sum_c is None:
            sum_c, sumsq_c = s, ss
        else:
            sum_c   += s
            sumsq_c += ss
        
        total += B * L

    # compute mean & std per channel:
    mu_c    = sum_c   / total           # (C,)
    var_c   = sumsq_c/total - mu_c**2   # (C,)
    std_c   = torch.sqrt(torch.clamp(var_c, min=1e-8))
    
    # reshape to (1, C, 1) for broadcasting
    mu  = mu_c.view(1, C, 1).float()
    std = std_c.view(1, C, 1).float()
    return mu, std

class FeaStackedSensorFusion_eeg_only(nn.Module):
    def __init__(self, eeg_channels=23, eeg_out_channels=16, num_classes=2,p_dropout=0.5):
        super(FeaStackedSensorFusion_eeg_only, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 4, (1,4), (1,1), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,8), stride=(1,8)),

            nn.Conv2d(4, 16, (1,16), (1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(1,8),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(16,1),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),

            nn.Conv2d(16, 16, (8,1),(1,1),padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p_dropout)   
        self.fcn = nn.Linear(16, num_classes)

    def forward(self, x1):
        out1 = self.net1(x1)
        out = self.fcn(out1)
        return out
    
class SSWCELoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        """
        Sensitivity-Specificity Weighted Cross-Entropy Loss:
        
          Loss = CE(logits, y) 
               + α * (1 - Specificity) 
               + β * (1 - Sensitivity)
        
        where
          Sensitivity = TP / (TP + FN)
          Specificity = TN / (TN + FP)
        are computed in a “soft” manner from the predicted probabilities.
        
        Args:
          alpha: weight on the specificity penalty
          beta:  weight on the sensitivity penalty
          eps:   small term to avoid division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.ce    = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        logits:  (N, 2) raw model outputs
        targets: (N,) integer class labels {0,1}
        """
        # 1) standard cross‐entropy
        ce_loss = self.ce(logits, targets)
        
        # 2) get predicted probability of the positive class
        probs = F.softmax(logits, dim=1)[:, 1]    # shape (N,)
        
        # 3) “soft” counts for TP, FN, TN, FP
        y_true = targets.float()                 # 0. or 1.
        TP = torch.sum(probs * y_true)
        FN = torch.sum((1 - probs) * y_true)
        TN = torch.sum((1 - probs) * (1 - y_true))
        FP = torch.sum(probs * (1 - y_true))
        
        # 4) sensitivity & specificity
        sensitivity  = TP / (TP + FN + self.eps)
        specificity  = TN / (TN + FP + self.eps)
        
        # 5) SSWCE = CE + α·(1–SP) + β·(1–SN)
        loss = ce_loss \
             + self.alpha * (1.0 - specificity) \
             + self.beta  * (1.0 - sensitivity)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction="mean"):
        """
        gamma: focusing parameter (>=0).  gamma=0 => CE loss.
        alpha: balance parameter in [0,1] for the positive class.
        """
        super().__init__()
        self.gamma    = gamma
        self.alpha    = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor):
        """
        logits: (B, C) raw outputs
        targets: (B,) in {0,..,C-1}
        """
        # 1) compute standard cross‐entropy loss per-example
        ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)

        # 2) compute p_t = exp(–ce)  => the model’s estimated prob for the true class
        p_t = torch.exp(-ce)

        # 3) focal modulation factor
        mod = (1 - p_t) ** self.gamma

        # 4) class‐balance alpha: assign alpha to the positive class
        #    here: assume binary C=2, and that `targets==1` are the positives
        alpha_factor = torch.ones_like(targets, dtype=torch.float).to(logits.device)
        alpha_factor = torch.where(targets == 1,
                                   self.alpha,
                                   1 - self.alpha)  # (B,)

        # 5) final loss
        loss = alpha_factor * mod * ce  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # "none"


class WeightedFocalLoss(FocalLoss):
    def __init__(self, gamma, alpha, class_weights, reduction="mean"):
        super().__init__(gamma, alpha, reduction)
        self.class_weights = class_weights  # torch.tensor([w_neg, w_pos])
    def forward(self, logits, targets):
        # weighted CE
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights.to(logits.device),
            reduction="none"
        )  # (B,)
        p_t = torch.exp(-ce)
        mod = (1 - p_t)**self.gamma
        alpha_factor = torch.where(targets==1,
                                   self.alpha,
                                   1-self.alpha)
        loss = alpha_factor * mod * ce
        return loss.mean() if self.reduction=="mean" else loss.sum()
    
    
def run(subject = '06', run_idx  = 0):
       
    
    base_path = Path('/usr/scratch/sassauna1/sem25f13')
    #subject = '01'
    preictal_duration_min = 30 #What timespan before a seizure is considered preictal
    interictal_preictal_buffer_min = 240 #What buffer time between preictal and interictal is used, ignore this data
    postictal_duration_min = 180 #What timespan after a seizure is considered postictal, ignore this data
    records = dataHandler.parse_summary_file(str(base_path / f'chb{subject}/chb{subject}-summary.txt'))
    
    interictal = dataHandler.extract_interictal_data(records,preictal_duration_sec=preictal_duration_min*60,   
                                           interictal_buffer_sec=interictal_preictal_buffer_min*60,  
                                           postictal_duration_min=postictal_duration_min) 
    preictal = dataHandler.extract_preictal_segments(records)
    preictal = dataHandler.extract_preictal_segments_distance(records,preictal_duration_min = 30,
                                                    postictal_duration_min = 10,
                                                    min_length_min = 10)
    abs_times_per_file = dataHandler.helper_absolute_time(records)
    file_abs_starts = { fn: info["abs_start"] for fn, info in abs_times_per_file.items() }
    print("Extracted interictal and preictal times")
    tn_chunks = dataHandler.combine_interictal_preictal(interictal, preictal)
    always_present_channels_for_subject = dataHandler.get_subject_channels(records, base_path/f'chb{subject}')
    interictal_data_list, preictal_data_list = dataHandler.process_tn_chunks(tn_chunks, str(base_path / f'chb{subject}'), always_present_channels_for_subject)
    print("processed interictal and preictal segments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{subject}_{int(time.time())}"
    log_dir = f"runs/swceelr2/{run_id}"
    ckpt_dir = f"checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)


    # hyperparams
    n_chunks      = len(interictal_data_list)
    window_len = 20    # seconds
    sampling_rate = 0.25
    val_frac = 0.25
    batch_size = 32         #use higher
    epochs = 100            #100 is fine
    patience = 30
    lr = 1e-3               #can also use 1e-4
    weight_decay = 1e-4     #bigger L2 regularization seems to work better
    dropout = 0.5           #Dropout right before the FC layer
    # for k-of-n filtering
    k = 15
    n = 20
    # for refractory period
    R = 30 #min
    # for evaluation
    SPH = 5  # min
    SOP = 30  # min
    
    hparams = {
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'epochs': epochs,
        'patience': patience,
        'k': k,
        'n': n,
        'R': R,
        'SPH': SPH,
        'SOP': SOP,
    }

    # 3) Log them right away so they show up in the HParams dashboard
    writer.add_hparams(hparams, {})

    all_fold_metrics = []
    all_fold_metrics_post = []
    test_results = []
    test_results_post = []
    training_history = []
    
    #For computation of final metrics
    total_sens = 0
    total_fa   = 0
    total_ih   = 0.0
    lead_times = [] 
    total_alarms = [] 
    y_pred_forecast = []
    run_summaries = []

    for fold in range(n_chunks):
        # ==== 1) Prepare the data for this fold ====
        # 1a) test chunk
        print(f"Fold {fold+1}/{n_chunks}")
        test_pair = (interictal_data_list[fold], preictal_data_list[fold])
        test_loader = dataHandler.make_test_loader(test_pair, window_len, batch_size)

        # 1b) balance & pool the other n-1 chunks -> no balancing anymore
        train_pairs = [
            (interictal_data_list[i], preictal_data_list[i])
            for i in range(n_chunks) if i != fold
        ]
        #Balance the dataset -> was used for the over/undersampling
        # balanced = make_balance_dataset(train_pairs, window_len, sampling_rate)
        # train_loader, val_loader = make_train_val_loaders_last_balanced(
        #     balanced, val_frac=val_frac, batch_size=batch_size)
        #unbalanced sets, no oversampling/undersampling 
        train_loader, val_loader = dataHandler.make_train_val_loaders(train_pairs, window_len, val_frac=val_frac, batch_size=batch_size)
        #count interictal & preictal windows in the training set from the loader
        interictal_count = 0
        preictal_count = 0
        for i, (X, y) in enumerate(train_loader):
            interictal_count += (y == 0).sum().item()
            preictal_count += (y == 1).sum().item()

        # ==== 2) Compute μ/σ *only on train* (streaming) ====
        mu, std = compute_fold_stats(train_loader, batch_size=batch_size)
        mu, std = mu.to(device), std.to(device)

        # ==== 3) Build model, optimizer, loss ====
        model = FeaStackedSensorFusion_eeg_only(p_dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        
        #Here different loss functions are tested
        pos_weight = interictal_count / preictal_count
        #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = SSWCELoss(alpha=0.5, beta=1.0) #experiment with different alpha, beta values
        #focal loss
        #criterion = FocalLoss(gamma=1.0, alpha=0.75)
        #criterion = WeightedFocalLoss(gamma=2.0,alpha=0.25,class_weights=torch.tensor([1.0, pos_weight]))

        # ==== 4) Train with early‑stopping on val ==== 
        # DO either via val_loss or val_acc
        global_step = 0
        best_val_loss = float('inf')
        best_val_acc = 0.0
        wait = 0
        for epoch in range(epochs):
            print(f"[Fold {fold:2d}] Epoch {epoch+1}/{epochs}")
            # -- train pass --
            model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = (Xb - mu) / std    # Xb: (batch, 23, 2560)
                Xb = Xb.unsqueeze(1)    # now Xb: (batch,  1, 23, 2560) -> for 2dConv

                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                global_step += 1
                
            avg_train_loss = np.mean(train_losses)
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

            # -- val pass --
            model.eval()
            val_losses = []
            y_true_val, y_score_val, y_pred_val = [], [], []
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    Xv = (Xv - mu) / std
                    Xv = Xv.unsqueeze(1)
                    val_losses.append(criterion(model(Xv), yv).item())
                    
                    logits = model(Xv)
                    probs = torch.softmax(logits, dim=1)[:,1]
                    preds = (probs >= 0.5).long()
                    
                    y_true_val .append(yv.cpu().numpy())
                    y_score_val.append(probs.cpu().numpy())
                    y_pred_val .append(preds.cpu().numpy())
                    
            avg_val_loss = np.mean(val_losses)          
                    

            # flatten the lists
            y_true_val  = np.concatenate(y_true_val)
            y_score_val = np.concatenate(y_score_val)
            y_pred_val  = np.concatenate(y_pred_val)

            # compute metrics
            avg_val_loss = np.mean(val_losses)
            val_auc      = roc_auc_score(y_true_val, y_score_val)
            val_acc      = accuracy_score(y_true_val, y_pred_val)
            val_sens     = recall_score(y_true_val, y_pred_val, pos_label=1)  # recall on class=1
            val_spec     = recall_score(y_true_val, y_pred_val, pos_label=0)  # recall on class=0
            val_fpr      = 1 - val_spec

            #write to tensorboard
            # epoch‐level logging
            writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
            writer.add_scalar("val/auc",        val_auc,      epoch)
            writer.add_scalar("val/accuracy",   val_acc,      epoch)
            writer.add_scalar("val/sensitivity",val_sens,     epoch)
            writer.add_scalar("val/fpr",        val_fpr,      epoch)

            
            # early‑stop check, choose one of the two:
            if  avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                wait = 0
                torch.save(model.state_dict(),os.path.join(ckpt_dir, f"best_fold{fold}.pt"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Fold {fold:2d}] Early stopping at epoch {epoch}")
                    break
            
            training_history.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_auc": val_auc,
                "val_sens": val_sens,
                "val_spec": val_spec,
                "val_fpr": val_fpr,
                "val_acc": val_acc                
            })
                        

        # ==== 5) Load best model & test ====
        ckpt_path = os.path.join(ckpt_dir, f"best_fold{fold}.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        y_true_all = []
        y_score_all = []  # prob of class=1
        y_pred_all = []
        y_pred_kn_all = []
        test_losses = []
        

        with torch.no_grad():
            for Xt, yt in test_loader:
                Xt, yt = Xt.to(device), yt.to(device)
                Xt = (Xt - mu) / std
                Xt = Xt.unsqueeze(1)

                logits = model(Xt)
                probs = torch.softmax(logits, dim=1)[:,1]   # Pr(preictal)
                preds = (probs >= 0.5).long()
                
                #calc test loss
                loss = criterion(logits, yt)
                test_losses.append(loss.item())

                y_true_all.append(yt.cpu().numpy())
                y_score_all.append(probs.cpu().numpy())
                y_pred_all.append(preds.cpu().numpy())
                y_pred_kn_all.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true_all)
        y_score = np.concatenate(y_score_all)
        y_pred = np.concatenate(y_pred_all)
        y_pred_kn  = np.concatenate(y_pred_kn_all)
        
        # k-of-n filtering and refractory period
        y_pred_kn = postProcessing.k_of_n_filter(y_pred_kn, k=k, n=n)
        y_pred_post = postProcessing.apply_refractory(window_len=window_len, alarm_seq=y_pred_kn, R=R)
        
        #probabalistic predictions, seizure risk levels/firing power
        y_pred_riskLevels = postProcessing.risk_levels(y_pred, n=n)

        # metrics
        auc = roc_auc_score(y_true, y_score)
        sens = recall_score(y_true, y_pred, pos_label=1)  # preictal recall
        spec = recall_score(y_true, y_pred, pos_label=0)  # interictal recall
        fpr = 1 - spec
        acc = accuracy_score(y_true, y_pred)
        
        # after refractory:
        sens_post = recall_score(y_true, y_pred_post,   pos_label=1)
        spec_post = recall_score(y_true, y_pred_post,   pos_label=0)
        fpr_post  = 1 - spec_post
        acc_post  = accuracy_score(y_true, y_pred_post)

      
        all_fold_metrics.append((auc, sens, spec, fpr, acc))
        
        sens, fp, interictal_seconds, lead_time, alarms,boundry_window_idx, first_preictal_window_idx = postProcessing.evaluate_seizure_fold_SPH_SOP_complex(
        y_pred_post,
        tn_chunks[fold][0],   # intericral segments
        tn_chunks[fold][1],   # preictal segments
        window_len=window_len,  
        SPH=SPH*60,            
        SOP=SOP*60,            
        file_abs_starts=file_abs_starts  # dict: file_name → absolute start (sec)
        )
        
        for e in alarms:
            print(f"  Alarm in {e['segment']:10s} {e['file']:15s} at local {e['local_time']:.1f}s (abs {e['abs_time']:.1f})")  
        total_alarms.append(alarms)
        total_sens += sens
        total_fa   += fp
        total_ih   += interictal_seconds
        if not np.isnan(lead_time):
            lead_times.append(lead_time)
        
        avg_test_loss = np.mean(test_losses)
        
        test_results.append({
            "fold": fold,
            "epoch": 0,              
            "test_loss": avg_test_loss,   
            "test_auc": auc,
            "test_sens": sens,
            "test_spec": spec,
            "test_fpr": fpr,
            "test_acc": acc
        })
       
        #get window index from alarm events
        alarm_idx = [e["window-index"] for e in alarms]
        #calculate the starting times of the chunks: 
        times = dataHandler.chunk_absolute_times(records, tn_chunks[fold])
        
        #append the risk levels to the list
        y_pred_forecast.append({"fold": fold,
                                "first_preictal_window_idx": first_preictal_window_idx,
                                "boundary_window_idx": boundry_window_idx,
                                "alarm_window_idx": alarm_idx,
                                "window_size": window_len, #in seconds -> use for plot afterwards
                                "risk": y_pred_riskLevels,
                                "start_end_times_of_files": times                                                 
                                })
        
        
    writer.close()
    # ==== 6) Aggregate across folds ====
    all_fold_metrics = np.array(all_fold_metrics)  # shape (n_chunks, 3)
    mean_auc, mean_sens, mean_spec, mean_fpr, mean_acc = all_fold_metrics.mean(axis=0)
    std_auc, std_sens, std_spec, std_fpr, std_acc = all_fold_metrics.std(axis=0)
    # after all folds, postprocessing metrics
    overall_sens = total_sens              # number of seizures correctly predicted
    overall_fpr  = total_fa / total_ih if total_ih > 0 else float('nan')     # false alarms per hour
    mean_lead    = float(np.mean(lead_times)) if lead_times else float('nan')  #mean of lead prediction time
    std_lead     = float(np.std( lead_times)) if lead_times else float('nan')  #std of lead prediction time
    
    run_summaries.append({
            "run": run_idx,
            "mean_auc":  mean_auc,
            "std_auc":   std_auc,
            "mean_sens": mean_sens,
            "std_sens":  std_sens,
            "mean_spec": mean_spec,
            "std_spec":  std_spec,
            "mean_fpr":  mean_fpr,
            "std_fpr":   std_fpr,
            "mean_acc":  mean_acc,
            "std_acc":   std_acc,
            "overall_sens_seizures": (overall_sens / n_chunks),
            "total_fa": total_fa,
            "total_ih": total_ih / 3600,
            "overall_fpr": overall_fpr * 3600,
            "mean_lead": mean_lead / 60,
            "std_lead":  std_lead / 60,
        })

    df_training_history = pd.DataFrame(training_history)
    df_test_results = pd.DataFrame(test_results)
    df_test_results_post = pd.DataFrame(test_results_post)
    df_risk_levels = pd.DataFrame(y_pred_forecast)
    df_test_results = pd.concat([df_test_results, df_test_results_post], axis=0)
    df_training_history.to_csv(f"training_history_{subject}.csv", index=False)
    df_test_results.to_csv(f"test_results_{subject}.csv", index=False)
    df_risk_levels.to_csv(f"risk_levels_{subject}.csv", index=False)
    
   
    return run_summaries
