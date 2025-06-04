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
#for the csv file generation
np.set_printoptions(threshold=np.inf)


def k_of_n_filter(preds: np.ndarray, k:int=8, n:int=10) -> np.ndarray:
    """
    preds: 1D array of 0/1 frame‑wise predictions
    returns: 1D array of 0/1 where you only set ones when
             sum(preds[i-n+1 : i+1]) >= k
    """
    smoothed = np.zeros_like(preds)
    # rolling sum over window of length n:
    cumsum = np.concatenate([[0], np.cumsum(preds)])
    # sum of preds[i-n+1..i] = cumsum[i+1] - cumsum[i-n+1]
    for i in range(len(preds)):
        start = max(0, i - n + 1)
        window_sum = cumsum[i+1] - cumsum[start]
        smoothed[i] = 1 if window_sum >= k else 0
    return smoothed



def apply_refractory(window_len, alarm_seq: np.ndarray, R:int=6,) -> np.ndarray:
    """
    alarm_seq: 1D array of 0/1 alarm flags (after k‑of‑n)
    R: number of windows to suppress after each alarm
    returns: new 1D alarm array with refractory enforced
    """
    R = int(R * 60 / window_len)
    out = alarm_seq.copy()
    i = 0
    N = len(out)
    while i < N:
        if out[i] == 1:
            # zero‑out the next R windows
            out[i+1 : i+1+R] = 0
            i += R  # skip ahead
        i += 1
    return out

def risk_levels(preds: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the Firing Power fp[i] = (1/n) * sum_{k=i-n+1..n} O[k]
    for a binary sequence O. The first n-1 entries are set to 0
    because the window is not yet full.
    """
    # cumulative sum with a leading zero for easy window sums
    cumsum = np.concatenate([[0], np.cumsum(preds)])
    # sliding window averages
    fp = (cumsum[n:] - cumsum[:-n]) / float(n)
    # pad front so fp matches preds’s length
    return np.concatenate([np.full(n-1, 0), fp])

def risk_levels_strings(fp: np.ndarray,
                high_thresh: float = 0.7,
                mod_thresh:  float = 0.3
               ) -> np.ndarray:
    """
    -> Could also use integers (0,1,2) instead of strings.
    Map each value in the Firing Power array to a risk label:
      - 'high'     if fp[i] > high_thresh
      - 'moderate' if mod_thresh < fp[i] <= high_thresh
      - 'low'      if fp[i] <= mod_thresh
      - 'unknown'  if fp[i] is NaN
    """
    risk = np.empty(fp.shape, dtype='<U8')
    risk[np.isnan(fp)] = 'unknown'
    # assign based on thresholds
    risk[fp > high_thresh] = 'high'
    mask_mod = (fp > mod_thresh) & (fp <= high_thresh)
    risk[mask_mod] = 'moderate'
    mask_low = fp <= mod_thresh
    risk[mask_low] = 'low'
    return risk


def evaluate_seizure_fold_SPH_SOP_complex(
    y_pred,            # 1D array of 0/1 alarms
    interictal_chunks, # list of {"file","start","end"} for interictal
    preictal_chunk,    # dict or tuple of dicts for preictal
    window_len,        # seconds
    SPH,               # seconds
    SOP,               # seconds
    file_abs_starts    # dict: file_name → absolute start (sec)
):
    import numpy as np

    # ————————————————
    # 1) Build window_times AND window_files in parallel
    # ————————————————
    times_i, files_i = [], []
    for seg in interictal_chunks:
        f = seg["file"]
        local0, local1 = seg["start"], seg["end"]
        n_win = int(np.floor((local1 - local0) / window_len))
        abs0 = file_abs_starts[f] + local0
        for i in range(n_win):
            times_i.append(abs0 + i * window_len)
            files_i.append((f, local0 + i * window_len, "interictal"))

    times_p, files_p = [], []
    pre_segs = list(preictal_chunk) if isinstance(preictal_chunk, tuple) else [preictal_chunk]
    for seg in pre_segs:
        f = seg["file"]
        local0, local1 = seg["start"], seg["end"]
        n_win = int(np.floor((local1 - local0) / window_len))
        abs0 = file_abs_starts[f] + local0
        for i in range(n_win):
            times_p.append(abs0 + i * window_len)
            files_p.append((f, local0 + i * window_len, "preictal"))

    print(f"times_i, start: {times_i[0]} , end:{times_i[-1]}")
    print(f"times_p = {times_p}")
    
    #find the boundary index of the window between files: 
    file_names = [f for (f,_,_) in files_i]
    # find every i where the file-name changes
    boundary_idxs = [
        i for i in range(1, len(file_names))
        if file_names[i] != file_names[i-1]
        ]
    
    #first preictal window, used for marking in the seizure plot
    first_pre_idx = len(times_i)
    
    window_times = np.array(times_i + times_p)
    window_files = files_i + files_p
    # now len(window_times) == len(window_files) == len(y_pred)

    # ————————————————
    # 2) Extract alarm events with file & time
    # ————————————————
    alarm_idxs = np.nonzero(y_pred.astype(bool))[0]
    print("alarm_idxs_len = ", len(alarm_idxs), "window_len = ", len(window_times))
    ####quick dirty fix -> could be that if alarm is in last window -> get error 
    N = len(window_files)
    alarm_idxs = alarm_idxs[ alarm_idxs < N ]
    ####quicck dirty fix end
    alarm_events = []
    for idx in alarm_idxs:
        f, local_t, segment = window_files[idx]
        abs_t = window_times[idx]
        alarm_events.append({
            "file": f,
            "segment": segment,
            "local_time": local_t,
            "abs_time": abs_t,
            "window-index": idx,
        })
    # ————————————————
    # 3) Compute seizure timestamp and valid‐window
    # ————————————————
    last = pre_segs[-1]
    t_seizure = file_abs_starts[last["file"]] + last["end"]
    valid_start = t_seizure - SOP
    valid_end   = t_seizure - SPH
    print(f"t_seizure = {t_seizure}")
    # ————————————————
    # 4) Sensitivity & lead time
    # ————————————————
    # grab only those alarms in the TP window
    tp_alarms = [e for e in alarm_events
                 if valid_start <= e["abs_time"] <= valid_end]
    sensitivity = float(len(tp_alarms) > 0)
    if sensitivity:
        lead_time = t_seizure - min(e["abs_time"] for e in tp_alarms)
    else:
        lead_time = np.nan

    # ————————————————
    # 5) False alarms (count + you already know exactly which)
    # ————————————————
    fa_alarms = [e for e in alarm_events
                 if e["segment"] == "interictal"
                 and not (valid_start <= e["abs_time"] <= valid_end)]
    fa = len(fa_alarms)

    # ————————————————
    # 6) Interictal seconds → for FPR
    # ————————————————
    total_interictal_seconds = sum((c["end"] - c["start"])
                                  for c in interictal_chunks)

    return sensitivity, fa, total_interictal_seconds, lead_time, alarm_events, boundary_idxs, first_pre_idx
