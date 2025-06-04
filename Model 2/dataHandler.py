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


#Note: Instead of reading the file summary, one could also directely read the information from the EDF files.
def parse_summary_file(summary_path):
    """
    Returns a list of dicts, each dict containing:
      {
        "file_name": ...,
        "start_time_of_file": datetime object for the file start,
        "end_time_of_file":   datetime object for the file end,
        "seizures": [
            {"start_seconds": int, "end_seconds": int}, ...
        ]
      }
    """
    records = []
    current_record = {}
    # track the last timestamp we saw, so we never go backwards
    last_ts = None

    def _parse_hms(tstr, base_day=1):
        # split into H, M, (optional) S
        parts = tstr.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else 0
        # build as day=base_day, then add any overflow from hours
        dt = datetime(2000, 1, base_day, 0, 0, 0) \
             + timedelta(hours=h, minutes=m, seconds=s)
        return dt

    with open(summary_path, "r") as f:
        day = 1
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("File Name:"):
                # flush previous
                if current_record:
                    records.append(current_record)
                current_record = {
                    "file_name": line.split("File Name:")[1].strip(),
                    "start_time_of_file": None,
                    "end_time_of_file": None,
                    "seizures": []
                }

            elif line.startswith("File Start Time:"):
                tstr = line.split("File Start Time:")[1].strip()
                ts = _parse_hms(tstr, base_day=day)
                # if this goes backwards, assume next day
                if last_ts is not None and ts <= last_ts:
                    day += 1
                    ts = _parse_hms(tstr, base_day=day)
                current_record["start_time_of_file"] = ts
                last_ts = ts

            elif line.startswith("File End Time:"):
                tstr = line.split("File End Time:")[1].strip()
                te = _parse_hms(tstr, base_day=day)
                # if end falls before start, push to next day
                if te < current_record["start_time_of_file"]:
                    day += 1
                    te = _parse_hms(tstr, base_day=day)
                current_record["end_time_of_file"] = te
                last_ts = te

            elif "Seizure" in line and "Start Time:" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    current_record["seizures"].append({
                        "start_seconds": int(m.group(1)),
                        "end_seconds": None
                    })

            elif "Seizure" in line and "End Time:" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m and current_record["seizures"]:
                    current_record["seizures"][-1]["end_seconds"] = int(m.group(1))

        # final flush
        if current_record:
            records.append(current_record)

    return records

#Extracts the preictal segments while considering the postictal duration for close to each other seizures
def extract_preictal_segments_distance(
    file_records,
    preictal_duration_min: float = 30,
    postictal_duration_min: float = 30,
    min_length_min: float = 20
):
    # convert to seconds
    P = preictal_duration_min * 60
    Q = postictal_duration_min * 60
    min_len = min_length_min * 60

    # sort files by absolute start
    file_records = sorted(file_records,
                          key=lambda r: r["start_time_of_file"])
    # annotate each record with its duration
    for r in file_records:
        r["duration"] = (r["end_time_of_file"] -
                         r["start_time_of_file"]).total_seconds()

    # 1) build a flat list of seizures with absolute times
    seizures = []
    for rec in file_records:
        base = rec["start_time_of_file"]
        for seiz in rec.get("seizures", []):
            abs_start = base + timedelta(seconds=seiz["start_seconds"])
            abs_end   = base + timedelta(seconds=seiz["end_seconds"])
            seizures.append({
                "rec": rec,
                "start_sec": seiz["start_seconds"],
                "end_sec":   seiz["end_seconds"],
                "abs_start": abs_start,
                "abs_end":   abs_end
            })
    # sort them by absolute seizure time
    seizures.sort(key=lambda s: s["abs_start"])

    # 2) filter out any seizure whose preictal window would overlap
    #    with the *postictal* of the previous kept seizure
    kept = []
    last_post_end = datetime.min
    for s in seizures:
        pre_start_abs = s["abs_start"] - timedelta(seconds=P)
        if pre_start_abs < last_post_end + timedelta(seconds=Q):
            # skip this seizure
            continue
        kept.append(s)
        last_post_end = s["abs_end"]

    # 3) now for each kept seizure, extract exactly the preictal
    out = []
    for s in kept:
        rec = s["rec"]
        t0  = s["start_sec"]
        if t0 >= P:
            # entirely in this file
            seg = {"file": rec["file_name"],
                   "start": t0 - P,
                   "end":   t0}
        else:
            idx = file_records.index(rec)
            if idx == 0:
                seg = {"file": rec["file_name"],
                       "start": 0,
                       "end":   t0}
            else:
                prev = file_records[idx-1]
                gap  = (rec["start_time_of_file"] -
                        prev["end_time_of_file"]).total_seconds()
                needed_prev = P - t0 - gap
                if needed_prev <= 0:
                    seg = {"file": rec["file_name"],
                           "start": 0, "end": t0}
                else:
                    seg = (
                        {
                          "file": prev["file_name"],
                          "start": max(prev["duration"] - needed_prev, 0),
                          "end":   prev["duration"]
                        },
                        {
                          "file": rec["file_name"],
                          "start": 0,
                          "end":   t0
                        }
                    )
        # 4) finally reject anything that doesn’t meet min_length
        '''
        Note: Here we discard segments that are shorter than min_len.
        If we want to keep them, we could remove this check, 
        -> in the postprocessing one would also have to count the positive prediction of the preceeding seizure as 2 correct predictions
           -> This is the case if we would remove the 2nd preictal since it doesn't meet min_length        
        '''
        length = (seg["end"] - seg["start"]) if isinstance(seg, dict) \
                 else sum(piece["end"]-piece["start"] for piece in seg)
        if length >= min_len:
            out.append(seg)

    return out


def helper_absolute_time(file_records):
    """
    Given a list of file records, compute the absolute start and end times for each file
    (in seconds from the first file’s start), *without* mutating the inputs.

    Parameters:
      file_records: list of dicts, each with keys
        - "file_name"
        - "start_time_of_file": datetime
        - "end_time_of_file":   datetime

    Returns:
      abs_times: dict mapping file_name -> {"abs_start": float, "abs_end": float}
    """
    # 1) sort by start time
    records_sorted = sorted(file_records, key=lambda rec: rec["start_time_of_file"])
    base_time = records_sorted[0]["start_time_of_file"]

    abs_times = {}
    for rec in records_sorted:
        dur = (rec["end_time_of_file"] - rec["start_time_of_file"]).total_seconds()
        abs_start = (rec["start_time_of_file"] - base_time).total_seconds()
        abs_end = abs_start + dur
        abs_times[rec["file_name"]] = {
            "abs_start": abs_start,
            "abs_end": abs_end
        }

    return abs_times
    
def chunk_absolute_times(records, tn_chunk):
    """
    records: list of {
       "file_name": str,
       "start_time_of_file": datetime,
       "end_time_of_file": datetime,
       …}
    tn_chunk: a tuple (interictal_list, preictal_chunk)
       - interictal_list: list of {"file","start","end"} for interictal
       - preictal_chunk: either dict or tuple of dicts for preictal

    Returns:
      {
        "interictal": [ (abs_start_dt, abs_end_dt), … ],
        "preictal":   [ (abs_start_dt, abs_end_dt), … ]
      }
    """
    # Build a lookup from filename → its record
    rec_map = { r["file_name"]: r for r in records }

    def _abs_times(seg):
        """Given a segment dict, return (dt_start, dt_end)."""
        rec = rec_map[seg["file"]]
        base = rec["start_time_of_file"]
        start_s = round(seg["start"])
        end_s   = round(seg["end"])
        return (base + timedelta(seconds=start_s),
                base + timedelta(seconds=end_s))

    interictal_list, preictal_chunk = tn_chunk

    # Compute for each interictal piece
    interictal_times = [ _abs_times(seg) for seg in interictal_list ]

    # Preictal may be a single dict or a tuple of dicts
    if isinstance(preictal_chunk, tuple):
        preictal_segs = list(preictal_chunk)
    else:
        preictal_segs = [preictal_chunk]
    preictal_times = [ _abs_times(seg) for seg in preictal_segs ]

    return {
        "interictal": interictal_times,
        "preictal":   preictal_times
    }
    

def extract_interictal_data(file_records,
                            preictal_duration_sec=30*60,    # duration in seconds that is considered preictal
                            interictal_buffer_sec=180*60,   # additional buffer (in sec) before seizure start
                            postictal_duration_min=30):     # postictal period in minutes
    """
    Extract interictal segments for one subject from multiple files.
    
    Each file record is a dictionary with:
       - "file_name": e.g. 'chb04_04.edf'
       - "start_time_of_file": a datetime object (absolute start)
       - "end_time_of_file": a datetime object (absolute end)
       - "seizures": a list of dictionaries; each seizure has:
             "start_seconds": seconds relative to file start
             "end_seconds": seconds relative to file start
             
    The data is considered interictal if it lies outside an "unsafe" window for any seizure.
    The unsafe window for a seizure is defined as:
         [ (seizure_abs_start - (preictal_duration_sec + interictal_buffer_sec)),
           (seizure_abs_end + postictal_duration_sec) ]
    where:
         seizure_abs_start = file_abs_start + seizure["start_seconds"]
         seizure_abs_end   = file_abs_start + seizure["end_seconds"]
         
    The function returns a dictionary mapping each file name to a list of segments,
    each segment being a dict with keys:
         - "start": local start time (in seconds relative to file start)
         - "end": local end time (in seconds relative to file start)
    
    Note: -> preictal_records is not directly used in this implementation because
          the unsafe intervals are computed using the seizure times.
          -> THe preictal_duration_sec is the buffer after the seizure, it's not the actual posictal of the seizure
    """
    
    # Use the start_time of the first file as the reference (base) time.
    # (Assumes file_records is nonempty and already sorted or sort them below.)
    file_records = sorted(file_records, key=lambda rec: rec["start_time_of_file"])
    base_time = file_records[0]["start_time_of_file"]
    
    # Compute duration in seconds for each file and add an absolute start time (in seconds from base).
    for rec in file_records:
        rec["duration"] = (rec["end_time_of_file"] - rec["start_time_of_file"]).total_seconds()
        rec["abs_start"] = (rec["start_time_of_file"] - base_time).total_seconds()
        rec["abs_end"] = rec["abs_start"] + rec["duration"]
    
    # Convert postictal period to seconds.
    postictal_duration_sec = postictal_duration_min * 60
    
    # Build list of unsafe intervals (in absolute seconds relative to base_time) from seizures.
    unsafe_intervals = []
    for rec in file_records:
        file_abs_start = rec["abs_start"]
        for seizure in rec.get("seizures", []):
            seizure_abs_start = file_abs_start + seizure["start_seconds"]
            seizure_abs_end = file_abs_start + seizure["end_seconds"]
            unsafe_start = seizure_abs_start - (preictal_duration_sec + interictal_buffer_sec)
            if unsafe_start < 0:
                unsafe_start = 0  # clip to start of subject recording
            unsafe_end = seizure_abs_end + postictal_duration_sec
            unsafe_intervals.append((unsafe_start, unsafe_end))
    
    # Sort unsafe intervals by start time and merge any that overlap.
    unsafe_intervals.sort(key=lambda interval: interval[0])
    merged_unsafe = []
    for interval in unsafe_intervals:
        if not merged_unsafe:
            merged_unsafe.append(interval)
        else:
            last = merged_unsafe[-1]
            # If intervals overlap or touch, merge them.
            if interval[0] <= last[1]:
                merged_unsafe[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged_unsafe.append(interval)
    
    # Define subject recording duration:
    subject_total_duration = file_records[-1]["abs_end"]
    
    # Compute safe (interictal) intervals as the complement of merged_unsafe in [0, subject_total_duration].
    safe_intervals = []
    prev_end = 0
    for (us, ue) in merged_unsafe:
        if us > prev_end:
            safe_intervals.append((prev_end, us))
        prev_end = max(prev_end, ue)
    if prev_end < subject_total_duration:
        safe_intervals.append((prev_end, subject_total_duration))
    
    # Now, map each safe interval back to each file.
    # We will build a dictionary mapping file_name to list of interictal segments (local times).
    result = { rec["file_name"]: [] for rec in file_records }
    
    for safe_start, safe_end in safe_intervals:
        # For each safe interval in absolute time, determine which files it overlaps.
        for rec in file_records:
            file_abs_start = rec["abs_start"]
            file_abs_end = rec["abs_end"]
            # If no overlap, skip.
            if safe_end <= file_abs_start or safe_start >= file_abs_end:
                continue
            # Compute the overlapping portion in absolute time.
            overlap_start_abs = max(file_abs_start, safe_start)
            overlap_end_abs = min(file_abs_end, safe_end)
            # Convert to local file time by subtracting file_abs_start.
            local_start = overlap_start_abs - file_abs_start
            local_end = overlap_end_abs - file_abs_start
            # Only add nonzero-length segments.
            if local_start < local_end:
                result[rec["file_name"]].append({"start": local_start, "end": local_end})
    
    return result


def flatten_interictal(interictal_dict):
    """
    Given a dictionary mapping file names to a list of interictal segments 
    (each segment is a dictionary with keys 'start' and 'end' in seconds, local to the file),
    flatten these segments into a continuous timeline.
    
    For simplicity, we assume the keys are sorted in increasing time order.
    This function returns a list of segments with the following keys:
     file       : file name (from which the segment comes)
     local_start: start in the file (seconds)
     local_end  : end in the file (seconds)
     length     : length of the segment
     global_start: start time on the continuous timeline (in seconds)
     global_end  : end time on the continuous timeline (in seconds)
         
    """
    files = sorted(interictal_dict.keys())
    flattened = []
    global_time = 0
    for f in files:
        for seg in interictal_dict[f]:
            seg_length = seg["end"] - seg["start"]
            flattened.append({
            "file": f,
            "local_start": seg["start"],
            "local_end": seg["end"],
            "length": seg_length,
            "global_start": global_time,
            "global_end": global_time + seg_length
            })
            global_time += seg_length
    return flattened, global_time

def partition_interictal(flattened, total_length, n): 
    """ Partition the continuous interictal timeline (flattened segments) into n contiguous intervals, 
    each of length total_length/n. If a partition boundary falls in the middle of a flattened segment that segment is split accordingly.
        
    Returns a list of n partitions, where each partition is a list of pieces. Each piece is a dictionary:
     file       : file name,
     start      : local start time (in seconds) of the piece,
     end        : local end time (in seconds) of the piece.
    """
    target = total_length / n
    partitions = []
    current_partition = []
    current_pos = 0  # how many seconds have we assigned in the current partition
    # Process the flattened segments in order.
    for seg in flattened:
        seg_remaining = seg["length"]
        local_offset = seg["local_start"]
        while seg_remaining > 0:
            needed = target - current_pos
            if seg_remaining <= needed + 1e-6:
                # Take entire remainder of this segment.
                current_partition.append({
                    "file": seg["file"],
                    "start": local_offset,
                    "end": seg["local_end"]
                })
                current_pos += seg_remaining
                seg_remaining = 0
            else:
                # We split the segment.
                piece_end = local_offset + needed
                current_partition.append({
                    "file": seg["file"],
                    "start": local_offset,
                    "end": piece_end
                })
                local_offset += needed
                seg_remaining -= needed
                current_pos += needed
            if current_pos >= target - 1e-6:
                # Partition complete.
                partitions.append(current_partition)
                current_partition = []
                current_pos = 0
                # If we've created n-1 partitions, the rest of the data goes to the last partition.
                if len(partitions) == n - 1:
                    break
        # If we already reached n - 1 partitions, break out after finishing the current segment.
        if len(partitions) == n - 1:
            break
    # Add any remaining pieces into the last partition.
    remaining = []
    # If we stopped in the middle of a segment, capture its remainder.
    if seg_remaining > 0:
        remaining.append({
            "file": seg["file"],
            "start": local_offset,
            "end": seg["local_end"]
        })
    # And include any remaining segments from the flattened list.
    idx = flattened.index(seg) + 1
    for s in flattened[idx:]:
        remaining.append({
            "file": s["file"],
            "start": s["local_start"],
            "end": s["local_end"]
        })
    # Append remaining pieces as the final partition.
    partitions.append(current_partition + remaining)
    return partitions

def combine_interictal_preictal(interictal_dict, preictal_list):
    """
    Given:
      interictal_dict: dictionary with keys as file names and values as lists of interictal segments
                       (each segment is a dict with 'start' and 'end' in seconds).
      preictal_list: a list of preictal segments (each is a dict or tuple; here we assume a dict with keys 
                      including 'file', 'start', and 'end').
                      
    Returns:
      A list of n pairs: each pair is a tuple (interictal_partition, preictal_segment).
      The interictal_partition is itself a list of pieces (each with a file name and local start/end).
    """

    # (1) Flatten interictal segments.
    flattened, total_time = flatten_interictal(interictal_dict)

    # (2) Let n be the number of preictal segments, (the number of seizures for the subject).
    n = len(preictal_list)
    if n == 0:
        raise ValueError("No preictal segments provided.")

    # (3) Partition the interictal timeline into n segments.
    partitions = partition_interictal(flattened, total_time, n)

    # (4) Randomly pair the partitions with the preictal segments.
    # Here we shuffle the preictal segments order.
    preictal_shuffled = preictal_list.copy()
    random.shuffle(preictal_shuffled)

    pairs = list(zip(partitions, preictal_shuffled))
    return pairs

#unused helper function, but ccn be used to generate the splits for the cross-validation manually
def generate_loocv_splits(pairs): 
    
    """ Given a list of pairs (each pair contains an interictal segment partition and a preictal segment),
    generate a list of splits. For each split, one pair is designated as the test set and the other pairs as training.
    Returns:

   A list of tuples: (train_pairs, test_pair)

    """
    splits = []
    n = len(pairs)
    for i in range(n):
       test_pair = pairs[i]
       train_pairs = pairs[:i] + pairs[i+1:]
       splits.append((train_pairs, test_pair))
    return splits


#Extract data from the annotated file pairs
def extract_segment_data(segment, base_path, present_channels):
    """ Given a segment dictionary (with keys: 'file': EDF file name, 'start': start time in seconds (relative to file start),
    'end': end time in seconds, ) extract the EEG data from that file over the specified time period.
    Returns:
     A numpy array of shape (n_channels, n_samples).
     """
    file_name = segment["file"]
    edf_file_path = Path(base_path) / file_name
    # Read EDF file (preloading data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(str(edf_file_path), preload=True, verbose=False)
   
    raw.pick_channels(present_channels, ordered=True) #only pick channels that appear in all recordings for that subject
    sfreq = raw.info["sfreq"]
    start_sample = int(segment["start"] * sfreq)
    end_sample = int(segment["end"] * sfreq)
    data = raw.get_data()[:, start_sample:end_sample]
    data = preprocess_eeg(data, sfreq)
    return data

def get_subject_channels(records, base_path):
    """
    Given summary-records with 'file_name', return the list of
    channels that appear in *every* EDF for this subject,
    excluding any whose name starts with '-' .
    """
    all_sets = []
    for rec in records:
        edf_path = Path(base_path) / rec["file_name"]
        # read header only
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        # filter out placeholder channels like “--0”, “--1”, etc.
        valid = {ch for ch in raw.ch_names if not ch.startswith('-')}
        all_sets.append(valid)

    if not all_sets:
        return []

    # intersection: channels present in every file
    common = set.intersection(*all_sets)
    return sorted(common)

def preprocess_eeg(
    sig: np.ndarray,
    fs: float,
    notch_freqs=(60.0, 120.0),
    notch_Q=30.0,
    highpass_freq: float = 0.5,
    bandpass: bool = False,
    bp_low: float = 0.5,
    bp_high: float = 60.0,
    bp_order: int = 4,
) -> np.ndarray:
    """
    Clean multi-channel EEG (n_ch × n_samples):
      • Optionally notch at f0 in notch_freqs
      • Then either:
         – band-pass between bp_low and bp_high (if bandpass=True)
         – or high-pass at highpass_freq (if bandpass=False)
    """

    # 1) Notch filters at each specified mains harmonic
    filtered = sig.copy()
    for f0 in notch_freqs:
        b_n, a_n = iirnotch(f0, notch_Q, fs)
        filtered = filtfilt(b_n, a_n, filtered, axis=-1)

    # 2) Band-pass or high-pass -> Some papers only consider frequencies up to 60Hz by taking a bandpass filter [0-60Hz]
    if bandpass:
        # design Butterworth band-pass
        nyq = fs/2
        low = bp_low  / nyq
        high = bp_high / nyq
        b_bp, a_bp = butter(bp_order, [low, high], btype='bandpass')
        filtered = filtfilt(b_bp, a_bp, filtered, axis=-1)
    else:
        # design Butterworth high-pass only, filter out DC noise at 0Hz
        nyq = fs/2
        wn  = highpass_freq / nyq
        b_hp, a_hp = butter(bp_order, wn, btype='highpass')
        filtered = filtfilt(b_hp, a_hp, filtered, axis=-1)

    return filtered


def extract_composite_data(chunks, base_path, present_channels): 
    """ Given a list of segments (each a dict with keys 'file', 'start', 'end'),
    extract the data for each segment and concatenate them along the time axis.
    Returns:
    A numpy array (concatenated data).
    """
    data_list = []
    for seg in chunks:
        seg_data = extract_segment_data(seg, base_path, present_channels)
        data_list.append(seg_data)
    if data_list:
        return np.concatenate(data_list, axis=1)
    else:
        return None
    
def process_tn_chunks(tn_chunks, base_path, present_channels):
    """ Given a list of t/n-chunks (each is a tuple of (interictal_segments, preictal_segments)
    where: interictal_segments: a list of dictionaries (each with 'file', 'start', 'end') preictal_segments:
    either a dictionary or a tuple of dictionaries (in the case the preictal spans multiple files)
    Extract the EEG data (using MNE) corresponding to the interictal and preictal segments.
    For segments spanning multiple files, their pieces are concatenated along time axis.
    
    Returns:
    Two lists: one for interictal data and one for preictal data. Each element is a 
    numpy array containing the extracted EEG segment.
    """
    interictal_data_list = []
    preictal_data_list = []

    for pair in tn_chunks:
        interictal_chunks, preictal_chunk = pair

        # Extract and concatenate interictal data for the current pair.
        inter_data = extract_composite_data(interictal_chunks, base_path, present_channels)
        interictal_data_list.append(inter_data)

        # Preictal chunk may be a single dictionary or a tuple (if spanning files)
        if isinstance(preictal_chunk, tuple):
            # Convert to list if needed
            preictal_segments = list(preictal_chunk)
        else:
            preictal_segments = [preictal_chunk]

        pre_data = extract_composite_data(preictal_segments, base_path, present_channels)
        preictal_data_list.append(pre_data)

    return interictal_data_list, preictal_data_list

#Performing the over/undersampling of the data -> Currently Not used
def sliding_windows(data: np.ndarray,
                    window_size: int,
                    stride: int) -> List[np.ndarray]:
    """
    Chop `data` (shape [n_chans, n_samples]) into overlapping windows.
    """
    n_samples = data.shape[1]
    windows = []
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data[:, start:end])
    return windows

def split_nonoverlap(data: np.ndarray,
                     window_size: int) -> List[np.ndarray]:
    """
    Chop `data` into non-overlapping windows of length window_size.
    """
    n_samples = data.shape[1]
    n_full = n_samples // window_size
    return [ data[:, i*window_size:(i+1)*window_size]
             for i in range(n_full) ]

def _build_window_times(segments, file_abs_starts, window_len):
    """
    Given a list of segments (each a dict with 'file','start','end'), and
    a lookup file_abs_starts[file_name] -> abs_start_time (sec),
    build the numpy array of window‐start times you get when you do
        split_nonoverlap(concat_data(segments), window_len)
    """
    # 1) segment durations (in sec):
    durs = [seg["end"] - seg["start"] for seg in segments]
    # 2) cumulative start‐positions (in sec) in the concatenated composite:
    cum = np.cumsum([0.0] + durs)      # len = len(segments)+1
    total = cum[-1]                    # total duration of composite
    # 3) number of full windows
    n_win = int(total // window_len)
    times = np.empty(n_win, dtype=float)

    for i in range(n_win):
        # position (sec) into the composite stream of window i
        pos = i * window_len
        # which segment is that in?
        k = np.searchsorted(cum, pos, side="right") - 1
        offset = pos - cum[k]                  # offset into segments[k]
        seg = segments[k]
        # absolute time = file's abs_start + local segment start + offset
        times[i] = file_abs_starts[seg["file"]] + seg["start"] + offset

    return times

#currently not used, but can be used to balance the data if over/undersamling is used to balance the data
def balance_one_pair(inter: np.ndarray,
                     pre: np.ndarray,
                     window_len: float,
                     sampling_rate: float = 0.25
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Given one interictal array and one preictal array:
      - oversample the preictal by sliding windows of its own length
        with stride = window_length * sampling_rate
      - split the interictal into non-overlapping windows of that same length
      - randomly discard interictal windows so you have exactly as many
        as the oversampled preictal windows.
    Returns two equally‑long lists of windows.
    """
    window_len *= 256 #convert seconds to samples
    stride = int(window_len * sampling_rate)
    # 1) oversample preictal
    pre_windows = sliding_windows(pre, window_len, stride)
    # 2) cut interictal into non-overlapping chunks
    inter_windows = split_nonoverlap(inter, window_len)
    # 3) undersample interictal to match count
    M = len(pre_windows)
    if len(inter_windows) > M:
        inter_windows = random.sample(inter_windows, M)
    # if inter_windows < M you could also oversample it, but paper only undersamples
    return inter_windows, pre_windows


#Methods to prep the dataset, balancing it if we want to use over/undersampling
def make_balance_dataset(train_pairs, window_len, sampling_rate):
    balanced = []
    for inter_data, pre_data in train_pairs:
        i_wins, p_wins = balance_one_pair(inter_data, pre_data,
                                          window_len=window_len,
                                          sampling_rate=sampling_rate)
        balanced.append((i_wins, p_wins))
    return balanced

class WindowDataset(Dataset):
    def __init__(self, windows: List[np.ndarray], labels: List[int]):
        self.windows = windows
        self.labels  = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # each window is a (C, L) numpy array
        w = torch.from_numpy(self.windows[idx]).float()
        y = self.labels[idx]
        return w, y

    # flatten + chronological split exactly as before
    
#Take the last val_frac of the training data as validation set each(for the interictal and preictal data)
def make_train_val_loaders(dataset,window_len, val_frac, batch_size): 
    
    window_len = int(window_len * 256) #convert seconds to samples
    inter, pre = [], []
    for inter_data, pre_data in dataset:
        i_wins = split_nonoverlap(inter_data, window_len)
        p_wins = split_nonoverlap(pre_data,   window_len)        
        
        inter.extend(i_wins)
        pre  .extend(p_wins)
    
    # split indices
    n_i, n_p = len(inter), len(pre)
    si = int(n_i * (1 - val_frac))
    sp = int(n_p * (1 - val_frac))

    # in‐place slicing of the Python lists
    i_train, i_val = inter[:si], inter[si:]
    p_train, p_val = pre[:sp],   pre[sp:]

    # build the datasets *without* stacking
    w_train = i_train + p_train
    y_train = [0]*len(i_train) + [1]*len(p_train)

    w_val   = i_val   + p_val
    y_val   = [0]*len(i_val)   + [1]*len(p_val)

    train_ds = WindowDataset(w_train, y_train)
    val_ds   = WindowDataset(w_val,   y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader

#Same as above but we have the already balanced dataset as input (#interictal windows == #preictal windows)
def make_train_val_loaders_last_balanced(balanced_dataset, val_frac, batch_size):
    inter, pre = [], []
    for i_wins, p_wins in balanced_dataset:
        inter.extend(i_wins)
        pre  .extend(p_wins)

    # split indices
    n_i, n_p = len(inter), len(pre)
    si = int(n_i * (1 - val_frac))
    sp = int(n_p * (1 - val_frac))

    # in‐place slicing of the Python lists
    i_train, i_val = inter[:si], inter[si:]
    p_train, p_val = pre[:sp],   pre[sp:]

    # build the datasets *without* stacking
    w_train = i_train + p_train
    y_train = [0]*len(i_train) + [1]*len(p_train)

    w_val   = i_val   + p_val
    y_val   = [0]*len(i_val)   + [1]*len(p_val)

    train_ds = WindowDataset(w_train, y_train)
    val_ds   = WindowDataset(w_val,   y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader

#this function takes the internal validation set randomly, this is our current function

def make_train_val_loaders_random_val(dataset, window_len, val_frac, batch_size, random_state=42):
   
   #Can use this if want to work with the balanced dataset
    # # 1) collect each class into flat lists
    # inter, pre = [], []
    # for i_wins, p_wins in balanced_dataset:
    #     inter.extend(i_wins)
    #     pre  .extend(p_wins)
    
    window_len = int(window_len * 256) #convert seconds to samples
    inter, pre = [], []
    for inter_data, pre_data in dataset:
        i_wins = split_nonoverlap(inter_data, window_len)
        p_wins = split_nonoverlap(pre_data,   window_len)        
        
        inter.extend(i_wins)
        pre  .extend(p_wins)

    # 2) build full window & label lists
    X = inter + pre
    y = [0] * len(inter) + [1] * len(pre)

    # 3) stratified random split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_frac,
        stratify=y,
        random_state=random_state
    )

    # 4) wrap into our WindowDataset
    train_ds = WindowDataset(X_tr, y_tr)
    val_ds   = WindowDataset(X_val, y_val)

    # 5) build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

def make_test_loader(test_pair, window_len, batch_size):
    # No oversample/undersampling in the test fold
    window_len *= 256 #convert seconds to samples
    
    inter_data, pre_data = test_pair
    i_wins = split_nonoverlap(inter_data, window_len)
    p_wins = split_nonoverlap(pre_data,   window_len)
    X_test = np.stack(i_wins + p_wins, axis=0)
    y_test = np.array([0]*len(i_wins) + [1]*len(p_wins), dtype=np.int64)
    
 
    test_ds     = TensorDataset(torch.from_numpy(X_test).float(),
                                torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader
