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
    
    with open(summary_path, "r") as f:
        day = 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse lines
            if line.startswith("File Name:"):
                # If there's a previous record, append it to records
                if current_record:
                    records.append(current_record)
                # Start a new record
                current_record = {
                    "file_name": line.split("File Name:")[1].strip(),
                    "start_time_of_file": None,
                    "end_time_of_file": None,
                    "seizures": []
                }
            elif line.startswith("File Start Time:"):
                time_str = line.split("File Start Time:")[1].strip()
                start_time = datetime.strptime(time_str, "%H:%M:%S").replace(day=day).replace(year=2000)
                current_record["start_time_of_file"] = start_time
            elif line.startswith("File End Time:"):
                time_str = line.split("File End Time:")[1].strip()
                hours = time_str[:2]
                if (time_str[1]!= ":"):
                    if(int(hours) >= 24):
                        # If the hours are greater than 24, it means the next day
                        hour_next_day = int(hours) - 24
                        time_str = str(hour_next_day) + ":" + time_str[3:]
                end_time = datetime.strptime(time_str, "%H:%M:%S").replace(day=day).replace(year=2000)
                if end_time < start_time:
                    end_time += timedelta(days=1)
                    day += 1
                current_record["end_time_of_file"] = end_time
            elif line.startswith("Number of Seizures in File:"):
                pass
            elif "Seizure" in line and "Start Time:" in line:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    start_s = int(match.group(1))
                    # Append a new seizure dict with an unset end time
                    current_record["seizures"].append({
                    "start_seconds": start_s,
                    "end_seconds": None
                    })

            # Example: "Seizure 1 End Time: 347 seconds"
            # or      "Seizure 2 End Time: 6231 seconds"
            elif "Seizure" in line and "End Time:" in line:
                match = re.search(r"(\d+)\s*seconds", line)
                if match and current_record["seizures"]:
                    end_s = int(match.group(1))
                    # Update the last seizure's end time
                    current_record["seizures"][-1]["end_seconds"] = end_s

        # Append the last record if any
        if current_record:
            records.append(current_record)
            
    for i in range(len(records) -1):
        if records[i]["end_time_of_file"] > records[i+1]["start_time_of_file"]:
         records[i+1]["start_time_of_file"] += timedelta(days=1)
         records[i+1]["end_time_of_file"] += timedelta(days=1)

    
    return records

def extract_preictal_segments(file_records, preictal_duration_min=30):
    """
    Extract preictal segments from multiple recordings for one subject.
    
    Parameters:
      file_records: a list of dictionaries, each having:
          - 'file_name'
          - 'start_time_of_file': datetime
          - 'end_time_of_file': datetime
          - 'seizures': list of dicts with keys 'start_seconds' and 'end_seconds'
      preictal_duration_min: duration in minutes to extract before a seizure.
    
    Returns:
      preictal_segments: a list of extracted segments.
         Each element is either:
           { 'file': <file_name>, 'start': <start_sec>, 'end': <end_sec> }
         or, for segments spanning two files, a tuple:
           ( { 'file': <prev_file>, 'start': <start_sec>, 'end': <prev_duration> },
             { 'file': <curr_file>, 'start': 0, 'end': <seizure_start> } )
    """
    preictal_duration = preictal_duration_min * 60  # convert minutes to seconds

    # Sort files by starting time
    file_records = sorted(file_records, key=lambda rec: rec["start_time_of_file"])
    
    # Compute duration for each file record (in seconds)
    for rec in file_records:
        rec["duration"] = (rec["end_time_of_file"] - rec["start_time_of_file"]).total_seconds()

    preictal_segments = []
    
    for i, rec in enumerate(file_records):
        seizures = rec.get("seizures", [])
        for seiz in seizures:
            t_seizure = seiz["start_seconds"]
            # Case 1: Entire preictal window is in the current file.
            if t_seizure >= preictal_duration:
                seg = { 'file': rec["file_name"],
                        'start': t_seizure - preictal_duration,
                        'end': t_seizure}
                preictal_segments.append(seg)
            else:
                # Case 2: Preictal spans from previous file into current file.
                if i == 0:
                    # No previous file available – use available part only.
                    seg = { 'file': rec["file_name"],
                            'start': 0,
                            'end': t_seizure}
                    preictal_segments.append(seg)
                else:
                    prev_rec = file_records[i-1]
                    # Compute gap between previous file's end and current file's start:
                    gap = (rec["start_time_of_file"] - prev_rec["end_time_of_file"]).total_seconds()
                    # Available from current file: t_seizure seconds.
                    # Needed remaining preictal time from previous file:
                    needed_prev = preictal_duration - t_seizure - gap
                    if (needed_prev <= 0):
                      # dont extract anything from prevous file, just use current file
                      seg = { 'file': rec["file_name"],
                                'start': 0,
                                'end': t_seizure}
                      preictal_segments.append(seg)
                    else:         
                      # Make sure we do not try to extract more than exists in previous file:
                      start_prev = max(0, prev_rec["duration"] - needed_prev)
                      seg_prev = { 'file': prev_rec["file_name"],
                                  'start': start_prev,
                                   'end': prev_rec["duration"]}
                      seg_curr = { 'file': rec["file_name"],
                                  'start': 0,
                                  'end': t_seizure}
                      preictal_segments.append((seg_prev, seg_curr))
    return preictal_segments

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
                            interictal_buffer_sec=180*60,    # additional buffer (in sec) before seizure start
                            postictal_duration_min=30):   # postictal period in minutes
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
    
    Note: preictal_records is not directly used in this implementation because
          the unsafe intervals are computed using the seizure times.
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

    # (2) Let n be the number of preictal segments.
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

#unused function, but can be used to generate the splits for the cross-validation
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

#Extract data from the annotaec file pairs
def extract_segment_data(segment, base_path):
    """ Given a segment dictionary (with keys: 'file': EDF file name, 'start': start time in seconds (relative to file start),
    'end': end time in seconds, ) extract the EEG data from that file over the specified time window.
    Returns:
     A numpy array of shape (n_channels, n_samples).
     """
    file_name = segment["file"]
    edf_file_path = Path(base_path) / file_name
    # Read EDF file (preloading data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(str(edf_file_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    start_sample = int(segment["start"] * sfreq)
    end_sample = int(segment["end"] * sfreq)
    data = raw.get_data()[:, start_sample:end_sample]
    return data

def extract_composite_data(chunks, base_path): 
    """ Given a list of segments (each a dict with keys 'file', 'start', 'end'),
    extract the data for each segment and concatenate them along the time axis.
    Returns:
    A numpy array (concatenated data).
    """
    data_list = []
    for seg in chunks:
        seg_data = extract_segment_data(seg, base_path)
        #Some subjects introduce ECG as the 24th channel, drop it if present
        if seg_data.shape[0] == 24:
            seg_data = seg_data[:23, :] 
        data_list.append(seg_data)
    if data_list:
        return np.concatenate(data_list, axis=1)
    else:
        return None
    
def process_tn_chunks(tn_chunks, base_path):
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
        inter_data = extract_composite_data(interictal_chunks, base_path)
        interictal_data_list.append(inter_data)

        # Preictal chunk may be a single dictionary or a tuple (if spanning files)
        if isinstance(preictal_chunk, tuple):
            # Convert to list if needed
            preictal_segments = list(preictal_chunk)
        else:
            preictal_segments = [preictal_chunk]

        pre_data = extract_composite_data(preictal_segments, base_path)
        preictal_data_list.append(pre_data)

    return interictal_data_list, preictal_data_list

#Performing the over/undersampling of the data
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


#Methods to prep the dataset
def make_balance_dataset(train_pairs, window_len, sampling_rate):
    balanced = []
    for inter_data, pre_data in train_pairs:
        i_wins, p_wins = balance_one_pair(inter_data, pre_data,
                                          window_len=window_len,
                                          sampling_rate=sampling_rate)
        balanced.append((i_wins, p_wins))
    return balanced

#More efficient compiutation of the dataset
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
def make_train_val_loaders_last_frac(balanced_dataset, val_frac, batch_size):
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

def make_train_val_loaders(balanced_dataset, val_frac, batch_size, random_state=42):
    """
    balanced_dataset: list of (i_wins, p_wins) pairs, where each is a list of
                      numpy arrays shape (C, L)
    val_frac:        fraction of examples (of each class) to reserve for validation
    """
    # 1) collect each class into flat lists
    inter, pre = [], []
    for i_wins, p_wins in balanced_dataset:
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
    def __init__(self, eeg_channels=23, eeg_out_channels=16, num_classes=2):
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
        self.fcn = nn.Linear(16, num_classes)

    def forward(self, x1):
        out1 = self.net1(x1)
        out = self.fcn(out1)
        return out

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

def firing_power(preds: np.ndarray, n: int) -> np.ndarray:
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

def risk_levels(fp: np.ndarray,
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

import numpy as np

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



def evaluate_seizure_fold_SPH_SOP(
    y_pred,            # 1D array of 0/1 alarms, length == # windows in test_loader
    interictal_chunks, # list of {'file', 'start', 'end'} for the test‐fold interictal
    preictal_chunk,    # dict (or tuple of dicts) for the held‑out preictal
    window_len,        # seconds
    SPH,               # seconds
    SOP,               # seconds
    file_abs_starts    # dict: file_name → absolute start (sec)
):
    import numpy as np

    # 1) Build window_times for interictal & preictal exactly as test_loader does
    times_i = _build_window_times(interictal_chunks, file_abs_starts, window_len)
    print(f"times_i, start: {times_i[0]} , end:{times_i[-1]}")

    # put preictal into a list
    if isinstance(preictal_chunk, tuple):
        pre_segs = list(preictal_chunk)
    else:
        pre_segs = [preictal_chunk]
    times_p = _build_window_times(pre_segs, file_abs_starts, window_len)
    print(f"times_p = {times_p}")
    # concatenation order matches: i_wins + p_wins
    window_times = np.concatenate([times_i, times_p])
    # len(window_times) == len(y_pred) guaranteed

    # 4) Now index safely:
    alarm_times = window_times[y_pred.astype(bool)]
    print(f"alarm_times = {alarm_times}")

    # 3) The seizure really happens at the END of the preictal segment:
    if isinstance(preictal_chunk, tuple):
        last = preictal_chunk[-1]
    else:
        last = preictal_chunk
    t_seizure = file_abs_starts[ last["file"] ] + last["end"]
    print(f"t_seizure = {t_seizure}")

    # 4) Define the “valid” prediction window [t_s − SOP, t_s − SPH]
    valid_start = t_seizure - SOP # -SPH  , but here we dont use it since len(preictal) = len(SOP)
    valid_end   = t_seizure - SPH
    print(f"valid_start = {valid_start}, valid_end = {valid_end}")

    # 5) Sensitivity: did we fire at least once in that interval? 
    #
    in_window = alarm_times[(alarm_times >= valid_start) &
                            (alarm_times <= valid_end)]
    sensitiviy = float(len(in_window) > 0)


    # 6) Count false alarms: alarms outside that interval but _inside_ any interictal chunk
    fa = 0
    for t in alarm_times:
        if not (valid_start <= t <= valid_end):
            for c in interictal_chunks:
                abs0 = file_abs_starts[c["file"]] + c["start"]
                abs1 = file_abs_starts[c["file"]] + c["end"]
                if abs0 <= t < abs1:
                    fa += 1
                    break
    print(f"fa = {fa}")

    # 7) Compute FPR per hour
    total_interictal_seconds = sum(
        (c["end"] - c["start"]) for c in interictal_chunks)    
    
    #Lead time: seizure_time − earliest alarm in window
    if sensitiviy:
        first_alarm = in_window.min()       # earliest correct alarm
        lead_time = t_seizure - first_alarm # seconds before seizure
    else:
        lead_time = np.nan                  # or 0.0

    return sensitiviy, fa, total_interictal_seconds, lead_time

def run():
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score
    
    base_path = Path('/usr/scratch/sassauna1/sem25f13')
    subject = '01'
    preictal_duration_min = 30 #What timespan before a seizure is considered preictal
    interictal_preictal_buffer_min = 240 #What buffer time between preictal and interictal is used, ignore this data
    postictal_duration_min = 180 #What timespan after a seizure is considered postictal, ignore this data
    records = parse_summary_file(str(base_path / f'chb{subject}/chb{subject}-summary.txt'))
    
    interictal = extract_interictal_data(records,preictal_duration_sec=preictal_duration_min*60,   
                                           interictal_buffer_sec=interictal_preictal_buffer_min*60,  
                                           postictal_duration_min=postictal_duration_min) 
    preictal = extract_preictal_segments(records)
    abs_times_per_file = helper_absolute_time(records)
    file_abs_starts = { fn: info["abs_start"] for fn, info in abs_times_per_file.items() }
    print("Extracted interictal and preictal times")
    tn_chunks = combine_interictal_preictal(interictal, preictal)
    interictal_data_list, preictal_data_list = process_tn_chunks(tn_chunks, str(base_path / f'chb{subject}'))
    print("processed interictal and preictal segments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparams
    n_chunks      = len(interictal_data_list)
    window_len = 20    # seconds
    sampling_rate = 0.25
    val_frac = 0.25
    batch_size = 32
    epochs = 30
    patience = 20
    lr = 1e-3
    weight_decay = 1e-5
    # for k-of-n filtering
    k = 15
    n = 20
    # for refractory period
    R = 60 #min
    # for evaluation
    SPH = 5  # min
    SOP = 30  # min

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

    for fold in range(n_chunks):
        # ==== 1) Prepare the data for this fold ====
        # 1a) test chunk
        print(f"Fold {fold+1}/{n_chunks}")
        test_pair = (interictal_data_list[fold], preictal_data_list[fold])
        test_loader = make_test_loader(test_pair, window_len, batch_size)
        print(f"Test: X={test_loader.dataset.tensors[0].shape}, y={test_loader.dataset.tensors[1].shape}")

        # 1b) balance & pool the other n-1 chunks
        train_pairs = [
            (interictal_data_list[i], preictal_data_list[i])
            for i in range(n_chunks) if i != fold
        ]
        balanced = make_balance_dataset(train_pairs, window_len, sampling_rate)
        print(f"balanced Train: {len(balanced)} pairs")
        train_loader, val_loader = make_train_val_loaders_last_frac(
            balanced, val_frac=val_frac, batch_size=batch_size)

        print("Val & Train loader created")
        # ==== 2) Compute μ/σ *only on train* (streaming) ====
        mu, std = compute_fold_stats(train_loader, batch_size=batch_size)
        mu, std = mu.to(device), std.to(device)

        print("Data normalization computed")

        # ==== 3) Build model, optimizer, loss ====
        model = FeaStackedSensorFusion_eeg_only().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #Test class‐weighted loss
        #pos_weight = len(interictal_data_list[0][0]) / len(preictal_data_list[0][0])
        #print(f"Pos weight: {pos_weight:.2f}")
        #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))
        criterion = torch.nn.CrossEntropyLoss()

        # ==== 4) Train with early‑stopping on val ==== 
        # DO either via val_loss or val_acc
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
                Xb = Xb.unsqueeze(1)    # now Xb: (batch,  1, 23, 2560)

                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

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

            # ---- print them ----
            print(f"[Fold {fold} | Epoch {epoch:03d}]  "f"TrainLoss={avg_train_loss:.4f}  ValLoss={avg_val_loss:.4f}  "
            f"ValAcc={val_acc:.4f}  ValAUC={val_auc:.4f}  "f"Sens={val_sens:.4f} Spec={val_spec:.4f}  FPR={val_fpr:.4f}")

            
            # early‑stop check, choose one of the two:
            if  avg_val_loss < best_val_loss: #val_acc > best_val_acc:
                best_val_loss = avg_val_loss
                #best_val_acc = val_acc
                wait = 0
                torch.save(model.state_dict(), f"best_fold{fold}.pt")
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
            
            ##Just for debugging, check on every eopch how it performs on the test set
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
            y_pred_kn = k_of_n_filter(y_pred_kn, k=k, n=n)
            y_pred_post = apply_refractory(window_len, y_pred_kn, R=R)
            

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

            print(f"[Fold {fold:2d}] AUC={auc:.3f}, Sens={sens:.3f},Spec={spec:.3f},  FPR={fpr:.3f} , Acc={acc:.3f}")
            print(f"[Fold {fold}] post→   AUC={auc:.3f},  Sens={sens_post:.3f}, Spec={spec_post:.3f}, FPR={fpr_post:.3f}, Acc={acc_post:.3f}")
        ######## Just for debbugging ends \#######################
            
        #Notify that training is done
        print(f"[Fold {fold:2d}] Training done. Best val acc : {best_val_acc} Best val loss: {best_val_loss:.4f}")

        # ==== 5) Load best model & test ====
        model.load_state_dict(torch.load(f"best_fold{fold}.pt"))
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
        y_pred_kn = k_of_n_filter(y_pred_kn, k=k, n=n)
        y_pred_post = apply_refractory(window_len=window_len, alarm_seq=y_pred_kn, R=R)
        
        #probabalistic predictions, seizure risk levels/firing power
        y_pred_riskLevels = firing_power(y_pred, n=n)

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

        print(f"[Fold {fold:2d}] AUC={auc:.3f}, Sens={sens:.3f},Spec={spec:.3f},  FPR={fpr:.3f} , Acc={acc:.3f}")
        print(f"[Fold {fold}] post→   AUC={auc:.3f},  Sens={sens_post:.3f}, Spec={spec_post:.3f}, FPR={fpr_post:.3f}, Acc={acc_post:.3f}")
        all_fold_metrics.append((auc, sens, spec, fpr, acc))
        all_fold_metrics_post.append((auc, sens_post, spec_post, fpr_post, acc_post))
        
        sens, fp, interictal_seconds, lead_time, alarms,boundry_window_idx, first_preictal_window_idx = evaluate_seizure_fold_SPH_SOP_complex(
        y_pred_post,
        tn_chunks[fold][0],   # intericral segments
        tn_chunks[fold][1],   # preictal segments
        window_len=window_len,  
        SPH=SPH*60,            
        SOP=SOP*60,            
        file_abs_starts=file_abs_starts  # dict: file_name → absolute start (sec)
        )
        print(f"interictal files: {tn_chunks[fold][0]}, and preictal: {tn_chunks[fold][1]}")
        print(f"Fold {fold}: sens={sens:.0f}, fp={fp}, interictal_h={interictal_seconds / 3600:.2f}h, FPR={(fp/interictal_seconds) * 3600:.2f}, lead_time={lead_time:.1f}s")
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
        test_results_post.append({
            "fold": fold,
            "epoch": 0,               
            "test_loss": avg_test_loss,  
            "test_auc": auc,
            "test_sens": sens_post,
            "test_spec": spec_post,
            "test_fpr": fpr_post,
            "test_acc": acc_post
        })
        #get window index from alarm events
        alarm_idx = [e["window-index"] for e in alarms]
        #calculate the starting times of the chunks: 
        times = chunk_absolute_times(records, tn_chunks[fold])
        
        #append the risk levels to the list
        y_pred_forecast.append({"fold": fold,
                                "first_preictal_window_idx": first_preictal_window_idx,
                                "boundary_window_idx": boundry_window_idx,
                                "alarm_window_idx": alarm_idx,
                                "window_size": window_len, #in seconds -> use for plot afterwards
                                "risk": y_pred_riskLevels,
                                "start_end_times_of_files": times                                                 
                                })
        
        

    # ==== 6) Aggregate across folds ====
    all_fold_metrics = np.array(all_fold_metrics)  # shape (n_chunks, 3)
    mean_auc, mean_sens, mean_spec, mean_fpr, mean_acc = all_fold_metrics.mean(axis=0)
    std_auc, std_sens, std_spec, std_fpr, std_acc = all_fold_metrics.std(axis=0)
    #Check postprocessed metrics
    all_fold_metrics_post = np.array(all_fold_metrics_post)  # shape (n_chunks, 3)
    mean_auc_post, mean_sens_post, mean_spec_post, mean_fpr_post, mean_acc_post = all_fold_metrics_post.mean(axis=0)
    std_auc_post, std_sens_post, std_spec_post, std_fpr_post, std_acc_post = all_fold_metrics_post.std(axis=0)

    print("\n=== Summary across folds ===")
    print(f"AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Sensitivity = {mean_sens:.3f} ± {std_sens:.3f}") 
    print(f"Specificity = {mean_spec:.3f} ± {std_spec:.3f}")
    print(f"False‑alarm rate = {mean_fpr:.3f} ± {std_fpr:.3f}")
    print(f"Acc   = {mean_acc:.3f} ± {std_acc:.3f}")
    
    print("\n=== Summary across folds postprocessing ===")
    print(f"AUC = {mean_auc_post:.3f} ± {std_auc_post:.3f}")
    print(f"Sensitivity = {mean_sens_post:.3f} ± {std_sens_post:.3f}")
    print(f"Specificity = {mean_spec_post:.3f} ± {std_spec_post:.3f}")
    print(f"False‑alarm rate = {mean_fpr_post:.3f} ± {std_fpr_post:.3f}")
    print(f"Acc   = {mean_acc_post:.3f} ± {std_acc_post:.3f}")
    
    # after all folds:
    overall_sens = total_sens              # number of seizures correctly predicted
    overall_fpr  = total_fa / total_ih if total_ih > 0 else float('nan')     # false alarms per hour
    mean_lead    = float(np.mean(lead_times)) if lead_times else float('nan')  #mean of lead prediction time
    std_lead     = float(np.std( lead_times)) if lead_times else float('nan')  #std of lead prediction time

    print(f"\n\n=== Summary across folds SPH/SOP ===")
    print(f"Seizures predicted: {overall_sens}/{n_chunks}")
    print(f"Total false alarms: {total_fa}")
    print(f"Total interictal hours: {total_ih / 3600 :.2f}h")
    print(f"Overall FPR: {overall_fpr * 3600 :.4f} per hour")
    print(f"Mean lead time:     {mean_lead / 60:.2f} ± {std_lead / 60:.2f} minutes")

    df_training_history = pd.DataFrame(training_history)
    df_test_results = pd.DataFrame(test_results)
    df_test_results_post = pd.DataFrame(test_results_post)
    df_risk_levels = pd.DataFrame(y_pred_forecast)
    df_test_results = pd.concat([df_test_results, df_test_results_post], axis=0)
    df_training_history.to_csv("training_history.csv", index=False)
    df_test_results.to_csv("test_results.csv", index=False)
    df_risk_levels.to_csv("risk_levels.csv", index=False)
    print("Saved metrics to training_history.csv")
    
    # Write the same summary into a text file
    summary_txt = []
    summary_txt.append(f"Subject: {subject} ,Window Size: {window_len} s, Dist Interical/Preictal: {interictal_preictal_buffer_min} min, Postictal duration: {postictal_duration_min} Epochs: {epochs}, Batch size: {batch_size}, k: {k}, n: {n}, R: {R}min, SPH: {SPH}min, SOP: {SOP}min")
    summary_txt.append("=== Summary across folds ===")
    summary_txt.append(f"AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    summary_txt.append(f"Sensitivity = {mean_sens:.3f} ± {std_sens:.3f}")
    summary_txt.append(f"Specificity = {mean_spec:.3f} ± {std_spec:.3f}")
    summary_txt.append(f"False‑alarm rate = {mean_fpr:.3f} ± {std_fpr:.3f}")
    summary_txt.append(f"Acc   = {mean_acc:.3f} ± {std_acc:.3f}")
    summary_txt.append("")  # blank line
    summary_txt.append("=== Summary across folds postprocessing ===")
    summary_txt.append(f"AUC = {mean_auc_post:.3f} ± {std_auc_post:.3f}")
    summary_txt.append(f"Sensitivity = {mean_sens_post:.3f} ± {std_sens_post:.3f}")
    summary_txt.append(f"Specificity = {mean_spec_post:.3f} ± {std_spec_post:.3f}")
    summary_txt.append(f"False‑alarm rate = {mean_fpr_post:.3f} ± {std_fpr_post:.3f}")
    summary_txt.append(f"Acc   = {mean_acc_post:.3f} ± {std_acc_post:.3f}")
    summary_txt.append("")  # blank line
    summary_txt.append(f"Across {n_chunks} folds:")
    summary_txt.append(f"  Seizures predicted: {overall_sens}/{n_chunks}")
    summary_txt.append(f"  Total false alarms: {total_fa}")
    summary_txt.append(f"  Total interictal hours: {total_ih/3600:.2f}h")
    summary_txt.append(f"  Overall FPR: {overall_fpr*3600:.2f} per hour")
    summary_txt.append(f"  Mean lead time: {mean_lead/60:.2f} ± {std_lead/60:.2f} minutes")
    #Summary about false positives/behaviour of the model
    summary_txt.append("=== Per File summary ===")
    for fold in range(n_chunks):
        summary_txt.append(f"===Used Files in Fold {fold}:====")
        summary_txt.append(f"==Interictal files: ")
        for c in tn_chunks[fold][0]:
            summary_txt.append(f"File: {c['file']}: start: {c['start']} end: {c['end']}")
        summary_txt.append(f"==Preictal files: ")
        if(len(tn_chunks[fold][1]) == 2):
            summary_txt.append(f"File: {tn_chunks[fold][1][0]['file']}: start: {tn_chunks[fold][1][0]['start']} end: {tn_chunks[fold][1][0]['end']}")
            summary_txt.append(f"File: {tn_chunks[fold][1][1]['file']}: start: {tn_chunks[fold][1][1]['start']} end: {tn_chunks[fold][1][1]['end']}")
        else: 
            summary_txt.append(f"File: {tn_chunks[fold][1]['file']}: start: {tn_chunks[fold][1]['start']} end: {tn_chunks[fold][1]['end']}")
        summary_txt.append(f"==Total Alarms in This fold: {len(total_alarms[fold])}")
        for e in total_alarms[fold]:
            summary_txt.append(f"Alarm in {e['segment']:10s} {e['file']:15s} at local {e['local_time']:.1f}s (abs {e['abs_time']:.1f})")

    path = "final_metrics.txt"
    # If the file exists, append; otherwise create it (write)
    mode = "a" if os.path.exists(path) else "w"

    with open(path, mode) as fout:
        # If appending, add a leading blank line to separate entries
        if mode == "a":
            fout.write("\n")
        fout.write("\n".join(summary_txt))
        fout.write("\n")

    print("Wrote summary metrics to final_metrics.txt")

if __name__ == "__main__":
    run()