import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


    
def parse_array(s):
    # Drop the leading/trailing quotes & brackets
    txt = s.strip().lstrip('"').rstrip('"').strip("[]")
    # Use fromstring to parse the whitespace-separated floats
    return np.fromstring(txt, sep=" ")

def parse_times_dict(s: str):
    """
    s looks like:
      "{'interictal': [(datetime.datetime(2000, 1, 1, 18, 8, 6), ...)], 'preictal': [...]}"
    We eval it with a namespace where only datetime.datetime is allowed.
    """
    namespace = {"datetime": datetime}
    return eval(s, namespace)

def first_interictal_hms(times_dict):
    # times_dict["interictal"] is a list of (dt_start, dt_end) pairs
    dt_start, dt_end = times_dict['interictal'][0]
    return dt_start.strftime("%H:%M:%S"), dt_end.strftime("%H:%M:%S")

def first_preictal_hms(times_dict):
    dt_start, dt_end = times_dict['preictal'][0]
    return dt_start.strftime("%H:%M:%S"), dt_end.strftime("%H:%M:%S")


def plot_risks(window_size, file_times,firing_power, boundary_window_idx ,preictal_start, low_thr=0.3, high_thr=0.7, alarm_idxs=None):
    fig, ax = plt.subplots(figsize=(12,4))
    
    times = np.arange(len(firing_power))
    firing_power = np.array(firing_power)
    
    interictal_times = file_times['interictal']
    preictal_start_time = file_times['preictal'][0][0].strftime("%H:%M:%S")
    
    ticks_left = [interictal_times[0][0].strftime("%H:%M:%S")]
    ticks_right = []
    
    for i in range(len(interictal_times)):
        ticks_left.append(interictal_times[i][1].strftime("%H:%M:%S"))        
        if i > 0 : ticks_right.append(interictal_times[i][0].strftime("%H:%M:%S"))
    
    ticks_right.append(preictal_start_time)
    
    tick_positions_left = [0]    
    tick_positions_right = []
    
    for wd_idx in boundary_window_idx:
        tick_positions_left.append(wd_idx)
        tick_positions_right.append(wd_idx)
    
    tick_positions_left.append(int(preictal_start))
    tick_positions_right.append(int(preictal_start))
    
    #add 1 to the right ticks to get the next window so matplot can show both times (start and end)
    tick_positions_right = [pos + 1 for pos in tick_positions_right]

    # 1) The continuous curve
    ax.plot(times, firing_power,
            color="black", linewidth=1.5,
            label="Firing Power")
    
    # 2) The three risk‐level fills, each with its label
    ax.fill_between(times, 0, firing_power,
                    where=(firing_power <= low_thr),
                    facecolor="green",    alpha=0.3,
                    label="Low risk")
    
    ax.fill_between(times, 0, firing_power,
                    where=(firing_power > low_thr) & (firing_power <= high_thr),
                    facecolor="yellow",   alpha=0.3,
                    label="Moderate risk")
    
    ax.fill_between(times, 0, firing_power,
                    where=(firing_power > high_thr),
                    facecolor="red",      alpha=0.3,
                    label="High risk")
    
    # 3) Threshold lines (optional to label these)
    ax.axhline(low_thr,  linestyle="--", color="grey", label="_nolegend_")
    ax.axhline(high_thr, linestyle="--", color="grey", label="_nolegend_")
    
    # 4) If you also want to highlight the true preictal segment
    #    add one more fill and label before the risk‐fills:
    # ax.fill_betweenx([0,1], preictal_start, preictal_end,
    #                  facecolor="orange", alpha=0.2,
    #                  label="Preictal period")
    #mark the start of the preictal period
    ax.axvline(preictal_start,
               color="blue",
               linestyle="--",
               linewidth=2,
               label="Preictal start")
    
    #mark the alarms
    if alarm_idxs is not None and len(alarm_idxs):
        # get the risk score at each alarm index
        alarm_times = np.array(alarm_idxs)
        alarm_scores = firing_power[alarm_idxs]
        ax.scatter(alarm_times, alarm_scores,
                   marker="v", s=100,
                   color="cyan",
                   edgecolor="k",
                   label="Alarms")
    
    # 5) Formatting
    ax.set_xlabel("Time ")
    ax.set_ylabel("Seizure risk")
    ax.set_title("Seizure Prediction: Risk Levels for imminent Seizure")
    ax.set_ylim(0, 1.05)
    
    print("Ticks left:", ticks_left)
    print("Ticks right:", ticks_right)
    print("Tick positions left:", tick_positions_left)
    print("Tick positions right:", tick_positions_right)
    # Set the x‐axis ticks to the boundary times, if uncomment -> get time as windows
    # plt.xticks(tick_positions_left, ticks_left, rotation=45, ha="right")
    # plt.xticks(tick_positions_right, ticks_right, rotation=45, ha="left")
    
    pos_lbl = []
    for p, lbl in zip(tick_positions_left,  ticks_left):
        pos_lbl.append((p, lbl, "right", 45))
    for p, lbl in zip(tick_positions_right, ticks_right):
        pos_lbl.append((p, lbl, "left", -45))

    # sort so ticks go in increasing x
    pos_lbl.sort(key=lambda x: x[0])

    print(pos_lbl)
    # unpack
    positions, labels, aligns, rots = zip(*pos_lbl)

    print(positions)
    print(labels)
    # set them all at once
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    
    print("X ticks:", ax.get_xticklabels())

    # now tweak the Text objects’ horizontal‐alignment
    for txt, ha, rot in zip(ax.get_xticklabels(), aligns, rots):
        txt.set_ha(ha)          
        txt.set_rotation(rot)
    
    
    
    # 6) Legend
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    fig.tight_layout()

def main(path, fold):
    df = pd.read_csv(f"path_to_preictal_risk_levels/{path}.csv")
    #fold = 6
    low_thr = 0.3
    high_thr = 0.7
    #Extract from csv and transform to int, list, etc
    preictal_start = df["first_preictal_window_idx"][fold]
    window_size = df["window_size"][fold]
    df["alarm_idxs_parsed"]= df["alarm_window_idx"].apply(ast.literal_eval)
    df["boundry_window_idx_parsed"] = df["boundary_window_idx"].apply(ast.literal_eval)                        
    df["risk_array"] = df["risk"].apply(parse_array)  
    #take desired fold
    preictal_start = df.loc[fold, "first_preictal_window_idx"]
    boundary_window_idx = df.loc[fold, "boundry_window_idx_parsed"]  # now a list of ints
    print("Boundary window idx:", boundary_window_idx)
    alarm_idxs     = df.loc[fold, "alarm_idxs_parsed"]  # now a list of ints
    array = df.loc[fold, "risk_array"]
    
    #Take the times of the interictal and preictal periods
    df["times_dict"] = df["start_end_times_of_files"].apply(parse_times_dict)
    interictal_preictal_times = df["times_dict"][fold]
    
    df[["int1_start", "int1_end"]] = df["times_dict"]\
    .apply(first_interictal_hms)\
    .tolist()
    
    print("First interictal start:", df["int1_start"][fold])
    
    
    plot_risks(window_size, interictal_preictal_times,array, boundary_window_idx, low_thr=low_thr, high_thr=high_thr, preictal_start=preictal_start, alarm_idxs=alarm_idxs)

if __name__ == "__main__":
    fold = 0
    #subject = 3
    path = "Validation_10_Lr4_Win20_drop05_Epoch32_risk_levels_Subject_07_run0"
    main(path, fold=fold)
    plt.show()
   