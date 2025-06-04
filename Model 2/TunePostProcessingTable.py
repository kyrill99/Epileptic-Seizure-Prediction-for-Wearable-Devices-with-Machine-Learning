from matplotlib import pyplot as plt
import pandas as pd
import ast
import re
import numpy as np
from postProcessing import k_of_n_filter, apply_refractory, evaluate_seizure_fold_SPH_SOP_complex

def parse_int_list(s: str) -> list:
    """Extracts all integers from a bracketed, space- or comma-separated string."""
    return list(map(int, re.findall(r'\d+', s)))

def postprocess_subject(
    csv_path: str,
    k: int,
    n: int,
    R_min: float,
    SPH_min: float,
    SOP_min: float
) -> dict:
    """
    Loads per-fold predictions CSV, applies post-processing,
    and returns overall metrics for a single subject.
    """
    # 1) Load and parse CSV
    df = pd.read_csv(csv_path, quoting=0, skipinitialspace=True, engine='python')
     # 2) Parse columns manually
    df['window_predictions'] = df['window_predictions'].apply(parse_int_list)
    df['risk'] = df['risk'].apply(parse_int_list)
    # other columns already comma-separated JSON-like strings can still use ast if needed
    df['interictal_chunks'] = df['interictal_chunks'].apply(eval)
    df['preictal_chunks'] = df['preictal_chunks'].apply(eval)
    df['file_abs_starts'] = df['file_abs_starts'].apply(eval)
    window_len = df['window_size'][0] #winlength same for all folds
    # Initialize accumulators
    total_sens = 0
    total_fa = 0
    total_interictal_sec = 0
    lead_times = []

    # 2) Loop through each fold
    for _, row in df.iterrows():
        # raw window predictions
        y_pred = row['window_predictions']

        # k-of-n filter + refractory
        y_pred_kn = k_of_n_filter(y_pred, k=k, n=n)
        y_pred_post = apply_refractory(window_len, y_pred_kn, R=R_min)

        # evaluate fold
        sens, fp, interictal_sec, lead_time, alarms, boundry_window_idx, first_preictal_window_idx = evaluate_seizure_fold_SPH_SOP_complex(
            y_pred_post,
            row['interictal_chunks'],
            row['preictal_chunks'],
            window_len=window_len,
            SPH=SPH_min * 60,
            SOP=SOP_min * 60,
            file_abs_starts=row['file_abs_starts']
        )
        total_sens += sens
        total_fa += fp
        total_interictal_sec += interictal_sec
        if not np.isnan(lead_time):
            lead_times.append(lead_time)

    # 3) Compute overall metrics
    n_seizures = len(df)  # one preictal fold per seizure
    overall_sensitivity = total_sens / n_seizures
    overall_fpr = total_fa / (total_interictal_sec / 3600)  # false alarms per hour
    mean_lead = np.mean(lead_times)
    std_lead = np.std(lead_times)

    return {
        'overall_seizure_sensitivity': overall_sensitivity,        
        'overall_FPR_per_hour': overall_fpr,
        'total_false_alarms': total_fa,
        'mean_lead_time_s': mean_lead,
        'std_lead_time_s': std_lead
    }

k=6
n=10
R_min=30
SPH_min=5
SOP_min=30
#subjects = ['01', '02','03','04','05','06','07','08','09','10','11','13','14','15','16','17','18','19','20','21','22','23']
subjects = ['01', '02','03','05','08','09','10','13','14','16','17','18','19','20','21','23']
runs = [0, 1, 2, 3]
path = 'Risk Levels/All_Channels_Win5_Val10/'

subject_stats = []
for subj in subjects:
    run_metrics = []
    for run in runs:
        csv_path = path + f'All_Channels_Win5_Val10_risk_levels_Subject_{subj}_run{run}.csv'
        metrics = postprocess_subject(
            csv_path,
            k, n,
            R_min, SPH_min, SOP_min
        )
        run_metrics.append(metrics)
    df_runs = pd.DataFrame(run_metrics)
    mean_vals = df_runs.mean()
    std_vals  = df_runs.std()
    # Convert lead time from seconds to minutes
    mean_lead_min = mean_vals['mean_lead_time_s'] / 60
    std_lead_min  = std_vals['mean_lead_time_s'] / 60

    subject_stats.append({
        'Subject': subj,
        'Overall Sens.':       f"{mean_vals['overall_seizure_sensitivity']:.3f} ± {std_vals['overall_seizure_sensitivity']:.3f}",
        'FPR/h':               f"{mean_vals['overall_FPR_per_hour']:.3f} ± {std_vals['overall_FPR_per_hour']:.3f}",
        'False Alarms':        f"{mean_vals['total_false_alarms']:.2f} ± {std_vals['total_false_alarms']:.2f}",
        'Lead (min)':          f"{mean_lead_min:.3f} ± {std_lead_min:.3f}"
    })

# Compute average across subjects
df_subj = pd.DataFrame(subject_stats)
# To compute numeric averages, parse back the mean part from each cell
numeric = {
    'Overall Sens.': [],
    'FPR/h': [],    
    'False Alarms': [],
    'Lead (min)': []
}
for col in numeric:
    for cell in df_subj[col]:
        mean_val = float(cell.split('±')[0].strip())
        numeric[col].append(mean_val)
avg_stats = {col: f"{pd.Series(vals).mean():.3f} ± {pd.Series(vals).std():.3f}"
             for col, vals in numeric.items()}

# Append average row
avg_row = {'Subject': 'Average'}
avg_row.update(avg_stats)
df_subj = df_subj._append(avg_row, ignore_index=True)

# Plot table
col_labels = ['Subject', 'Overall Sens.', 'FPR/h', 'False Alarms', 'Lead (min)']
cell_text = df_subj[col_labels].values.tolist()

fig, ax = plt.subplots(figsize=(12, 0.5 * len(df_subj) + 1))
ax.axis('off')
tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
plt.show()