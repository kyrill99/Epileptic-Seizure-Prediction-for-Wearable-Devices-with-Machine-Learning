import matplotlib.pyplot as plt
import pandas as pd
import TunePostProcessing as helpers
from matplotlib.offsetbox import AnchoredText

configs = {
    '(K=1 & N=10)': {
        'path': 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/',
        'k': 1, 'n': 10, 'R_min': 25, 'SPH_min': 5, 'SOP_min': 30
    },
    '(K=3 & N=10)': {
        'path': 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/',
        'k': 3, 'n': 10, 'R_min': 25, 'SPH_min': 5, 'SOP_min': 30
    },
    '(K=5 & N=10)': {
        'path': 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/',
        'k': 5, 'n': 10, 'R_min': 25, 'SPH_min': 5, 'SOP_min': 30
    },
    '(K=8 & N=10)': {
        'path': 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/',
        'k': 8, 'n': 10, 'R_min': 25, 'SPH_min': 5, 'SOP_min': 30
    },
    '(K=10 & N=10)': {
        'path': 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/',
        'k': 10, 'n': 10, 'R_min': 25, 'SPH_min': 5, 'SOP_min': 30
    }
}


#subjects = ['01','02','03','05','08','09','10','13','14','16','17','18', '20','21','23']
subjects = ['01', '02','03','04','05','06','07','08','09','10','11','13','14','15','16','17','18','19','20','21','22','23']
runs = [0, 1] #depents on how many runs we ran our models
path = 'Risk Levels/Window_05_Val_10_Lr4_drop05_Epoch32/'
Val = 10   #just for the plot
Window = 5 #just for the plot


# 2) Compute average across subjects for each config
config_results = []
for cfg_name, params in configs.items():
    subj_sens = []
    subj_fpr  = []
    for subj in subjects:
        # Collect metrics for both runs
        run_metrics = []
        for run in runs:
            csv_path = (f"{path}Window_05_Val_10_Lr4_drop05_Epoch32_"
                        f"risk_levels_Subject_{subj}_run{run}.csv")
            m = helpers.postprocess_subject(
                csv_path,
                k=params['k'], n=params['n'],
                R_min=params['R_min'],
                SPH_min=params['SPH_min'],
                SOP_min=params['SOP_min']
            )
            run_metrics.append(m)
        # average per subject
        df_runs = pd.DataFrame(run_metrics)
        subj_sens.append(df_runs['overall_seizure_sensitivity'].mean())
        subj_fpr.append(df_runs['overall_FPR_per_hour'].mean())
    # average across subjects
    config_results.append({
        'config': cfg_name,
        'mean_sens': 100 * pd.Series(subj_sens).mean(),   # %
        'mean_fpr': pd.Series(subj_fpr).mean()            # per hour
    })

df_cfg = pd.DataFrame(config_results)

x = range(len(df_cfg))
width = 0.4

fig, ax1 = plt.subplots(figsize=(8,4))

# Sensitivity on the left axis
bars1 = ax1.bar(
    [i - width/2 for i in x],
    df_cfg['mean_sens'],
    width,
    label='Event Sensitivity (%)',
    color='C0'
)
ax1.set_ylabel('Event Sensitivity (%)', color='C0')
ax1.set_ylim(60, 106)
ax1.tick_params(axis='y', labelcolor='C0')

# Label each sensitivity bar with its height
ax1.bar_label(bars1, fmt='%.1f', padding=3, color='C0')

# Create a second y-axis, sharing the same x
ax2 = ax1.twinx()
bars2 = ax2.bar(
    [i + width/2 for i in x],
    df_cfg['mean_fpr'],
    width,
    label='FPR (/h)',
    color='C1'
)
ax2.set_ylabel('FPR (/h)', color='C1')
ax2.set_ylim(0.3, df_cfg['mean_fpr'].max() * 1.5)
ax2.tick_params(axis='y', labelcolor='C1')

# Label each FPR bar with its height
ax2.bar_label(bars2, fmt='%.2f', padding=3, color='C1')

# X-labels
ax1.set_xticks(x)
ax1.set_xticklabels(df_cfg['config'], rotation=20, ha='right')

# Title and legend
fig.suptitle(f'K-N Majority-Vote Effects on Model Performance', fontsize=14)

params = f"Val. Size: {Val}%\nWindow: {Window} s"
at = AnchoredText(params,
                  prop=dict(size=10),
                  frameon=True,
                  loc='upper right',
                  pad=0.5,
                  bbox_to_anchor=(0.9, 0.85),
                  bbox_transform=fig.transFigure)
at.patch.set_edgecolor("gray")
at.patch.set_alpha(0.2)
fig.add_artist(at)
# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines+lines2, labels+labels2, loc='upper left')

fig.tight_layout()
plt.show()