from matplotlib import pyplot as plt
import pandas as pd
from io import StringIO

# Function to format the DataFrame into a table with rounded values and mean±std
def make_formatted_table(df):
    # 1) per‐subject formatting
    df_fmt = pd.DataFrame({
        "Subject": df["subject"],
        "AUC": df["mean_auc"].round(3).astype(str) + " ± " + df["std_auc"].round(3).astype(str),
        "Sensitivity": df["mean_sens"].round(3).astype(str) + " ± " + df["std_sens"].round(3).astype(str),
        "Specificity": df["mean_spec"].round(3).astype(str) + " ± " + df["std_spec"].round(3).astype(str),
        "FPR": df["mean_fpr"].round(3).astype(str) + " ± " + df["std_fpr"].round(3).astype(str),
        "Accuracy": df["mean_acc"].round(3).astype(str) + " ± " + df["std_acc"].round(3).astype(str),
        "Overall Sens.": df["overall_sens_seizures"].round(3),
        "False Alarms": df["total_fa"],
        "Interictal h": df["total_ih"].round(2),
        "FPR/h": df["overall_fpr"].round(3),
        "Lead (min)": df["mean_lead"].round(2).astype(str)
                       + " ± " 
                       + df["std_lead"].round(2).astype(str)
    })

    # 2) compute overall mean±std
    numeric_cols = [
        "mean_auc","mean_sens","mean_spec","mean_fpr","mean_acc",
        "overall_sens_seizures","total_fa","total_ih","overall_fpr","mean_lead"
    ]
    means = df[numeric_cols].mean()
    stds  = df[numeric_cols].std()

    overall = {
        "Subject": "ALL",
        "AUC": f"{means['mean_auc']:.3f} ± {stds['mean_auc']:.3f}",
        "Sensitivity": f"{means['mean_sens']:.3f} ± {stds['mean_sens']:.3f}",
        "Specificity": f"{means['mean_spec']:.3f} ± {stds['mean_spec']:.3f}",
        "FPR": f"{means['mean_fpr']:.3f} ± {stds['mean_fpr']:.3f}",
        "Accuracy": f"{means['mean_acc']:.3f} ± {stds['mean_acc']:.3f}",
        "Overall Sens.": means["overall_sens_seizures"].round(3),
        "False Alarms": means["total_fa"].round(1),
        "Interictal h": means["total_ih"].round(2),
        "FPR/h": means["overall_fpr"].round(3),
        "Lead (min)": f"{means['mean_lead']:.2f} ± {stds['mean_lead']:.2f}"
    }

    # 3) append overall row
    return pd.concat([df_fmt, pd.DataFrame([overall])], ignore_index=True)


# list of (title, path) for each table, this are the result files after running the traiing & model
base_path = "path/to/Summaries/"

tables = [("Title of Run", "per_subject_summary_title_of_run_defined_before_training_loop_in_model.csv"),]


def plot_all(base_path=base_path, tables=tables):
    for title, fname in tables:
        # 1) load & format
        df = pd.read_csv(base_path + fname)
        df_tbl = make_formatted_table(df)

        # 2) make a fresh figure & axis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # 3) draw the table
        tbl = ax.table(
            cellText=df_tbl.values,
            colLabels=df_tbl.columns,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)

        # 4) title & show
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # give a little padding at the top so suptitle doesn't overlap table:
        fig.subplots_adjust(top=0.85)
        fig.tight_layout()
    plt.show()  # non-blocking show for interactive mode

#Used to compare against the SoA Models
def make_simple_summary(df):
    # 1) only keep the subjects you had in your table, in that order
    subjects = [1,2,3,5,8,9,10,13,14,16,17,18,19,20,21,23]
    df = df[df.subject.isin(subjects)].copy()
    df['subject'] = pd.Categorical(df.subject, categories=subjects, ordered=True)
    df = df.sort_values('subject')

    # 2) compute the three metrics, formatting exactly as in your first table
    #    AUC and Sens are multiplied by 100 and shown with 1 decimal
    #    FPR/hour is already per-hour, shown with 3 decimals
    summary = pd.DataFrame({
        'Pt': df['subject'].astype(str),
        'AUC (%)': (df['mean_auc'] * 100).round(1),
        'Sₙ (%)':  (df["overall_sens_seizures"]* 100).round(1),
        'FPR/h':   df['overall_fpr'].round(3),
    })

    # 3) add the “Aver” row at the bottom
    mean_vals = summary[['AUC (%)','Sₙ (%)','FPR/h']].mean()
    aver = {
        'Pt': 'Aver',
        'AUC (%)': round(mean_vals['AUC (%)'], 1),
        'Sₙ (%)':  round(mean_vals['Sₙ (%)'], 1),
        'FPR/h':   round(mean_vals['FPR/h'], 3),
    }
    return pd.concat([summary, pd.DataFrame([aver])], ignore_index=True)


def plot_simple(base_path=base_path,tables=tables):
    for title, fname in tables:
        df = pd.read_csv(base_path + fname)
        tbl_df = make_simple_summary(df)

        fig, ax = plt.subplots(figsize=(1, 6))
        ax.axis('off')
        table = ax.table(
            cellText=tbl_df.values,
            colLabels=tbl_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.tight_layout()
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.subplots_adjust(top=0.85)
    plt.show()



#decide if big or small plot (More or less subjects)
plot_all(base_path, tables)
#plot_simple(base_path, tables)

#Plot the imbalance ration roughly 
def imbalance_ratio():
    # Data                    
    subjects = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23]
    seizures = [7,3,7,4,5,10,3,5,4,7,2,10,8,17,9,3,6,3,8,4,3,7]
    interictal_h = [15.87,25.61,28.39,133.35,14.96,26.15,51.32,1.74,48.59,25.78,
                    32.00,15.30,5.21,1.33,5.14,14.16,26.48,27.00,18.57,23.87,17.56,15.22]

    # Compute imbalance (rounded)
    imbalance = [h/(sz*0.5) for h, sz in zip(interictal_h, seizures)]

    # Prepare table cell text (string formatted)
    rows = [[str(s), str(sz), f"{h:.2f}", f"{ib:.2f}"] 
            for s, sz, h, ib in zip(subjects, seizures, interictal_h, imbalance)]

    # Plot table
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 6))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows, 
        colLabels=['Subject','Seizures','Interictal h','Imbalance'],
        cellLoc='center', 
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    plt.tight_layout()
    plt.show()