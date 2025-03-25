import pandas as pd # type: ignore
import numpy as np # type: ignore
import click
import json

def get_reference_metric(df, ref):

    try:
        kp = 1/df.loc[ref, 'Kr']
        ki = kp/df.loc[ref, 'Tn']
        kd = kp* df.loc[ref, 'Tv']

        # Compute natural frequency (approximation)
        omega_n = np.sqrt(df.loc[ref, 'Kr'] / (df.loc[ref, 'Tn'] * df.loc[ref, 'Tv']))

        # Compute steady-state error for different inputs
        ess_step = 1 / (1 + df.loc[ref, 'Kr'])  # Step input

        # Compute IAE and ITAE
        IAE = 1.5 / omega_n
        ITAE = 1.2 / (omega_n ** 2)

    except:
        print("check this value, can't not be zero")
        
    ref_metrics= {
        'kr' : df.loc[ref, 'Kr'],
        'Tn' : df.loc[ref, 'Tn'],
        'Tv' : df.loc[ref, 'Tv'],
        'PR' : df.loc[ref, 'PR'],
        'RT' : df.loc[ref, 'RT'],
        'kp' : kp,
        'ki' : ki,
        'kd' : kd,
        'ess_step': ess_step,
        'IAE': IAE,
        'ITAE': ITAE,
    }

    return ref_metrics


def get_kp(df):

    df['kp'] = 0

    for index, row in df.iterrows():

        try:
            kp = 1/row['Kr']
            df['kp'] = df['kp'].astype(float)
            df.at[index, 'kp'] = float(round(kp,3))

        except:
            e=9
    return df

def get_mt_metrics(df):
    # Ensure required columns exist and are float before modifying them
    for col in ['ki', 'kd', 'ess_step', 'IAE', 'ITAE', 'Warning']:
        if col not in df.columns:
            df[col] = np.nan  # Initialize missing columns
    
    df[['ki', 'kd', 'ess_step', 'IAE', 'ITAE']] = df[['ki', 'kd', 'ess_step', 'IAE', 'ITAE']].astype(float)
    
    for index, row in df.iterrows():
        try:
            warning_message = ""

            ki, kd, omega_n, ess_step, IAE, ITAE = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            if row['Tn'] != 0 and row['Tv'] != 0:
                ki = row['kp'] / row['Tn']
                kd = row['kp'] * row['Tv']

                # Compute natural frequency (approximation)
                omega_n = np.sqrt(row['Kr'] / (row['Tn'] * row['Tv']))

                # Compute steady-state error for step input
                ess_step = 1 / (1 + row['Kr'])

                # Compute IAE and ITAE
                IAE = 1.5 / omega_n
                ITAE = 1.2 / (omega_n ** 2)

            elif row['Tn'] == 0 and row['Tv'] != 0:
                kd = row['kp'] * row['Tv']
                warning_message = "Warning: Tn is zero, Ki cannot be computed."

            elif row['Tn'] != 0 and row['Tv'] == 0:
                ki = row['kp'] / row['Tn']
                warning_message = "Warning: Tv is zero, Kd cannot be computed."

            else:
                warning_message = "Warning: Both Tn and Tv are zero. No computations possible."

            # Store computed values in DataFrame
            df.at[index, 'ki'] = round(float(ki), 3) if not np.isnan(ki) else np.nan
            df.at[index, 'kd'] = round(float(kd), 3) if not np.isnan(kd) else np.nan
            df.at[index, 'ess_step'] = round(float(ess_step), 3) if not np.isnan(ess_step) else np.nan
            df.at[index, 'IAE'] = round(float(IAE), 3) if not np.isnan(IAE) else np.nan
            df.at[index, 'ITAE'] = round(float(ITAE), 3) if not np.isnan(ITAE) else np.nan
            df.at[index, 'Warning'] = warning_message  # Store warning message for debugging

        except Exception as e:
            print("***********************************************************")
            print(f"Error at index {index}: {e}")
            print("***********************************************************")

    return df

def mr_rt_vs_pr_faster_than_base(df):
    
    "If the RT is lower, should stay in the range"

    base_rt = df.at['tc100_def_0', 'RT']
    base_pr = df.at['tc100_def_0', 'PR']

    df['mr_rt_vs_pr_f'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_rt >= row['RT']:
                
                if base_pr <= row['PR'] <= 1.05:
                    df.at[index, 'mr_rt_vs_pr_f'] = 1
                
                else:
                    df.at[index, 'mr_rt_vs_pr_f'] = 0
    
    return df

def mr_rt_vs_pr_lower_than_base(df):
    
    "If the RT is lower, should stay in the range"

    base_rt = df.at['tc100_def_0', 'RT']
    base_pr = df.at['tc100_def_0', 'PR']

    df['mr_rt_vs_pr_l'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_rt <= row['RT']:
                
                if base_pr <= row['PR'] <= 1.05:
                    df.at[index, 'mr_rt_vs_pr_l'] = 1
                
                else:
                    df.at[index, 'mr_rt_vs_pr_l'] = 0
    
    return df

def mr_eng_p(df):

    "If Kp increase the related energy should increase o remains equal"

    base_kp = df.at['tc100_def_0', 'kp']
    base_eng_p = df.at['tc100_def_0', 'P-Eng']/df.at['tc100_def_0', 'S-Eng']

    df['mr_eng_p'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kp < row['kp']:
                
                if base_eng_p <= (row['P-Eng']/row['S-Eng']):
                    df.at[index, 'mr_eng_p'] = 1
                
                else:
                    df.at[index, 'mr_eng_p'] = 0
    return df
        
def mr_kp_rt(df):

    "Increasing Kp should reduce the RT"

    base_kp = df.at['tc100_def_0', 'kp']
    base_rt = df.at['tc100_def_0', 'RT']

    df['mr_kp_rt'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kp < row['kp']:
                
                if base_rt >= row['RT']:
                    df.at[index, 'mr_kp_rt'] = 1
                
                else:
                    df.at[index, 'mr_kp_rt'] = 0
    return df

def mr_kpki(df):

    base_kp = df.at['tc100_def_0', 'kp']
    base_ki = df.at['tc100_def_0', 'ki']
    base_rt = df.at['tc100_def_0', 'RT']

    df['mr_kpki'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kp >= row['kp'] and base_ki >= row['ki']:
                
                if base_rt >= row['RT']:
                    df.at[index, 'mr_kp_rt'] = 1
                
                else:
                    df.at[index, 'mr_kp_rt'] = 0
    return df

def mr_kp_ess(df):

    base_kp = df.at['tc100_def_0', 'kp']
    base_ess_step = df.at['tc100_def_0', 'ess_step']

    df['mr_kp_ess'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kp < row['kp'] :
                
                if base_ess_step <= row['ess_step']:
                    df.at[index, 'mr_kp_ess'] = 1
                
                else:
                    df.at[index, 'mr_kp_ess'] = 0    
    return df

def mr_tn_ess(df):

    base_Tn = df.at['tc100_def_0', 'Tn']
    base_ess_step = df.at['tc100_def_0', 'ess_step']

    df['mr_Tn_ess'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_Tn < row['Tn'] :
                
                if base_ess_step <= row['ess_step']:
                    df.at[index, 'mr_kp_ess'] = 1
                
                else:
                    df.at[index, 'mr_kp_ess'] = 0    
    return df

def mr_kd_ki(df):

    base_kd = df.at['tc100_def_0', 'kd']
    base_ki = df.at['tc100_def_0', 'ki']

    base_eng =  base_eng_p = df.at['tc100_def_0', 'P-Eng']/df.at['tc100_def_0', 'S-Eng']

    df['mr_kd_ki'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kd < row['kd'] and base_ki < row['ki']:
                
                if base_eng <= row['P-Eng']/row["S-Eng"]:
                    df.at[index, 'mr_kd_ki'] = 1
                
                else:
                    df.at[index, 'mr_kd_ki'] = 0    
    return df

def mr_kp_kd(df):

    base_kd = df.at['tc100_def_0', 'kd']
    base_kp = df.at['tc100_def_0', 'kp']
    base_rt = df.at['tc100_def_0', 'RT']

    df['mr_kp_kd'] = 'n/a'

    for index, row in df.iterrows():
        if index != 'tc100_def_0':

            if base_kp < row['kp'] and base_kd > row['kd']:
                
                if base_rt >= row['RT']:
                    df.at[index, 'mr_kp_kd'] = 1
                
                else:
                    df.at[index, 'mr_kp_kd'] = 0    
    return df

def analysis(df):
    # List of target columns
    target_columns = ['mr_rt_vs_pr_f','mr_rt_vs_pr_l','mr_eng_p','mr_kp_rt','mr_kpki','mr_kp_ess','mr_Tn_ess','mr_kd_ki','mr_kp_kd']

    # Count occurrences row-wise
    df['total_1'] = df[target_columns].apply(lambda row: (row == 1).sum(), axis=1)
    df['total_0'] = df[target_columns].apply(lambda row: (row == 0).sum(), axis=1)
    df['total_na'] = df[target_columns].apply(lambda row: (row == 'n/a').sum(), axis=1)

    return df

@click.command()
@click.argument('file_path', type=click.Path(exists=True))

def init(file_path):
 
    global based_metrics # Glabal dictionary to get the based metric in this case the starting parameter set :)  

    with open(file_path, 'r') as file:
        data = json.load(file)

    df_data = pd.DataFrame.from_dict(data, orient="index")
    df_data = get_kp(df_data)


    df_data = get_mt_metrics(df_data)

    # df_data.set_index("test_id", inplace=True)
    # df = df_data.reset_index().rename(columns={"index": "test_id"}, inplace=True)

    df_data = df_data.drop(columns=['tc', 'top_score', 'score_relative', 
                         'top_score_apr_a', 'score_apr_a_relative', 
                          'top_score_apr_b', 'score_apr_b_relative',
                          'top_score_apr_c', 'score_apr_c_relative',
                          'score', 'score_apr_a', 'score_apr_b',
                          'score_apr_c', 'top_score_def', 'def_OR', 'def_GR', 'OR', 'GR', "time",
                          "pressure",
                          "speed",
                          "pressureT",
                          "speedT"])
    
    df_data = mr_rt_vs_pr_faster_than_base(df_data)
    df_data = mr_rt_vs_pr_lower_than_base(df_data)
    df_data = mr_eng_p(df_data)
    df_data = mr_kp_rt(df_data)
    df_data = mr_kpki(df_data)
    df_data = mr_kp_ess(df_data)
    df_data = mr_tn_ess(df_data)
    df_data = mr_kd_ki(df_data)
    df_data = mr_kp_kd(df_data)
    df_data = analysis(df_data)
    df=  df_data.sort_values(by='total_1', ascending=False)
    df.to_csv('out_sorting_by_total.csv')

    # Identify unique combinations of the specified columns
    unique_combinations = df[['mr_rt_vs_pr_f','mr_rt_vs_pr_l','mr_eng_p','mr_kp_rt','mr_kpki','mr_kp_ess','mr_Tn_ess','mr_kd_ki','mr_kp_kd']].drop_duplicates()

    # Assign a unique group number to each combination
    unique_combinations = unique_combinations.reset_index(drop=True)
    unique_combinations['group'] = unique_combinations.index + 1

    # Merge back with the original DataFrame to assign group numbers
    df = df.merge(unique_combinations, on=['mr_rt_vs_pr_f','mr_rt_vs_pr_l','mr_eng_p','mr_kp_rt','mr_kpki','mr_kp_ess','mr_Tn_ess','mr_kd_ki','mr_kp_kd'], how='left')
    # based_metrics = get_reference_metric(df_data)
    # get_mt_metrics(df_data)
    df.to_csv('grupe.csv')

if __name__ == "__main__":
    init()



