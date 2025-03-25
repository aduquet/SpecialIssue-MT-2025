from get_metrics import getting_metrics
from mr_kp_ki_kd import mrs
import pandas as pd
import numpy as np
import pathlib
import json
import os
import matplotlib.pyplot as plt

def get_test_case_id(tc: str) -> str:
    return tc.split("_")[0]


def get_pid_metric(kr, Tn, Tv):

    # for index, row in df.iterrows():

    try:
        kp = float(1/kr)
        ki = float(kp/Tn)
        kd = float(kp* Tv)
    except:
        e=0
    return kp, ki, kd

# def compare(val, ref):
#     if val > ref:
#         return 'h'
#     elif val < ref:
#         return 'l'
#     else:
#         return 'e'
    
with open('prueba.json', 'r') as file:
    data = json.load(file)

df_data = pd.DataFrame.from_dict(data, orient="index")
df_data = df_data.reset_index().rename(columns={"index": "test_id"})
df_data['tc'] = None
df_data['id'] = None

for index, row in df_data.iterrows():
    tc_id = get_test_case_id(row["test_id"])
    id = int(row["test_id"].split("_")[-1])
    df_data.at[index, "tc"] = tc_id
    df_data.at[index, "id"] = id
    kp, ki, kd = get_pid_metric(row['Kr'], row['Tn'], row['Tv'])
    df_data.at[index,'kp'] = kp
    df_data.at[index,'ki'] = ki
    df_data.at[index,'kd'] = kd

df_data = getting_metrics(df_data)

df_data =  df_data[df_data['tc'] == 'tc100']

# final_data = df_data[[ 'tc', 'test #', 'kp', 'kd', 'ki', 
#                       'ess_step', 'IAE', 'ITAE','Kr', 'Tn', 'Tv', 
#                       'PR', 'RT', 'P-Eng','S-Eng', 'Eng-rel','osc_p', 'osc_s']]

# ref_kp = final_data.loc[0, 'kp']
# ref_ki = final_data.loc[0, 'ki']
# ref_kd = final_data.loc[0, 'kd']

# for index, row in final_data.iterrows():
#     code = (
#         compare(row['kp'], ref_kp) +
#         compare(row['ki'], ref_ki) +
#         compare(row['kd'], ref_kd)
#     )
#     final_data.at[index,'parameter_code'] = code

# final_data.to_csv('final_data.csv')

final_data = pd.read_csv('final_data.csv')
tc100 =  final_data[final_data['tc'] == 'tc100']

ref_rt = final_data.loc[0, 'RT']
# ref_eng = final_data.loc[0, 'Eng-rel']
ref_eng = final_data.loc[0,'S-Eng']
ref_osc = final_data.loc[0, 'osc_p']
ref_IAE = final_data.loc[0, 'IAE']

for index, row in tc100.iterrows():
    tc100.at[index,'eng-diff'] =ref_eng - row['S-Eng']

ref_eng_diff = tc100.loc[0, 'eng-diff']

for index, row in tc100.iterrows():
    if row['parameter_code'] == 'lll':
        tc100.at[index,'mr_lll_RT'] = mrs.mr_lll_one(ref_rt, row['RT'])
        tc100.at[index,'mr_lll_EngS'] = mrs.mr_lll_two(ref_eng, row['S-Eng'])
        tc100.at[index,'mr_lll_osc'] = mrs.mr_lll_tree(row['osc_p'], row['osc_s'])
        tc100.at[index,'mr_IAE'] = mrs.mr_AIE(ref_IAE, row['IAE'])
        tc100.at[index,'mr_PR'] = mrs.mr_PR(row['PR'])

for index, row in tc100.iterrows():
    if row['parameter_code'] == 'lll':
        tc100.at[index,'total_one'] = row['mr_lll_RT'] + row['mr_IAE']+row['mr_lll_EngS'] + row['mr_lll_osc'] +row['mr_PR'] 
# for index, row in final_data.iterrows():
#     tc_id = get_test_case_id(row["test_id"])
#     id = int(row["test_id"].split("_")[-1])
#     df_data.at[index, "tc"] = tc_id
#     df_data.at[index, "id"] = id
#     kp, ki, kd = get_pid_metric(row['Kr'], row['Tn'], row['Tv'])
#     df_data.at[index,'kp'] = kp
#     df_data.at[index,'ki'] = ki
#     df_data.at[index,'kd'] = kd

tc100_pr = tc100[tc100['mr_PR'] == 1]

tc100_list = list(set(tc100_pr['total_one']))
print(tc100_list)
print(tc100_pr['total_one'].value_counts())

for i in tc100_list:

    folder_path = str(pathlib.Path().absolute()) + "\\" + "lll-MRs-" + str(i)

    if not os.path.exists(folder_path):
        os.mkdir( "lll-MRs-" + str(i))

    aux = tc100_pr[tc100_pr['total_one'] == i]

    # Iterate through rows in the filtered dataframe
    for index, row in aux.iterrows():
        # Get test ID
        test_id = row['test #']
        
        # Safely filter df_data for current test ID
        df_data_aux = df_data[df_data['test #'] == test_id]
        time = df_data_aux.at[index,'time']

        # Ensure the data is converted to float correctly
        try:
            time = df_data_aux.at[index,'time']
            pressureT = df_data_aux.at[index,'pressureT']
            speedT = df_data_aux.at[index,'speedT']
            pressureR = df_data_aux.at[index,'pressure']
            speedR = df_data_aux.at[index,'speed']
            rising_time = df_data_aux.at[index, 'RT']
            
            # Create plot
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.suptitle(f"Response Curves for Test ID: {test_id}, Total MRs: {i}", fontsize=14)
            
            # Pressure Plot
            axs[0].plot(time, pressureT, 'r-', label="Speed Target")
            axs[0].plot(time, pressureR, 'b-', label=f"Speed {test_id}")
            axs[0].axvline(rising_time, color='purple', linestyle='--', linewidth=1, alpha=0.6, label="Rising Time")
            axs[0].set_ylabel("Pressure")
            axs[0].legend()
            axs[0].grid(True)
            
            # Speed Plot
            axs[1].plot(time, speedT, 'r-', label="Speed Target")
            axs[1].plot(time, speedR, 'b-', label=f"Speed {test_id}")
            axs[1].axvline(rising_time, color='purple', linestyle='--', linewidth=1, alpha=0.6, label="Rising Time")
            axs[1].set_ylabel("Speed")
            axs[1].set_xlabel("Time")
            axs[1].legend()
            axs[1].grid(True)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(folder_path, f'{test_id}_MR{i}.png'), dpi=300)
            plt.close(fig)
            # plt.show()
        
        except Exception as e:
            print(f"Error processing test ID {test_id}: {e}")
            # Optional: continue to next iteration if there's an error
            continue

tc100.to_csv('tc1002.csv')
# print(final_data)