from get_metrics import getting_metrics
# from mr_kp_ki_kd import mrs
from final_mrs import mrs_pid
from aux_plot import do_plot
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

def compare(val, ref):
    if val > ref:
        return 'h'
    elif val < ref:
        return 'l'
    else:
        return 'e'
    
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
tcstr = 'tc106'

# df_data =  df_data[df_data['tc'] == tcstr]


final_data = df_data[[ 'tc', 'test #', 'kp', 'kd', 'ki', 
                      'ess_step', 'IAE', 'ITAE', 'R_ess_step', 'R_IAE', 'R_ITAE','Kr', 'Tn', 'Tv', 
                      'PR', 'RT', 'P-Eng','S-Eng', 'Eng-rel','osc_p', 'osc_s']]

ref_kp = final_data.loc[0, 'kp']
ref_ki = final_data.loc[0, 'ki']
ref_kd = final_data.loc[0, 'kd']

for index, row in final_data.iterrows():
    code = (
        compare(row['kp'], ref_kp) +
        compare(row['ki'], ref_ki) +
        compare(row['kd'], ref_kd)
    )
    final_data.at[index,'parameter_code'] = code

final_data.to_csv('final_data.csv')

final_data = pd.read_csv('final_data.csv')
tc100 =  final_data[final_data['tc'] == tcstr]


ref_rt = final_data.loc[0, 'RT']
# ref_eng = final_data.loc[0, 'Eng-rel']
ref_eng_s = final_data.loc[0,'S-Eng']
ref_osc = final_data.loc[0, 'osc_p']
ref_IAE = final_data.loc[0, 'R_IAE']
ref_ITAE = final_data.loc[0, 'R_ITAE']

ref_ess = final_data.loc[0, 'R_ess_step']
ref_eng_rel= final_data.loc[0,'P-Eng'] / final_data.loc[0,'S-Eng']

for index, row in tc100.iterrows():
    tc100.at[index,'eng-diff'] =ref_eng_s - row['S-Eng']

ref_eng_diff = 0

parameter_count_list = list(set(final_data['parameter_code']))

for index, row in tc100.iterrows():
    eng_rel= row['P-Eng'] / row['S-Eng']
    if row['parameter_code'] == 'lll':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_EngS'] = mrs_pid.mr_lll_two(ref_eng_s, row['S-Eng'])
        tc100.at[index,'mr_osc'] = mrs_pid.mr_osc(row['osc_p'], row['osc_s'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_l(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])

    if row['parameter_code'] == 'hlh':
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_g(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_g(ref_eng_rel, eng_rel)
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])

    if row['parameter_code'] == 'hll':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_g(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_g(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_g(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_g(ref_eng_rel, eng_rel)

    if row['parameter_code'] == 'lhl' or row['parameter_code'] == 'ehl' :
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_l(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_l(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_g(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_g(ref_eng_rel, eng_rel)

    if row['parameter_code'] == 'ell' or row['parameter_code'] == 'llh' or row['parameter_code'] == 'eel' or row['parameter_code'] == 'elh':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_g(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_g(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_g(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_l(ref_eng_rel, eng_rel)

    if row['parameter_code'] == 'hhl':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_l(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_g(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_g(ref_eng_rel, eng_rel)

    if row['parameter_code'] == 'hhh':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_l(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_l(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_l(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_l(ref_eng_rel, eng_rel)

    if row['parameter_code'] == 'ehh' or row['parameter_code'] == 'heh':
        tc100.at[index,'mr_RT'] = mrs_pid.mr_rt_f(ref_rt, row['RT'])
        tc100.at[index,'mr_PR'] = mrs_pid.mr_PR(row['PR'])
        tc100.at[index,'mr_ess_g'] = mrs_pid.mr_ess_step_l(ref_ess, row['R_ess_step'])
        tc100.at[index,'mr_IAE'] = mrs_pid.mr_IAE_l(ref_IAE, row['R_IAE'])
        tc100.at[index,'mr_ITAE'] = mrs_pid.mr_ITAE_l(ref_ITAE, row['R_ITAE'])
        tc100.at[index,'mr_Eng_p_s'] = mrs_pid.mr_eng_p_s_g(ref_eng_rel, eng_rel)


for index, row in tc100.iterrows():
    if row['parameter_code'] == 'lll':
        tc100.at[index,'total_one'] = row['mr_RT'] + row['mr_EngS']+row['mr_osc'] + row['mr_IAE'] +row['mr_PR'] 
    if row['parameter_code'] == 'hlh':
        tc100.at[index,'total_one'] = row['mr_ess_g'] + row['mr_PR']+row['mr_Eng_p_s'] + row['mr_RT']  
    if row['parameter_code'] == 'hll' or row['parameter_code'] == 'ell' or row['parameter_code'] == 'llh' or row['parameter_code'] == 'hhh'or row['parameter_code'] == 'eel' or row['parameter_code'] == 'elh':
        tc100.at[index,'total_one'] = row['mr_RT'] + row['mr_PR']+row['mr_ess_g'] + row['mr_IAE'] +row['mr_ITAE'] +row['mr_Eng_p_s']
    if row['parameter_code'] == 'hhl':
        tc100.at[index,'total_one'] = row['mr_ess_g'] + row['mr_PR']+row['mr_Eng_p_s'] + row['mr_RT']  + row['mr_ITAE'] 

    if row['parameter_code'] == 'lhl' or row['parameter_code'] == 'ehl' or row['parameter_code'] == 'heh' :
        tc100.at[index,'total_one'] = row['mr_ess_g'] + row['mr_PR']+row['mr_Eng_p_s'] + row['mr_RT']  + row['mr_ITAE'] + row['mr_IAE'] 


para_list = list(set(tc100['parameter_code']))
print(para_list)
for para in para_list:

    tc100_pr = tc100[tc100['parameter_code'] == para]

    tc100_pr = tc100_pr[tc100_pr['mr_PR'] == 1]

    tc100_list = list(set(tc100_pr['total_one']))

    do_plot(tc100_list, tc100_pr, df_data, 'prueba_106_' + para )


tc100.to_csv('prueba_' + tcstr + '.csv')
