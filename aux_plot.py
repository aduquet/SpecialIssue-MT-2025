from get_metrics import getting_metrics
from mr_kp_ki_kd import mrs
import pandas as pd
import numpy as np
import pathlib
import json
import os
import matplotlib.pyplot as plt


def do_plot(tc_list, tc_pr, df_data, parameter_code ):
        
    for i in tc_list:

        folder_path = str(pathlib.Path().absolute()) + "\\" + parameter_code+"-MRs-" + str(i)

        if not os.path.exists(folder_path):
            os.mkdir( parameter_code+"-MRs-" + str(i))

        aux = tc_pr[tc_pr['total_one'] == i]

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
                plt.savefig(os.path.join(folder_path, f'{test_id}_MR{i}.pdf'), dpi=300)
                plt.close(fig)
                # plt.show()
            
            except Exception as e:
                print(f"Error processing test ID {test_id}: {e}")
                # Optional: continue to next iteration if there's an error
                continue