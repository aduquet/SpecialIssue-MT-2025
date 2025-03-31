from scipy.signal import find_peaks
from signal_metrics import SignalMetrics
from pathlib import Path
import numpy as np
import pandas as pd

def getting_metrics(df):
    df_aux = []

    for index, row in df.iterrows():

        time = np.array(row["time"])
        pressureR = np.array(row["pressureR"])
        pressureT = np.array(row["pressureT"])
        speedR = np.array(row["speedR"])
        speedT = np.array(row["speedT"])

        pressure_peaks, _ = find_peaks(pressureR, height=0)
        pressure_amplitudes = pressureR[pressure_peaks]

        pressure_max_peak_idx_T = np.argmax(pressureT)
        pressure_target_value = pressureT[pressure_max_peak_idx_T]

        # Find the max peak pressure
        pressure_max_peak_idx = np.argmax(pressure_amplitudes)
        pressure_max_peak_time = time[pressure_peaks[pressure_max_peak_idx]]
        pressure_max_peak_value = pressure_amplitudes[pressure_max_peak_idx]

        currentPressure_ratio = round(
            float(pressure_max_peak_value / pressure_target_value), 3
        )
        pressure_metrics = SignalMetrics(
            signal=pressureR, time=time, max_peak_time=pressure_max_peak_time
        )
        
        osc = pressure_metrics.oscillation_frequency()
        

        # Calculate the first derivative (rate of change) to detect perturbations
        pressure_derivative = np.gradient(pressureR, time)

        pressure_energy = pressure_metrics.total_energy(pressure_derivative)

        """SPEED"""

        # Find peaks in the speed data
        
        speed_metrics = SignalMetrics(
            signal=speedR, time=time, max_peak_time=pressure_max_peak_time
        )
        osc_s = speed_metrics.oscillation_frequency()

        # Calculate the first derivative (rate of change) to detect perturbations
        speed_derivative = np.gradient(speedR, time)
        # Define threshold for detecting significant perturbations
        speed_energy = speed_metrics.total_energy(speed_derivative)

        rising_time, rise_index, overshoot_detected = pressure_metrics.get_rising_time(
            pressureT
        )

        # Error signal
        error   = pressureT - pressureR
        r_ess_step = abs(pressureT[-1] - pressureR[-1])
        r_IAE = np.trapezoid(np.abs(error), time)
        r_ITAE = np.trapezoid(time * np.abs(error), time)

        try:
            # Compute natural frequency (approximation)
            omega_n = np.sqrt(row['Kr'] / (row['Tn'] * row['Tv']))

            # Compute steady-state error for different inputs
            ess_step = 1 / (1 + row['Kr'])  # Step input

            # Compute IAE and ITAE
            IAE = 1.5 / omega_n
            ITAE = 1.2 / (omega_n ** 2)

        except:
           
            ess_step = 'div0'
            IAE = IAE
            IAE = ITAE


        df1 = {
            "test_id": row["test_id"],
            "tc": row["tc"],
            "test #": row["id"],
            "kp": row["kp"],
            "ki": row["ki"],
            "kd": row["kd"],
            'ess_step': ess_step,
            'IAE': IAE,
            'ITAE': ITAE,
            'R_ess_step': r_ess_step,
            'R_IAE': r_IAE,
            'R_ITAE': r_ITAE,
            'ess_step': ess_step,
            "Kr": row["Kr"],
            "Tn": row["Tn"],
            "Tv": row["Tv"],
            "PR": currentPressure_ratio,
            "RT": rising_time,
            "P-Eng": pressure_energy,
            "S-Eng": speed_energy,
            "Eng-rel": pressure_energy/speed_energy,
            "osc_p": osc,
            "osc_s": osc_s,
            "speed": speedR,
            "time": time,
            "pressure": pressureR,
            "pressureT": pressureT,
            "speedT": speedT,
            "pressure_max_peak_idx": pressure_max_peak_idx,
        }

        df_aux.append(df1)

    df = pd.DataFrame(df_aux)
    return df
