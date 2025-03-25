import numpy as np # type: ignore
from scipy.signal import find_peaks, peak_widths # type: ignore
from scipy.optimize import curve_fit # type: ignore

# from ranking_metrics.constants import *
# Fit an exponential decay to the oscillation peaks

# Define CONSTANTS
# when pressure reaches 90% of the target value, the duration is called 'rise time'
RISETIME_REACHED_RATIO = 0.9

# pressure must not exceed the target value more than 5% (it was call MAX_PEAK_RATIO)
OVERSHOOT_RATIO = 1.05

# after the peak is reached the pressure must not fall down below 97% of the target value
SETTLING_MINIMUM_RATIO = 0.97

# system is intended to reach stable mode if pressure does not differ more than 1.5 bar to target value
ALLOWED_PRESSURE_DEVIATION = 1.5

# the time until the pressure stays within  must not exceed 0.3 seconds
MAX_TRANSIENT_TIME = 0.3

# the time until the pressure has settled must not exceed 0.3 seconds
MAX_SETTLING_TIME = 999  # currently not assessed

# when system is settled the pressure must not exceed 0.2 bar the target value in average
ALLOWED_AVERAGE_SETTLING_DEVIATION = 0.2


def exponential_decay(t, A, decay_rate, offset):
    return A * np.exp(-decay_rate * t) + offset


# Define the model for logarithmic fitting
def log_decay(t, A, decay_rate):
    return A - decay_rate * t


# Define the logistic function (sigmoid curve)
def logistic_function(t, L, k, t0, offset):
    return L / (1 + np.exp(-k * (t - t0))) + offset


def findPeaks(signal):
    return find_peaks(signal)


class SignalMetrics:

    def __init__(self, signal, time, max_peak_time):
        self.signal = signal
        self.time = time
        self.max_peak_time = max_peak_time
        self._before_peak_mask = self.time < self.max_peak_time
        self._signal_before_peak = self.signal[self._before_peak_mask]
        self._time_before_peak = self.time[self._before_peak_mask]

    def oscillation_frequency(self):
        oscillation_peaks, _ = findPeaks(self._signal_before_peak)
        oscillation_valleys, _ = findPeaks(-self._signal_before_peak)
        num_oscillations = len(oscillation_peaks) + len(oscillation_valleys)
        return num_oscillations

    def average_oscillation_amplitude(self):
        oscillation_peaks, _ = find_peaks(self._signal_before_peak)
        oscillation_valleys, _ = find_peaks(-self._signal_before_peak)
        peak_values = self._signal_before_peak[oscillation_peaks]
        valley_values = self._signal_before_peak[oscillation_valleys]
        min_length = min(len(peak_values), len(valley_values))
        peak_values = peak_values[:min_length]
        valley_values = valley_values[:min_length]
        avg_amplitude = np.mean(np.abs(peak_values - valley_values))
        return round(float(avg_amplitude), 3)

    def growing_factor(self):
        try:
            # Provide a reasonable initial guess for [L, k, t0, offset]
            initial_guess = [
                max(self._signal_before_peak),
                1,
                np.median(self._time_before_peak),
                0,
            ]

            # Bounds for the parameters: L, k, t0, offset
            bounds = (
                [0, 0, 0, -np.inf],  # Lower bounds
                [np.inf, np.inf, max(self._time_before_peak), np.inf],  # Upper bounds
            )
            popt, _ = curve_fit(
                logistic_function,
                self._time_before_peak,
                self._signal_before_peak,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )  # Increased maxfev

            # Unpack the fitted parameters
            L, k, t0, offset = popt
            # Generate fitted values
            fitted_values = logistic_function(self._time_before_peak, *popt)
            # Calculate residuals
            residuals = self._signal_before_peak - fitted_values
            # Total sum of squares (SS_tot)
            ss_tot = np.sum(
                (self._signal_before_peak - np.mean(self._signal_before_peak)) ** 2
            )
            # Residual sum of squares (SS_res)
            ss_res = np.sum(residuals**2)
            # R-squared (R²) value
            r_squared = 1 - (ss_res / ss_tot)
            # Convert the parameters to regular floats and round them
            return (
                round(float(k), 3),
                round(float(r_squared), 3),
                np.round(fitted_values, 3).tolist(),
                self._time_before_peak,
            )

        except ValueError as e:
            print(f"Error during curve fitting: {e}")
            return None, None, None
        except RuntimeError as e:
            print(f"Curve fitting did not converge: {e}")
            return None, None, None
        except Exception as e:
            print(f"Unexpected error during curve fitting: {e}")
            return None, None, None

    def compute_amplitude_score(self, signal_derivative):
        amplitude = np.max(np.abs(signal_derivative))
        return amplitude

    def compute_frequency_score(self, perturbations):
        frequency = np.sum(perturbations > 0)
        return frequency

    def compute_duration_score(self, perturbations, time):
        perturbation_duration = np.sum(perturbations > 0) * np.mean(np.diff(time))
        return perturbation_duration

    def total_energy(self, signal_derivative):
        energy = np.sum(signal_derivative**2)
        return round(float(energy), 3)

    def energy_before_peak(self, signal_derivative_before_peak):
        dev_signal_before_peak = signal_derivative_before_peak[self._before_peak_mask]
        energy = np.sum(dev_signal_before_peak**2)
        return round(float(energy), 3)

    def energy_after_peak(self, signal_derivative_after_peak):
        after_peak_mask = self.time > self.max_peak_time
        dev_signal_after_peak = signal_derivative_after_peak[after_peak_mask]
        energy = np.sum(dev_signal_after_peak**2)
        return round(float(energy), 3)

    def growing_rate(self):
        """
        Calculate the growing rate based on the start and peak of the signal.
        Growth rate is calculated as the change in signal divided by the change in time.
        """
        try:
            # Initial signal and time values (start point)
            initial_signal = self._signal_before_peak[0]  # First value before the peak
            initial_time = self._time_before_peak[0]  # First time before the peak

            # Peak signal and corresponding time (peak point)
            peak_signal = np.max(
                self._signal_before_peak
            )  # Peak value of the signal before the peak
            peak_time = self._time_before_peak[
                np.argmax(self._signal_before_peak)
            ]  # Time at which the peak occurs

            # Compute the growing rate
            growing_rate = (peak_signal - initial_signal) / (peak_time - initial_time)

            return round(float(growing_rate), 3)

        except Exception as e:
            print(f"Error during growing rate calculation: {e}")
            return None

    def growing_rate_speed(self, rise_index):
        """
        Calculate the growing rate based on the start and peak of the signal.
        Growth rate is calculated as the change in signal divided by the change in time.
        """
        signal_at_start = self.signal[0]
        signal_at_rise = self.signal[rise_index]
        time_at_start = self.time[0]
        time_at_rise = self.time[rise_index]

        # Compute the growing rate as change in signal over change in time
        growing_rate = (abs(signal_at_rise) - abs(signal_at_start)) / (
            time_at_rise - time_at_start
        )

        # Return the growing rate
        return growing_rate

    def get_rising_time(self, target_signal):

        rise_time = None
        rise_index = -1
        overshoot_detected = False

        for i in range(len(self.signal)):
            # Check for rise time (when pressure reaches 90% of the target)
            if (
                rise_time is None
                and self.signal[i] >= RISETIME_REACHED_RATIO * target_signal[i]
            ):
                rise_time = self.time[i]
                rise_index = i

            # Check for overshoot (when pressure exceeds 105% of the target)
            if self.signal[i] > OVERSHOOT_RATIO * target_signal[i]:
                overshoot_detected = True

        return rise_time, rise_index, overshoot_detected

    def get_gr_before_rising_time(self, rise_index):
        signal_at_start = self.signal[0]
        signal_at_rise = self.signal[rise_index]
        time_at_start = self.time[0]
        time_at_rise = self.time[rise_index]

        # Compute the growing rate as change in signal over change in time
        growing_rate = (signal_at_rise - signal_at_start) / (
            time_at_rise - time_at_start
        )

        # Return the growing rate
        return growing_rate
