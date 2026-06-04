"""
03_delay_doppler_map_synthetic_gnssr.py

Synthetic GNSS-R Delay-Doppler Map demo.

This script simulates a simplified GNSS-R reflected signal and forms a
Delay-Doppler Map (DDM) by correlating the received signal with a local
GPS C/A PRN replica over delay and Doppler hypotheses.

The purpose is to demonstrate the signal-processing principle behind
GNSS-R DDM generation:

    reflected GNSS-like signal
    + local PRN replica
    + Doppler search
    + delay search
    = Delay-Doppler Map

Author: Yusof Ghiasi
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_ca_code(prn: int) -> np.ndarray:
    """
    Generate GPS L1 C/A code for selected PRN.

    Parameters
    ----------
    prn : int
        GPS satellite PRN number. This demo supports PRNs 1 to 32.

    Returns
    -------
    ca_code : np.ndarray
        GPS C/A code sequence with values -1 and +1.
        Length is 1023 chips.
    """

    g2_taps = {
        1: (2, 6),
        2: (3, 7),
        3: (4, 8),
        4: (5, 9),
        5: (1, 9),
        6: (2, 10),
        7: (1, 8),
        8: (2, 9),
        9: (3, 10),
        10: (2, 3),
        11: (3, 4),
        12: (5, 6),
        13: (6, 7),
        14: (7, 8),
        15: (8, 9),
        16: (9, 10),
        17: (1, 4),
        18: (2, 5),
        19: (3, 6),
        20: (4, 7),
        21: (5, 8),
        22: (6, 9),
        23: (1, 3),
        24: (4, 6),
        25: (5, 7),
        26: (6, 8),
        27: (7, 9),
        28: (8, 10),
        29: (1, 6),
        30: (2, 7),
        31: (3, 8),
        32: (4, 9),
    }

    if prn not in g2_taps:
        raise ValueError("This demo supports GPS PRNs 1 to 32.")

    tap1, tap2 = g2_taps[prn]

    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    ca_code = np.zeros(1023, dtype=int)

    for i in range(1023):
        g1_output = g1[-1]
        g2_output = g2[tap1 - 1] ^ g2[tap2 - 1]

        ca_code[i] = g1_output ^ g2_output

        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        g1[1:] = g1[:-1]
        g1[0] = g1_feedback

        g2[1:] = g2[:-1]
        g2[0] = g2_feedback

    ca_code = 1 - 2 * ca_code

    return ca_code


def resample_ca_code(ca_code: np.ndarray, fs: float, code_rate: float, n_samples: int) -> np.ndarray:
    """
    Resample GPS C/A code to a selected sampling frequency.
    """

    sample_index = np.arange(n_samples)
    chip_index = np.floor(sample_index * code_rate / fs).astype(int) % 1023
    sampled_code = ca_code[chip_index]

    return sampled_code


def simulate_reflected_signal(
    prn: int,
    fs: float,
    coherent_ms: int,
    reflected_delay_samples: int,
    reflected_doppler_hz: float,
    reflected_amplitude: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a simplified GNSS-R reflected signal.

    Parameters
    ----------
    prn : int
        GPS PRN number.
    fs : float
        Sampling frequency in Hz.
    coherent_ms : int
        Coherent integration time in milliseconds.
    reflected_delay_samples : int
        Delay of reflected signal in samples.
    reflected_doppler_hz : float
        Doppler shift of reflected signal in Hz.
    reflected_amplitude : float
        Amplitude of reflected signal.
    noise_std : float
        Standard deviation of complex noise.

    Returns
    -------
    rx_signal : np.ndarray
        Simulated received reflected signal.
    local_code : np.ndarray
        Local sampled C/A code.
    """

    code_rate = 1.023e6
    n_samples = int(fs * coherent_ms * 1e-3)

    ca_code = generate_ca_code(prn)
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples)

    reflected_code = np.roll(local_code, reflected_delay_samples)

    t = np.arange(n_samples) / fs
    reflected_carrier = np.exp(1j * 2 * np.pi * reflected_doppler_hz * t)

    clean_reflection = reflected_amplitude * reflected_code * reflected_carrier

    noise = noise_std * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)

    rx_signal = clean_reflection + noise

    return rx_signal, local_code


def form_ddm(
    rx_signal: np.ndarray,
    local_code: np.ndarray,
    fs: float,
    doppler_bins: np.ndarray,
    delay_bins: np.ndarray,
) -> np.ndarray:
    """
    Form a simplified Delay-Doppler Map.

    Parameters
    ----------
    rx_signal : np.ndarray
        Complex received signal.
    local_code : np.ndarray
        Local sampled PRN code.
    fs : float
        Sampling frequency in Hz.
    doppler_bins : np.ndarray
        Doppler search bins in Hz.
    delay_bins : np.ndarray
        Delay search bins in samples.

    Returns
    -------
    ddm : np.ndarray
        Delay-Doppler Map with shape [doppler, delay].
    """

    n_samples = len(rx_signal)
    t = np.arange(n_samples) / fs

    ddm = np.zeros((len(doppler_bins), len(delay_bins)))

    for i, doppler in enumerate(doppler_bins):
        wipeoff = np.exp(-1j * 2 * np.pi * doppler * t)
        mixed_signal = rx_signal * wipeoff

        for j, delay in enumerate(delay_bins):
            shifted_code = np.roll(local_code, delay)
            correlation = np.sum(mixed_signal * np.conj(shifted_code))
            ddm[i, j] = np.abs(correlation) ** 2

    return ddm


def detect_ddm_peak(
    ddm: np.ndarray,
    doppler_bins: np.ndarray,
    delay_bins: np.ndarray,
) -> tuple[int, float, float]:
    """
    Detect the strongest Delay-Doppler peak.
    """

    peak_index = np.unravel_index(np.argmax(ddm), ddm.shape)

    doppler_index = peak_index[0]
    delay_index = peak_index[1]

    estimated_delay = delay_bins[delay_index]
    estimated_doppler = doppler_bins[doppler_index]
    peak_power = ddm[doppler_index, delay_index]

    return estimated_delay, estimated_doppler, peak_power


def main() -> None:
    """Run the synthetic GNSS-R DDM demo."""

    np.random.seed(11)

    prn = 1
    fs = 4.092e6
    coherent_ms = 1

    # True reflected-signal parameters
    true_reflected_delay_samples = 950
    true_reflected_doppler_hz = -1750.0

    reflected_amplitude = 1.2
    noise_std = 1.0

    rx_signal, local_code = simulate_reflected_signal(
        prn=prn,
        fs=fs,
        coherent_ms=coherent_ms,
        reflected_delay_samples=true_reflected_delay_samples,
        reflected_doppler_hz=true_reflected_doppler_hz,
        reflected_amplitude=reflected_amplitude,
        noise_std=noise_std,
    )

    # Search grid
    doppler_bins = np.arange(-5000, 5001, 250)
    delay_bins = np.arange(600, 1301, 5)

    ddm = form_ddm(
        rx_signal=rx_signal,
        local_code=local_code,
        fs=fs,
        doppler_bins=doppler_bins,
        delay_bins=delay_bins,
    )

    estimated_delay, estimated_doppler, peak_power = detect_ddm_peak(
        ddm=ddm,
        doppler_bins=doppler_bins,
        delay_bins=delay_bins,
    )

    print("Synthetic GNSS-R Delay-Doppler Map Demo")
    print("---------------------------------------")
    print(f"PRN: {prn}")
    print(f"Sampling frequency: {fs / 1e6:.3f} MHz")
    print(f"Coherent integration: {coherent_ms} ms")
    print("")
    print(f"True reflected delay:      {true_reflected_delay_samples} samples")
    print(f"Estimated reflected delay: {estimated_delay} samples")
    print("")
    print(f"True reflected Doppler:      {true_reflected_doppler_hz:.1f} Hz")
    print(f"Estimated reflected Doppler: {estimated_doppler:.1f} Hz")
    print("")
    print(f"Peak DDM power: {peak_power:.2e}")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        10 * np.log10(ddm + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[
            delay_bins[0],
            delay_bins[-1],
            doppler_bins[0],
            doppler_bins[-1],
        ],
    )
    plt.colorbar(label="DDM power (dB)")
    plt.scatter(
        estimated_delay,
        estimated_doppler,
        marker="x",
        s=100,
        linewidths=2,
        label="Detected DDM peak",
    )
    plt.xlabel("Delay (samples)")
    plt.ylabel("Doppler frequency (Hz)")
    plt.title("Synthetic GNSS-R Delay-Doppler Map")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
