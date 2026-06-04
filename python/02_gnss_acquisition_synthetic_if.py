"""
02_gnss_acquisition_synthetic_if.py

Synthetic GNSS raw IF/IQ acquisition demo.

This script simulates a GPS L1 C/A-like baseband signal with a known code delay
and Doppler shift, adds noise, and then performs a delay-Doppler acquisition
search to recover the strongest correlation peak.

This demonstrates the core signal-processing concept behind GNSS acquisition
and delay-Doppler processing:

    received signal + local PRN replica + Doppler wipe-off + correlation
    = delay-Doppler acquisition map

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
    Resample the 1023-chip GPS C/A code to the sampling frequency.

    Parameters
    ----------
    ca_code : np.ndarray
        1023-chip C/A code with values -1 and +1.
    fs : float
        Sampling frequency in Hz.
    code_rate : float
        GPS C/A code chip rate in chips/second.
    n_samples : int
        Number of output samples.

    Returns
    -------
    sampled_code : np.ndarray
        Sampled C/A code at the requested sampling frequency.
    """

    sample_index = np.arange(n_samples)
    chip_index = np.floor(sample_index * code_rate / fs).astype(int) % 1023
    sampled_code = ca_code[chip_index]

    return sampled_code


def simulate_gnss_signal(
    prn: int,
    fs: float,
    coherent_ms: int,
    true_code_delay_samples: int,
    true_doppler_hz: float,
    cnr_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a complex baseband GNSS-like signal.

    Parameters
    ----------
    prn : int
        GPS PRN number.
    fs : float
        Sampling frequency in Hz.
    coherent_ms : int
        Coherent integration time in milliseconds.
    true_code_delay_samples : int
        Applied circular code delay in samples.
    true_doppler_hz : float
        Applied Doppler shift in Hz.
    cnr_scale : float
        Signal amplitude scale relative to noise.

    Returns
    -------
    rx_signal : np.ndarray
        Simulated complex received signal.
    local_code : np.ndarray
        Undelayed local sampled C/A code.
    """

    code_rate = 1.023e6
    n_samples = int(fs * coherent_ms * 1e-3)

    ca_code = generate_ca_code(prn)
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples)

    delayed_code = np.roll(local_code, true_code_delay_samples)

    t = np.arange(n_samples) / fs
    carrier = np.exp(1j * 2 * np.pi * true_doppler_hz * t)

    clean_signal = cnr_scale * delayed_code * carrier

    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    rx_signal = clean_signal + noise

    return rx_signal, local_code


def acquire_signal(
    rx_signal: np.ndarray,
    local_code: np.ndarray,
    fs: float,
    doppler_bins: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """
    Perform delay-Doppler acquisition search.

    For each Doppler bin, the received signal is mixed by the negative Doppler
    hypothesis and circularly correlated with the local PRN code using FFTs.

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

    Returns
    -------
    acquisition_map : np.ndarray
        2D correlation power map with shape [doppler, delay].
    estimated_delay_samples : int
        Estimated code delay in samples.
    estimated_doppler_hz : float
        Estimated Doppler frequency in Hz.
    peak_value : float
        Maximum correlation power.
    """

    n_samples = len(rx_signal)
    t = np.arange(n_samples) / fs

    code_fft = np.fft.fft(local_code)
    acquisition_map = np.zeros((len(doppler_bins), n_samples))

    for i, doppler in enumerate(doppler_bins):
        wipeoff = np.exp(-1j * 2 * np.pi * doppler * t)
        mixed_signal = rx_signal * wipeoff

        signal_fft = np.fft.fft(mixed_signal)

        # Circular correlation using FFT.
        correlation = np.fft.ifft(signal_fft * np.conj(code_fft))

        acquisition_map[i, :] = np.abs(correlation) ** 2

    peak_index = np.unravel_index(np.argmax(acquisition_map), acquisition_map.shape)

    doppler_index = peak_index[0]
    delay_index = peak_index[1]

    estimated_doppler_hz = doppler_bins[doppler_index]
    estimated_delay_samples = delay_index
    peak_value = acquisition_map[doppler_index, delay_index]

    return acquisition_map, estimated_delay_samples, estimated_doppler_hz, peak_value


def main() -> None:
    """Run the synthetic GNSS acquisition demo."""

    np.random.seed(7)

    # Simulation settings
    prn = 1
    fs = 4.092e6
    coherent_ms = 1

    # True signal parameters to recover
    true_code_delay_samples = 850
    true_doppler_hz = 1500.0

    # Increase this value if the peak is not visible enough.
    cnr_scale = 1.5

    # Create synthetic received signal
    rx_signal, local_code = simulate_gnss_signal(
        prn=prn,
        fs=fs,
        coherent_ms=coherent_ms,
        true_code_delay_samples=true_code_delay_samples,
        true_doppler_hz=true_doppler_hz,
        cnr_scale=cnr_scale,
    )

    # Doppler search settings
    doppler_bins = np.arange(-5000, 5001, 250)

    acquisition_map, estimated_delay, estimated_doppler, peak_value = acquire_signal(
        rx_signal=rx_signal,
        local_code=local_code,
        fs=fs,
        doppler_bins=doppler_bins,
    )

    print("Synthetic GNSS Acquisition Demo")
    print("--------------------------------")
    print(f"PRN: {prn}")
    print(f"Sampling frequency: {fs / 1e6:.3f} MHz")
    print(f"Coherent integration: {coherent_ms} ms")
    print("")
    print(f"True code delay:      {true_code_delay_samples} samples")
    print(f"Estimated code delay: {estimated_delay} samples")
    print("")
    print(f"True Doppler:         {true_doppler_hz:.1f} Hz")
    print(f"Estimated Doppler:    {estimated_doppler:.1f} Hz")
    print("")
    print(f"Peak correlation power: {peak_value:.2e}")

    # Plot acquisition map
    plt.figure(figsize=(10, 6))
    plt.imshow(
        10 * np.log10(acquisition_map + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[
            0,
            acquisition_map.shape[1] - 1,
            doppler_bins[0],
            doppler_bins[-1],
        ],
    )
    plt.colorbar(label="Correlation power (dB)")
    plt.scatter(
        estimated_delay,
        estimated_doppler,
        marker="x",
        s=100,
        linewidths=2,
        label="Detected peak",
    )
    plt.xlabel("Code delay (samples)")
    plt.ylabel("Doppler frequency (Hz)")
    plt.title("Synthetic GNSS Acquisition Delay-Doppler Map")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
