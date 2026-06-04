"""
04_tdoa_fdoa_two_receiver_demo.py

Two-receiver TDOA/FDOA estimation demo.

This script simulates an unknown RF signal received by two receivers. The
second receiver observes the same signal with a time delay and a frequency
offset. A joint delay-frequency ambiguity search is then used to estimate
the relative time delay and frequency offset.

This is a simplified signal-processing demo relevant to RF geolocation:

    receiver 1 signal
    receiver 2 signal
    -> delay/frequency search
    -> TDOA estimate
    -> FDOA estimate

Author: Yusof Ghiasi
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_unknown_rf_signal(
    fs: float,
    duration_s: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """
    Generate a synthetic unknown complex RF-like baseband signal.

    The signal is generated as complex noise filtered in the frequency domain
    to create a band-limited waveform. This avoids assuming a known PRN code
    and is closer to a generic RF emitter scenario.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration_s : float
        Signal duration in seconds.
    bandwidth_hz : float
        Approximate signal bandwidth in Hz.

    Returns
    -------
    signal : np.ndarray
        Complex baseband RF-like signal.
    """

    n_samples = int(fs * duration_s)

    white_noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

    spectrum = np.fft.fftshift(np.fft.fft(white_noise))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / fs))

    bandpass_mask = np.abs(freqs) <= bandwidth_hz / 2
    filtered_spectrum = spectrum * bandpass_mask

    signal = np.fft.ifft(np.fft.ifftshift(filtered_spectrum))

    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    return signal


def apply_delay_and_fdoa(
    signal: np.ndarray,
    fs: float,
    delay_samples: int,
    fdoa_hz: float,
) -> np.ndarray:
    """
    Apply integer-sample time delay and frequency offset to a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input complex signal.
    fs : float
        Sampling frequency in Hz.
    delay_samples : int
        Time delay in samples.
    fdoa_hz : float
        Frequency offset in Hz.

    Returns
    -------
    shifted_signal : np.ndarray
        Delayed and frequency-shifted signal.
    """

    n_samples = len(signal)

    delayed_signal = np.roll(signal, delay_samples)

    t = np.arange(n_samples) / fs
    frequency_shift = np.exp(1j * 2 * np.pi * fdoa_hz * t)

    shifted_signal = delayed_signal * frequency_shift

    return shifted_signal


def add_complex_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex Gaussian noise to a signal at a selected SNR.

    Parameters
    ----------
    signal : np.ndarray
        Clean complex signal.
    snr_db : float
        Signal-to-noise ratio in dB.

    Returns
    -------
    noisy_signal : np.ndarray
        Noisy complex signal.
    """

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )

    noisy_signal = signal + noise

    return noisy_signal


def estimate_tdoa_fdoa(
    rx1: np.ndarray,
    rx2: np.ndarray,
    fs: float,
    delay_bins: np.ndarray,
    fdoa_bins: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """
    Estimate TDOA and FDOA using a joint delay-frequency search.

    For each FDOA hypothesis, receiver 2 is frequency-corrected. For each
    delay hypothesis, the corrected receiver 2 signal is shifted and compared
    with receiver 1 using a coherent inner product.

    Parameters
    ----------
    rx1 : np.ndarray
        Complex signal at receiver 1.
    rx2 : np.ndarray
        Complex signal at receiver 2.
    fs : float
        Sampling frequency in Hz.
    delay_bins : np.ndarray
        Candidate delay values in samples.
    fdoa_bins : np.ndarray
        Candidate frequency offsets in Hz.

    Returns
    -------
    ambiguity_surface : np.ndarray
        Joint delay-frequency ambiguity surface.
    estimated_delay_samples : int
        Estimated relative delay in samples.
    estimated_fdoa_hz : float
        Estimated relative frequency offset in Hz.
    peak_value : float
        Maximum ambiguity value.
    """

    n_samples = len(rx1)
    t = np.arange(n_samples) / fs

    ambiguity_surface = np.zeros((len(fdoa_bins), len(delay_bins)))

    for i, fdoa in enumerate(fdoa_bins):

        # Correct receiver 2 by the negative candidate FDOA.
        fdoa_correction = np.exp(-1j * 2 * np.pi * fdoa * t)
        rx2_corrected = rx2 * fdoa_correction

        for j, delay in enumerate(delay_bins):

            # Undo candidate delay by shifting receiver 2 backward.
            rx2_aligned = np.roll(rx2_corrected, -delay)

            metric = np.vdot(rx1, rx2_aligned)

            ambiguity_surface[i, j] = np.abs(metric) ** 2

    peak_index = np.unravel_index(np.argmax(ambiguity_surface), ambiguity_surface.shape)

    fdoa_index = peak_index[0]
    delay_index = peak_index[1]

    estimated_fdoa_hz = fdoa_bins[fdoa_index]
    estimated_delay_samples = delay_bins[delay_index]
    peak_value = ambiguity_surface[fdoa_index, delay_index]

    return ambiguity_surface, estimated_delay_samples, estimated_fdoa_hz, peak_value


def main() -> None:
    """Run the two-receiver TDOA/FDOA estimation demo."""

    np.random.seed(21)

    # Signal settings
    fs = 2.0e6
    duration_s = 0.002
    bandwidth_hz = 300e3

    # True receiver-2 relative observables
    true_tdoa_samples = 37
    true_fdoa_hz = -2200.0

    snr_db = 5.0

    # Generate unknown RF-like source signal
    source_signal = generate_unknown_rf_signal(
        fs=fs,
        duration_s=duration_s,
        bandwidth_hz=bandwidth_hz,
    )

    # Receiver 1 receives the reference version
    rx1_clean = source_signal

    # Receiver 2 receives delayed and frequency-shifted version
    rx2_clean = apply_delay_and_fdoa(
        signal=source_signal,
        fs=fs,
        delay_samples=true_tdoa_samples,
        fdoa_hz=true_fdoa_hz,
    )

    rx1 = add_complex_noise(rx1_clean, snr_db=snr_db)
    rx2 = add_complex_noise(rx2_clean, snr_db=snr_db)

    # Search grid
    delay_bins = np.arange(-80, 81, 1)
    fdoa_bins = np.arange(-5000, 5001, 100)

    ambiguity_surface, estimated_delay, estimated_fdoa, peak_value = estimate_tdoa_fdoa(
        rx1=rx1,
        rx2=rx2,
        fs=fs,
        delay_bins=delay_bins,
        fdoa_bins=fdoa_bins,
    )

    print("Two-Receiver TDOA/FDOA Estimation Demo")
    print("--------------------------------------")
    print(f"Sampling frequency: {fs / 1e6:.3f} MHz")
    print(f"Signal duration: {duration_s * 1e3:.2f} ms")
    print(f"Signal bandwidth: {bandwidth_hz / 1e3:.1f} kHz")
    print(f"SNR: {snr_db:.1f} dB")
    print("")
    print(f"True TDOA:      {true_tdoa_samples} samples")
    print(f"Estimated TDOA: {estimated_delay} samples")
    print("")
    print(f"True FDOA:      {true_fdoa_hz:.1f} Hz")
    print(f"Estimated FDOA: {estimated_fdoa:.1f} Hz")
    print("")
    print(f"Peak ambiguity value: {peak_value:.2e}")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        10 * np.log10(ambiguity_surface + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[
            delay_bins[0],
            delay_bins[-1],
            fdoa_bins[0],
            fdoa_bins[-1],
        ],
    )
    plt.colorbar(label="Ambiguity power (dB)")
    plt.scatter(
        estimated_delay,
        estimated_fdoa,
        marker="x",
        s=100,
        linewidths=2,
        label="Detected TDOA/FDOA peak",
    )
    plt.xlabel("Relative delay / TDOA (samples)")
    plt.ylabel("Relative frequency offset / FDOA (Hz)")
    plt.title("Two-Receiver TDOA/FDOA Ambiguity Surface")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
