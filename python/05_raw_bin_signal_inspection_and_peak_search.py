"""
05_raw_bin_signal_inspection_and_peak_search.py

Raw .bin IF/IQ signal inspection and GNSS-style correlation peak search.

This script demonstrates how to work with local raw binary IF/IQ files:

1. Read binary samples from a .bin file
2. Convert samples into complex IQ data
3. Plot time-domain I/Q samples
4. Plot signal magnitude
5. Estimate and plot spectrum
6. Detect strong spectral peaks
7. Optionally perform GPS C/A PRN delay-Doppler correlation search
8. Report strongest delay-Doppler peaks

Important:
---------
Large .bin files should NOT be uploaded to GitHub. Place them locally in the
data/ folder and update the parameters below.

Because raw binary formats depend on receiver settings, you must confirm:

- sampling frequency
- IF frequency, if any
- data type
- I/Q interleaving format
- whether the file contains real IF samples or complex IQ samples

Author: Yusof Ghiasi
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

# Put your local .bin file in the data folder.
# Example:
# BIN_FILE = "data/sample_if_data_1.bin"
BIN_FILE = "data/sample_if_data_1.bin"

# Sampling frequency in Hz.
# Update this based on your file metadata.
FS = 8.184e6

# Data type in the .bin file.
# Common options: "int8", "int16", "float32"
DATA_TYPE = "int8"

# File format options:
# "real_if"         : file contains real-valued IF samples
# "interleaved_iq" : file contains I,Q,I,Q,... samples
IQ_FORMAT = "interleaved_iq"

# Number of complex samples to read.
# Keep this moderate for quick testing.
MAX_COMPLEX_SAMPLES = 200_000

# Remove DC offset before analysis?
REMOVE_DC = True

# Normalize signal power?
NORMALIZE_POWER = True

# Spectral peak detection
NUMBER_OF_SPECTRAL_PEAKS = 10

# GNSS PRN search settings
RUN_GNSS_PRN_SEARCH = True
PRN = 1

# Coherent integration time in milliseconds.
# GPS C/A code repeats every 1 ms.
COHERENT_MS = 1

# Doppler search bins in Hz.
DOPPLER_BINS = np.arange(-5000, 5001, 250)

# Report this many strongest delay-Doppler peaks.
NUMBER_OF_DD_PEAKS = 10


# =============================================================================
# GPS C/A code generation
# =============================================================================

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
    """

    sample_index = np.arange(n_samples)
    chip_index = np.floor(sample_index * code_rate / fs).astype(int) % 1023
    sampled_code = ca_code[chip_index]

    return sampled_code


# =============================================================================
# Raw binary reading
# =============================================================================

def read_raw_bin_file(
    bin_file: str,
    data_type: str,
    iq_format: str,
    max_complex_samples: int,
) -> np.ndarray:
    """
    Read raw .bin file and return complex samples.

    Parameters
    ----------
    bin_file : str
        Path to binary file.
    data_type : str
        Data type in file. Example: "int8", "int16", "float32".
    iq_format : str
        "real_if" or "interleaved_iq".
    max_complex_samples : int
        Maximum number of complex samples to return.

    Returns
    -------
    samples : np.ndarray
        Complex samples.
    """

    path = Path(bin_file)

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find file:\n{path}\n\n"
            "Place your .bin file in the data/ folder or update BIN_FILE at the top of this script."
        )

    dtype = np.dtype(data_type)

    if iq_format == "interleaved_iq":
        # Need 2 raw values per complex sample.
        raw_count = max_complex_samples * 2
        raw = np.fromfile(path, dtype=dtype, count=raw_count)

        if len(raw) < 2:
            raise ValueError("File does not contain enough samples.")

        if len(raw) % 2 != 0:
            raw = raw[:-1]

        i_samples = raw[0::2].astype(np.float64)
        q_samples = raw[1::2].astype(np.float64)

        samples = i_samples + 1j * q_samples

    elif iq_format == "real_if":
        raw = np.fromfile(path, dtype=dtype, count=max_complex_samples)
        samples = raw.astype(np.float64).astype(np.complex128)

    else:
        raise ValueError("IQ_FORMAT must be either 'real_if' or 'interleaved_iq'.")

    return samples


def preprocess_samples(
    samples: np.ndarray,
    remove_dc: bool,
    normalize_power: bool,
) -> np.ndarray:
    """
    Remove DC offset and optionally normalize signal power.
    """

    x = samples.copy()

    if remove_dc:
        x = x - np.mean(x)

    if normalize_power:
        power = np.mean(np.abs(x) ** 2)
        if power > 0:
            x = x / np.sqrt(power)

    return x


# =============================================================================
# Signal inspection
# =============================================================================

def plot_time_domain(samples: np.ndarray, fs: float, number_of_samples: int = 2000) -> None:
    """
    Plot I, Q, and magnitude in time domain.
    """

    n = min(number_of_samples, len(samples))
    t_ms = np.arange(n) / fs * 1e3

    plt.figure(figsize=(10, 5))
    plt.plot(t_ms, np.real(samples[:n]), label="I / real")
    plt.plot(t_ms, np.imag(samples[:n]), label="Q / imag")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Raw IF/IQ Samples: I and Q")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_ms, np.abs(samples[:n]))
    plt.xlabel("Time (ms)")
    plt.ylabel("Magnitude")
    plt.title("Raw IF/IQ Sample Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_spectrum(samples: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simple FFT spectrum.
    """

    n = len(samples)

    # Window reduces spectral leakage.
    window = np.hanning(n)
    xw = samples * window

    spectrum = np.fft.fftshift(np.fft.fft(xw))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))

    psd_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    return freqs, psd_db


def plot_spectrum(freqs: np.ndarray, psd_db: np.ndarray) -> None:
    """
    Plot spectrum.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e6, psd_db)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Raw IF/IQ Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_spectral_peaks(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    number_of_peaks: int,
    guard_bins: int = 20,
) -> list[tuple[float, float]]:
    """
    Find strongest spectral peaks using simple non-maximum suppression.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in Hz.
    psd_db : np.ndarray
        Spectrum magnitude in dB.
    number_of_peaks : int
        Number of peaks to report.
    guard_bins : int
        Number of neighboring bins to suppress after selecting a peak.

    Returns
    -------
    peaks : list of tuple
        List of (frequency_hz, magnitude_db).
    """

    spectrum_copy = psd_db.copy()
    peaks = []

    for _ in range(number_of_peaks):
        idx = int(np.argmax(spectrum_copy))

        peak_freq = freqs[idx]
        peak_mag = psd_db[idx]

        peaks.append((peak_freq, peak_mag))

        start = max(0, idx - guard_bins)
        end = min(len(spectrum_copy), idx + guard_bins + 1)
        spectrum_copy[start:end] = -np.inf

    return peaks


# =============================================================================
# Delay-Doppler PRN search
# =============================================================================

def gnss_delay_doppler_search(
    samples: np.ndarray,
    fs: float,
    prn: int,
    coherent_ms: int,
    doppler_bins: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """
    Perform GPS C/A PRN delay-Doppler search on raw samples.

    This assumes the samples contain a GPS C/A-like signal and that the
    sampling frequency is set correctly.

    Parameters
    ----------
    samples : np.ndarray
        Complex raw samples.
    fs : float
        Sampling frequency in Hz.
    prn : int
        GPS PRN number.
    coherent_ms : int
        Coherent integration time in milliseconds.
    doppler_bins : np.ndarray
        Doppler bins in Hz.

    Returns
    -------
    acquisition_map : np.ndarray
        Correlation power map [doppler, delay].
    estimated_delay_samples : int
        Estimated code delay in samples.
    estimated_doppler_hz : float
        Estimated Doppler frequency in Hz.
    peak_power : float
        Maximum correlation power.
    """

    code_rate = 1.023e6
    n_samples = int(fs * coherent_ms * 1e-3)

    if len(samples) < n_samples:
        raise ValueError(
            f"Not enough samples for {coherent_ms} ms coherent integration. "
            f"Need {n_samples}, but file has {len(samples)}."
        )

    x = samples[:n_samples]

    ca_code = generate_ca_code(prn)
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples)

    t = np.arange(n_samples) / fs

    code_fft = np.fft.fft(local_code)
    acquisition_map = np.zeros((len(doppler_bins), n_samples))

    for i, doppler in enumerate(doppler_bins):
        wipeoff = np.exp(-1j * 2 * np.pi * doppler * t)
        mixed = x * wipeoff

        signal_fft = np.fft.fft(mixed)
        corr = np.fft.ifft(signal_fft * np.conj(code_fft))

        acquisition_map[i, :] = np.abs(corr) ** 2

    peak_idx = np.unravel_index(np.argmax(acquisition_map), acquisition_map.shape)

    doppler_idx = peak_idx[0]
    delay_idx = peak_idx[1]

    estimated_doppler_hz = doppler_bins[doppler_idx]
    estimated_delay_samples = delay_idx
    peak_power = acquisition_map[doppler_idx, delay_idx]

    return acquisition_map, estimated_delay_samples, estimated_doppler_hz, peak_power


def find_delay_doppler_peaks(
    acquisition_map: np.ndarray,
    doppler_bins: np.ndarray,
    number_of_peaks: int,
    delay_guard_bins: int = 20,
    doppler_guard_bins: int = 1,
) -> list[tuple[int, float, float]]:
    """
    Find strongest delay-Doppler peaks using simple non-maximum suppression.

    Returns
    -------
    peaks : list of tuple
        List of (delay_samples, doppler_hz, peak_power).
    """

    surface = acquisition_map.copy()
    peaks = []

    for _ in range(number_of_peaks):
        peak_idx = np.unravel_index(np.argmax(surface), surface.shape)

        doppler_idx = peak_idx[0]
        delay_idx = peak_idx[1]

        peak_power = acquisition_map[doppler_idx, delay_idx]
        peak_doppler = doppler_bins[doppler_idx]

        peaks.append((delay_idx, peak_doppler, peak_power))

        d0 = max(0, doppler_idx - doppler_guard_bins)
        d1 = min(surface.shape[0], doppler_idx + doppler_guard_bins + 1)

        c0 = max(0, delay_idx - delay_guard_bins)
        c1 = min(surface.shape[1], delay_idx + delay_guard_bins + 1)

        surface[d0:d1, c0:c1] = -np.inf

    return peaks


def plot_acquisition_map(
    acquisition_map: np.ndarray,
    doppler_bins: np.ndarray,
    estimated_delay: int,
    estimated_doppler: float,
) -> None:
    """
    Plot delay-Doppler acquisition map.
    """

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
        label="Strongest peak",
    )
    plt.xlabel("Code delay (samples)")
    plt.ylabel("Doppler frequency (Hz)")
    plt.title("Raw .bin GNSS PRN Delay-Doppler Search")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main script
# =============================================================================

def main() -> None:
    """
    Run raw .bin inspection and optional GNSS PRN peak search.
    """

    print("Raw .bin IF/IQ Signal Inspection and Peak Search")
    print("================================================")
    print(f"File: {BIN_FILE}")
    print(f"Sampling frequency: {FS / 1e6:.3f} MHz")
    print(f"Data type: {DATA_TYPE}")
    print(f"IQ format: {IQ_FORMAT}")
    print("")

    samples = read_raw_bin_file(
        bin_file=BIN_FILE,
        data_type=DATA_TYPE,
        iq_format=IQ_FORMAT,
        max_complex_samples=MAX_COMPLEX_SAMPLES,
    )

    samples = preprocess_samples(
        samples=samples,
        remove_dc=REMOVE_DC,
        normalize_power=NORMALIZE_POWER,
    )

    print(f"Loaded complex samples: {len(samples):,}")
    print(f"Mean power after preprocessing: {np.mean(np.abs(samples) ** 2):.3f}")
    print("")

    # Time-domain inspection
    plot_time_domain(samples, fs=FS)

    # Spectrum inspection
    freqs, psd_db = compute_spectrum(samples, fs=FS)
    plot_spectrum(freqs, psd_db)

    spectral_peaks = find_spectral_peaks(
        freqs=freqs,
        psd_db=psd_db,
        number_of_peaks=NUMBER_OF_SPECTRAL_PEAKS,
    )

    print("Strongest spectral peaks")
    print("------------------------")
    for idx, (freq_hz, mag_db) in enumerate(spectral_peaks, start=1):
        print(f"{idx:02d}: frequency = {freq_hz / 1e6:+.6f} MHz, magnitude = {mag_db:.2f} dB")

    print("")

    if RUN_GNSS_PRN_SEARCH:
        print("Running GNSS PRN delay-Doppler search")
        print("-------------------------------------")
        print(f"PRN: {PRN}")
        print(f"Coherent integration: {COHERENT_MS} ms")
        print(f"Doppler range: {DOPPLER_BINS[0]} to {DOPPLER_BINS[-1]} Hz")
        print(f"Doppler step: {DOPPLER_BINS[1] - DOPPLER_BINS[0]} Hz")
        print("")

        acquisition_map, estimated_delay, estimated_doppler, peak_power = gnss_delay_doppler_search(
            samples=samples,
            fs=FS,
            prn=PRN,
            coherent_ms=COHERENT_MS,
            doppler_bins=DOPPLER_BINS,
        )

        print("Strongest delay-Doppler result")
        print("------------------------------")
        print(f"Estimated code delay: {estimated_delay} samples")
        print(f"Estimated Doppler:    {estimated_doppler:.1f} Hz")
        print(f"Peak power:           {peak_power:.2e}")
        print("")

        dd_peaks = find_delay_doppler_peaks(
            acquisition_map=acquisition_map,
            doppler_bins=DOPPLER_BINS,
            number_of_peaks=NUMBER_OF_DD_PEAKS,
        )

        print("Top delay-Doppler peaks")
        print("-----------------------")
        for idx, (delay, doppler, power) in enumerate(dd_peaks, start=1):
            print(
                f"{idx:02d}: delay = {delay:5d} samples, "
                f"doppler = {doppler:+8.1f} Hz, "
                f"power = {power:.2e}"
            )

        plot_acquisition_map(
            acquisition_map=acquisition_map,
            doppler_bins=DOPPLER_BINS,
            estimated_delay=estimated_delay,
            estimated_doppler=estimated_doppler,
        )

    else:
        print("GNSS PRN delay-Doppler search skipped.")
        print("Set RUN_GNSS_PRN_SEARCH = True to enable it.")


if __name__ == "__main__":
    main()
