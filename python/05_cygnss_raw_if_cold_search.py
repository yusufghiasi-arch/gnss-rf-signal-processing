"""
05_cygnss_raw_if_cold_search.py

CYGNSS raw IF processing demo.

This script is a Python translation and cleaned portfolio version of the
MATLAB workflow developed for CYGNSS raw IF data.

Workflow
--------
1. Read CYGNSS metadata file
2. Read CYGNSS raw data file
3. Remove DRT0 header if present
4. Unpack CYGNSS 2-bit sign-magnitude samples
5. Separate channels:
   - Channel 0: zenith navigation antenna
   - Channel 1: nadir science antenna
   - Channel 2: nadir science antenna
6. Plot:
   - channel waveforms
   - channel histograms
   - channel spectra
7. Perform cold search over PRNs and Doppler bins
8. Plot strongest peak by PRN
9. Report best PRN, Doppler, delay, and peak quality
10. Build detailed DDM for selected/best PRN
11. Plot:
   - full DDM
   - zoomed DDM
   - delay waveform
   - Doppler waveform

Important
---------
Large .bin files should NOT be uploaded to GitHub.
Place CYGNSS raw .bin files locally in the data/ folder.

Author: Yusof Ghiasi
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

META_FILE = "data/cyg04_raw_if_s20250407_035355_e20250407_035455_meta.bin"
DATA_FILE = "data/cyg04_raw_if_s20250407_035355_e20250407_035455_data.bin"

# CYGNSS constants used in the MATLAB workflow
FS = 16.0362e6
IF_HZ = 3.8722e6
CODE_RATE = 1.023e6

# Number of raw bytes to read from the data file
NUM_BYTES_TO_READ = 3_000_000

# FFT size for spectrum plots
NFFT_SPECTRUM = 262_144

# Cold search settings
COLD_SEARCH_CHANNEL = "ch1"       # "ch0", "ch1", or "ch2"
COLD_SEARCH_NUM_MS = 20
PRN_LIST = np.arange(1, 33)
COLD_DOPPLER_BINS = np.arange(-10_000, 10_001, 500)

# Detailed DDM settings
# If AUTO_DETAILED_FROM_COLD_SEARCH = True, the detailed DDM uses the best
# PRN and Doppler found in the cold search.
AUTO_DETAILED_FROM_COLD_SEARCH = True

DETAILED_CHANNEL = "ch2"          # used only if AUTO_DETAILED_FROM_COLD_SEARCH is False
SELECTED_PRN = 10
CENTER_DOPPLER_HZ = 1500
DETAILED_NUM_MS = 100
FINE_DOPPLER_HALF_WIDTH_HZ = 3000
FINE_DOPPLER_STEP_HZ = 100

# Zoomed DDM window
DELAY_WINDOW_SAMPLES = 600
DOPPLER_WINDOW_HZ = 1500

# Figure options
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURES_DIR = "figures"


# =============================================================================
# Utility functions
# =============================================================================

def finish_figure(filename: str) -> None:
    """Save and/or show the current matplotlib figure."""

    if SAVE_FIGURES:
        fig_dir = Path(FIGURES_DIR)
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def read_uint8_file(filename: str, count: int | None = None) -> np.ndarray:
    """Read uint8 data from a binary file."""

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find file:\n{filename}\n\n"
            "Place the file in the data/ folder or update the filename at the top of this script."
        )

    if count is None:
        data = np.fromfile(path, dtype=np.uint8)
    else:
        data = np.fromfile(path, dtype=np.uint8, count=count)

    return data


def little_endian_uint16(values: np.ndarray) -> int:
    """Convert two uint8 bytes to little-endian uint16."""

    return int(np.frombuffer(values.astype(np.uint8).tobytes(), dtype="<u2")[0])


def little_endian_uint32(values: np.ndarray) -> int:
    """Convert four uint8 bytes to little-endian uint32."""

    return int(np.frombuffer(values.astype(np.uint8).tobytes(), dtype="<u4")[0])


# =============================================================================
# Metadata reading
# =============================================================================

def parse_drt0_metadata(meta: np.ndarray) -> dict:
    """
    Parse the first DRT0 metadata block.

    This follows the indexing used in the MATLAB script.
    """

    if len(meta) < 36:
        raise ValueError("Metadata file is too short to contain DRT0 block.")

    scid = int(meta[0])
    drt0 = meta[1:36]

    packet_type = bytes(drt0[0:4]).decode(errors="replace")

    gps_week = little_endian_uint16(drt0[4:6])
    gps_seconds = little_endian_uint32(drt0[6:10])

    data_format = int(drt0[10])
    sample_rate = little_endian_uint32(drt0[11:15])

    ch0_frontend = int(drt0[15])
    ch0_lo = little_endian_uint32(drt0[16:20])

    ch1_frontend = int(drt0[20])
    ch1_lo = little_endian_uint32(drt0[21:25])

    ch2_frontend = int(drt0[25])
    ch2_lo = little_endian_uint32(drt0[26:30])

    ch3_frontend = int(drt0[30])
    ch3_lo = little_endian_uint32(drt0[31:35])

    return {
        "spacecraft_id": scid,
        "packet_type": packet_type,
        "gps_week": gps_week,
        "gps_seconds": gps_seconds,
        "data_format": data_format,
        "sample_rate": sample_rate,
        "ch0_frontend": ch0_frontend,
        "ch0_lo": ch0_lo,
        "ch1_frontend": ch1_frontend,
        "ch1_lo": ch1_lo,
        "ch2_frontend": ch2_frontend,
        "ch2_lo": ch2_lo,
        "ch3_frontend": ch3_frontend,
        "ch3_lo": ch3_lo,
    }


def print_metadata(metadata: dict, meta_length: int) -> None:
    """Print metadata information."""

    print(f"Metadata bytes = {meta_length}")
    print("")
    print("DRT0 metadata:")
    print(f"Spacecraft ID = {metadata['spacecraft_id']}")
    print(f"Packet type = {metadata['packet_type']}")
    print(f"GPS week = {metadata['gps_week']}")
    print(f"GPS seconds = {metadata['gps_seconds']}")
    print(f"Data format = {metadata['data_format']}")
    print(f"Sample rate = {metadata['sample_rate'] / 1e6:.6f} MHz")
    print(
        f"Channel 0 frontend = {metadata['ch0_frontend']} | "
        f"LO = {metadata['ch0_lo'] / 1e6:.6f} MHz"
    )
    print(
        f"Channel 1 frontend = {metadata['ch1_frontend']} | "
        f"LO = {metadata['ch1_lo'] / 1e6:.6f} MHz"
    )
    print(
        f"Channel 2 frontend = {metadata['ch2_frontend']} | "
        f"LO = {metadata['ch2_lo'] / 1e6:.6f} MHz"
    )
    print(
        f"Channel 3 frontend = {metadata['ch3_frontend']} | "
        f"LO = {metadata['ch3_lo'] / 1e6:.6f} MHz"
    )


# =============================================================================
# CYGNSS 2-bit unpacking
# =============================================================================

def signmag_to_value(sign_bit: int, mag_bit: int) -> int:
    """
    Convert CYGNSS 2-bit sign-magnitude sample to value.

    mag_bit = 0 -> amplitude 1
    mag_bit = 1 -> amplitude 3

    sign_bit = 0 -> positive
    sign_bit = 1 -> negative
    """

    amp = 1 if mag_bit == 0 else 3
    return amp if sign_bit == 0 else -amp


def unpack_cygnss_2bit(byte_array: np.ndarray) -> np.ndarray:
    """
    Unpack CYGNSS 2-bit sign-magnitude samples.

    Each byte contains four 2-bit samples:

    sample 0: bit 8 = sign, bit 7 = magnitude
    sample 1: bit 6 = sign, bit 5 = magnitude
    sample 2: bit 4 = sign, bit 3 = magnitude
    sample 3: bit 2 = sign, bit 1 = magnitude

    This matches the MATLAB bitget implementation.
    """

    byte_array = byte_array.astype(np.uint8)
    n = len(byte_array)

    x = np.zeros(n * 4, dtype=np.float64)

    for k, b in enumerate(byte_array):
        # Python bit numbering: bit 7 is MSB, bit 0 is LSB.
        s0 = (b >> 7) & 1
        m0 = (b >> 6) & 1

        s1 = (b >> 5) & 1
        m1 = (b >> 4) & 1

        s2 = (b >> 3) & 1
        m2 = (b >> 2) & 1

        s3 = (b >> 1) & 1
        m3 = b & 1

        x[4 * k + 0] = signmag_to_value(s0, m0)
        x[4 * k + 1] = signmag_to_value(s1, m1)
        x[4 * k + 2] = signmag_to_value(s2, m2)
        x[4 * k + 3] = signmag_to_value(s3, m3)

    return x


def split_cygnss_channels(raw: np.ndarray) -> dict:
    """
    Split raw CYGNSS bytes into three channels and unpack each channel.

    Byte pattern:
    byte 1 -> channel 0
    byte 2 -> channel 1
    byte 3 -> channel 2
    then repeats.
    """

    first4_alt = bytes(raw[:4]).decode(errors="replace")

    if first4_alt == "DRT0":
        print("Data file starts with DRT0 header, removing first 36 bytes")
        raw = raw[36:]
    else:
        print("No DRT0 header removed from data file")

    num_groups = len(raw) // 3
    raw = raw[: num_groups * 3]

    b0 = raw[0::3]
    b1 = raw[1::3]
    b2 = raw[2::3]

    ch0 = unpack_cygnss_2bit(b0)
    ch1 = unpack_cygnss_2bit(b1)
    ch2 = unpack_cygnss_2bit(b2)

    return {
        "ch0": ch0,
        "ch1": ch1,
        "ch2": ch2,
    }


# =============================================================================
# GPS C/A code generation
# =============================================================================

def generate_ca_code(prn: int) -> np.ndarray:
    """Generate GPS L1 C/A code with values -1/+1."""

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
        raise ValueError("PRN must be between 1 and 32.")

    tap1, tap2 = g2_taps[prn]

    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    ca = np.zeros(1023, dtype=int)

    for i in range(1023):
        g1_output = g1[-1]
        g2_output = g2[tap1 - 1] ^ g2[tap2 - 1]

        ca[i] = g1_output ^ g2_output

        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        g1[1:] = g1[:-1]
        g1[0] = g1_feedback

        g2[1:] = g2[:-1]
        g2[0] = g2_feedback

    return 1 - 2 * ca


def sampled_ca_code_for_one_ms(prn: int, fs: float, samples_per_ms: int) -> np.ndarray:
    """Generate sampled GPS C/A code for one 1-ms block."""

    ca = generate_ca_code(prn)

    t = np.arange(samples_per_ms) / fs
    chip_index = np.floor(t * CODE_RATE).astype(int)

    chip_index[chip_index > 1022] = 1022

    local_code = ca[chip_index].astype(np.float64)

    return local_code


# =============================================================================
# Plots: channels, histograms, spectra
# =============================================================================

def plot_channels(channels: dict, n: int = 2000) -> None:
    """Plot time-domain samples for each CYGNSS channel."""

    labels = {
        "ch0": "Channel 0 - zenith navigation antenna",
        "ch1": "Channel 1 - nadir science antenna",
        "ch2": "Channel 2 - nadir science antenna",
    }

    for name, x in channels.items():
        plt.figure(figsize=(10, 4))
        plt.plot(x[:n])
        plt.grid(True)
        plt.title(labels[name])
        plt.xlabel("Sample number")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        finish_figure(f"06_{name}_time_series.png")


def plot_histograms(channels: dict) -> None:
    """Plot histograms for CYGNSS channels."""

    for name, x in channels.items():
        plt.figure(figsize=(7, 4))
        plt.hist(x, bins=np.arange(-3.5, 4.5, 1.0), edgecolor="black")
        plt.grid(True)
        plt.title(f"{name} histogram")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        finish_figure(f"06_{name}_histogram.png")


def plot_spectra(channels: dict, fs: float, nfft: int) -> None:
    """Plot spectrum for each CYGNSS channel."""

    f = np.linspace(-fs / 2, fs / 2, nfft)

    for name, x in channels.items():
        if len(x) < nfft:
            raise ValueError(f"Not enough samples in {name} for NFFT={nfft}")

        xx = x[:nfft].astype(np.float64)
        xx = xx - np.mean(xx)

        spectrum_power = np.fft.fftshift(np.abs(np.fft.fft(xx)) ** 2)
        spectrum_db = 10 * np.log10(spectrum_power + 1e-12)

        plt.figure(figsize=(10, 4))
        plt.plot(f / 1e6, spectrum_db)
        plt.grid(True)
        plt.title(f"Spectrum - {name}")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power (dB)")
        plt.tight_layout()
        finish_figure(f"06_{name}_spectrum.png")


# =============================================================================
# Acquisition / DDM
# =============================================================================

def build_ddm(
    rx_channel: np.ndarray,
    prn: int,
    fs: float,
    if_hz: float,
    doppler_bins: np.ndarray,
    num_ms: int,
) -> np.ndarray:
    """
    Build noncoherently integrated delay-Doppler map.

    For each 1-ms block:
    - wipe off IF + Doppler
    - correlate with local PRN
    - add correlation power
    """

    samples_per_ms = int(round(fs * 0.001))
    needed_samples = num_ms * samples_per_ms

    if len(rx_channel) < needed_samples:
        raise ValueError("Not enough samples in selected channel.")

    rx = rx_channel[:needed_samples].astype(np.float64)
    rx = rx - np.mean(rx)
    rx = rx.reshape(-1)

    t = np.arange(samples_per_ms) / fs

    local_code = sampled_ca_code_for_one_ms(prn, fs, samples_per_ms)
    code_fft = np.fft.fft(local_code)

    ddm_sum = np.zeros((len(doppler_bins), samples_per_ms), dtype=np.float64)

    for m in range(num_ms):
        start = m * samples_per_ms
        end = start + samples_per_ms

        signal_block = rx[start:end]

        for d, fd in enumerate(doppler_bins):
            carrier = np.exp(-1j * 2 * np.pi * (if_hz + fd) * t)

            signal_wiped = signal_block * carrier

            corr_fft = np.fft.ifft(np.fft.fft(signal_wiped) * np.conj(code_fft))

            ddm_sum[d, :] += np.abs(corr_fft) ** 2

    ddm_avg = ddm_sum / num_ms

    return ddm_avg


def cold_search(
    rx_channel: np.ndarray,
    channel_name: str,
    prn_list: np.ndarray,
    doppler_bins: np.ndarray,
    num_ms: int,
    fs: float,
    if_hz: float,
) -> tuple[np.ndarray, dict]:
    """Cold search over PRNs and Doppler bins."""

    results = []

    print("")
    print("Starting CYGNSS cold search...")
    print(f"Channel = {channel_name}")
    print(f"num_ms = {num_ms}")

    for prn in prn_list:
        ddm_avg = build_ddm(
            rx_channel=rx_channel,
            prn=int(prn),
            fs=fs,
            if_hz=if_hz,
            doppler_bins=doppler_bins,
            num_ms=num_ms,
        )

        max_index = np.unravel_index(np.argmax(ddm_avg), ddm_avg.shape)

        doppler_index = max_index[0]
        delay_index = max_index[1]

        max_value = ddm_avg[doppler_index, delay_index]
        best_doppler = doppler_bins[doppler_index]

        results.append(
            {
                "prn": int(prn),
                "best_doppler": float(best_doppler),
                "best_delay": int(delay_index + 1),  # MATLAB-style display
                "best_delay_python": int(delay_index),
                "peak": float(max_value),
                "ddm": ddm_avg,
            }
        )

        print(
            f"PRN {prn:2d} | Peak = {max_value:.6e} | "
            f"Doppler = {best_doppler:+8.1f} Hz | Delay = {delay_index + 1}"
        )

    result_table = np.array(
        [
            [r["prn"], r["best_doppler"], r["best_delay"], r["peak"]]
            for r in results
        ],
        dtype=np.float64,
    )

    best_result = max(results, key=lambda item: item["peak"])

    return result_table, best_result


def plot_cold_search_results(results_cold: np.ndarray, channel_name: str, num_ms: int) -> None:
    """Plot strongest peak for each PRN."""

    plt.figure(figsize=(10, 5))
    plt.bar(results_cold[:, 0], results_cold[:, 3])
    plt.grid(True)
    plt.title(f"CYGNSS Raw IF cold search - {channel_name} - {num_ms} ms")
    plt.xlabel("PRN")
    plt.ylabel("Peak power")
    plt.tight_layout()
    finish_figure("06_cold_search_peak_by_prn.png")


def print_peak_quality(results_cold: np.ndarray, best_result: dict) -> None:
    """Print best result and peak quality metrics."""

    peaks = results_cold[:, 3]
    sorted_peaks = np.sort(peaks)[::-1]

    print("")
    print("Best cold-search result:")
    print(f"Best PRN = {best_result['prn']}")
    print(f"Best Doppler = {best_result['best_doppler']:.1f} Hz")
    print(f"Best delay = {best_result['best_delay']}")
    print(f"Best peak = {best_result['peak']:.6e}")

    print("")
    print("Cold-search peak quality:")
    print(f"Best / second best = {sorted_peaks[0] / sorted_peaks[1]:.6f}")
    print(f"Best / median = {sorted_peaks[0] / np.median(peaks):.6f}")
    print(f"Best / mean = {sorted_peaks[0] / np.mean(peaks):.6f}")


def detailed_ddm_analysis(
    channels: dict,
    channel_name: str,
    selected_prn: int,
    center_doppler_hz: float,
    num_ms: int,
    fs: float,
    if_hz: float,
) -> None:
    """Build and plot detailed DDM for selected PRN and channel."""

    rx_channel = channels[channel_name]

    doppler_bins_fine = np.arange(
        center_doppler_hz - FINE_DOPPLER_HALF_WIDTH_HZ,
        center_doppler_hz + FINE_DOPPLER_HALF_WIDTH_HZ + FINE_DOPPLER_STEP_HZ,
        FINE_DOPPLER_STEP_HZ,
    )

    ddm_avg = build_ddm(
        rx_channel=rx_channel,
        prn=selected_prn,
        fs=fs,
        if_hz=if_hz,
        doppler_bins=doppler_bins_fine,
        num_ms=num_ms,
    )

    max_index = np.unravel_index(np.argmax(ddm_avg), ddm_avg.shape)

    doppler_index = max_index[0]
    delay_index = max_index[1]

    max_value = ddm_avg[doppler_index, delay_index]
    best_doppler = doppler_bins_fine[doppler_index]
    best_delay = delay_index + 1  # MATLAB-style display

    print("")
    print("Detailed DDM result:")
    print(f"Channel = {channel_name}")
    print(f"PRN = {selected_prn}")
    print(f"Best Doppler = {best_doppler:.1f} Hz")
    print(f"Best delay = {best_delay} samples")
    print(f"Peak = {max_value:.6e}")

    samples_per_ms = int(round(fs * 0.001))
    delay_axis = np.arange(1, samples_per_ms + 1)

    # Full DDM
    plt.figure(figsize=(10, 6))
    plt.imshow(
        ddm_avg,
        aspect="auto",
        origin="lower",
        extent=[
            delay_axis[0],
            delay_axis[-1],
            doppler_bins_fine[0],
            doppler_bins_fine[-1],
        ],
    )
    plt.colorbar(label="Power")
    plt.grid(True)
    plt.scatter(best_delay, best_doppler, marker="o", s=100, facecolors="none", edgecolors="white", linewidths=2)
    plt.title(f"Detailed DDM - {channel_name} - PRN {selected_prn}")
    plt.xlabel("Delay sample")
    plt.ylabel("Doppler (Hz)")
    plt.tight_layout()
    finish_figure("06_detailed_ddm_full.png")

    # Zoomed DDM
    delay_min = max(1, best_delay - DELAY_WINDOW_SAMPLES)
    delay_max = min(samples_per_ms, best_delay + DELAY_WINDOW_SAMPLES)

    doppler_min = best_doppler - DOPPLER_WINDOW_HZ
    doppler_max = best_doppler + DOPPLER_WINDOW_HZ

    delay_min_idx = delay_min - 1
    delay_max_idx = delay_max

    doppler_rows = (doppler_bins_fine >= doppler_min) & (doppler_bins_fine <= doppler_max)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        ddm_avg[doppler_rows, delay_min_idx:delay_max_idx],
        aspect="auto",
        origin="lower",
        extent=[
            delay_min,
            delay_max,
            doppler_bins_fine[doppler_rows][0],
            doppler_bins_fine[doppler_rows][-1],
        ],
    )
    plt.colorbar(label="Power")
    plt.grid(True)
    plt.scatter(best_delay, best_doppler, marker="o", s=100, facecolors="none", edgecolors="white", linewidths=2)
    plt.title(f"Zoomed DDM - {channel_name} - PRN {selected_prn}")
    plt.xlabel("Delay sample")
    plt.ylabel("Doppler (Hz)")
    plt.tight_layout()
    finish_figure("06_detailed_ddm_zoomed.png")

    # Delay waveform at best Doppler
    best_delay_waveform = ddm_avg[doppler_index, :]

    plt.figure(figsize=(10, 4))
    plt.plot(delay_axis, best_delay_waveform)
    plt.grid(True)
    plt.scatter(best_delay, max_value, marker="o", s=80)
    plt.title(f"Delay waveform - PRN {selected_prn} - Doppler {best_doppler:.1f} Hz")
    plt.xlabel("Delay sample")
    plt.ylabel("Power")
    plt.tight_layout()
    finish_figure("06_delay_waveform.png")

    # Doppler waveform at best delay
    best_doppler_waveform = ddm_avg[:, delay_index]

    plt.figure(figsize=(10, 4))
    plt.plot(doppler_bins_fine, best_doppler_waveform)
    plt.grid(True)
    plt.scatter(best_doppler, max_value, marker="o", s=80)
    plt.title(f"Doppler waveform - PRN {selected_prn} - delay {best_delay}")
    plt.xlabel("Doppler (Hz)")
    plt.ylabel("Power")
    plt.tight_layout()
    finish_figure("06_doppler_waveform.png")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run CYGNSS raw IF processing workflow."""

    print("CYGNSS Raw IF Processing Demo")
    print("=============================")
    print(f"Metadata file: {META_FILE}")
    print(f"Data file: {DATA_FILE}")

    # Metadata
    meta = read_uint8_file(META_FILE)
    metadata = parse_drt0_metadata(meta)
    print_metadata(metadata, len(meta))

    samples_per_ms = int(round(FS * 0.001))

    print("")
    print(f"Using Fs = {FS / 1e6:.6f} MHz")
    print(f"Using IF = {IF_HZ / 1e6:.6f} MHz")
    print(f"Samples per 1 ms = {samples_per_ms}")

    # Data
    raw = read_uint8_file(DATA_FILE, count=NUM_BYTES_TO_READ)

    print("")
    print(f"Read data bytes = {len(raw)}")

    print("")
    print("First 40 data bytes:")
    print(raw[:40])

    if len(raw) >= 5:
        first4 = bytes(raw[1:5]).decode(errors="replace")
        if first4 == "DRT":
            print("DRT header seems present in data file")

    channels = split_cygnss_channels(raw)

    print("")
    print(f"Channel 0 samples = {len(channels['ch0'])}")
    print(f"Channel 1 samples = {len(channels['ch1'])}")
    print(f"Channel 2 samples = {len(channels['ch2'])}")

    # Basic plots
    plot_channels(channels)
    plot_histograms(channels)
    plot_spectra(channels, fs=FS, nfft=NFFT_SPECTRUM)

    # Cold search
    rx_channel = channels[COLD_SEARCH_CHANNEL]

    results_cold, best_result = cold_search(
        rx_channel=rx_channel,
        channel_name=COLD_SEARCH_CHANNEL,
        prn_list=PRN_LIST,
        doppler_bins=COLD_DOPPLER_BINS,
        num_ms=COLD_SEARCH_NUM_MS,
        fs=FS,
        if_hz=IF_HZ,
    )

    plot_cold_search_results(
        results_cold=results_cold,
        channel_name=COLD_SEARCH_CHANNEL,
        num_ms=COLD_SEARCH_NUM_MS,
    )

    print_peak_quality(results_cold, best_result)

    # Detailed DDM
    if AUTO_DETAILED_FROM_COLD_SEARCH:
        detailed_channel = COLD_SEARCH_CHANNEL
        selected_prn = best_result["prn"]
        center_doppler = best_result["best_doppler"]
    else:
        detailed_channel = DETAILED_CHANNEL
        selected_prn = SELECTED_PRN
        center_doppler = CENTER_DOPPLER_HZ

    detailed_ddm_analysis(
        channels=channels,
        channel_name=detailed_channel,
        selected_prn=int(selected_prn),
        center_doppler_hz=float(center_doppler),
        num_ms=DETAILED_NUM_MS,
        fs=FS,
        if_hz=IF_HZ,
    )


if __name__ == "__main__":
    main()
