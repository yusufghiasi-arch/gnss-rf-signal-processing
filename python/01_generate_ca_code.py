"""
01_generate_ca_code.py

Generate and plot a GPS L1 C/A PRN code.

This script demonstrates the first step in GNSS signal processing:
creating a local PRN replica that can later be correlated with raw IF/IQ
samples during acquisition or delay-Doppler processing.

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

    # G2 tap selection for GPS PRNs 1 to 32.
    # Each PRN uses a different pair of G2 taps.
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

    # Initialize G1 and G2 shift registers with ones.
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    ca_code = np.zeros(1023, dtype=int)

    for i in range(1023):
        # G1 output is the last register.
        g1_output = g1[-1]

        # G2 output is the modulo-2 sum of the selected taps.
        # Convert tap numbers from 1-based to 0-based indexing.
        g2_output = g2[tap1 - 1] ^ g2[tap2 - 1]

        # C/A code chip.
        ca_code[i] = g1_output ^ g2_output

        # Feedback definitions for GPS C/A code.
        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        # Shift registers.
        g1[1:] = g1[:-1]
        g1[0] = g1_feedback

        g2[1:] = g2[:-1]
        g2[0] = g2_feedback

    # Convert binary 0/1 code to bipolar -1/+1 code.
    ca_code = 1 - 2 * ca_code

    return ca_code


def main() -> None:
    """Run the GPS C/A code generation demo."""

    prn = 1
    ca_code = generate_ca_code(prn)

    print(f"Generated GPS L1 C/A code for PRN {prn}")
    print(f"Code length: {len(ca_code)} chips")
    print(f"Unique chip values: {np.unique(ca_code)}")

    # Plot the first 100 chips.
    plt.figure(figsize=(10, 4))
    plt.step(np.arange(100), ca_code[:100], where="post")
    plt.xlabel("Chip index")
    plt.ylabel("Amplitude")
    plt.title(f"GPS L1 C/A Code, PRN {prn}, First 100 Chips")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
