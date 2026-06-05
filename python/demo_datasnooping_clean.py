"""
demo_datasnooping_clean.py

GitHub-friendly Python demo for time-series data snooping.

This script simulates a geodetic/remote-sensing-style time series with:
  - deterministic terms: offset, linear trend, annual and semi-annual cycles
  - stochastic noise: white noise + flicker-like power-law noise
  - artificial blunders/outliers

It then applies Baarda-style data snooping using normalized residuals
(w-test) to identify suspicious epochs. The test can be run in two modes:
  1) white-noise assumption
  2) colored-noise covariance estimated with LS-VCE

The code is self-contained and includes all helper functions in this file.

Original MATLAB concept: Yusof Ghiasi, 2016
Cleaned/reorganized and translated to Python for portfolio/GitHub use.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SimulationResult:
    time: np.ndarray
    y: np.ndarray
    y_signal: np.ndarray
    y_noise: np.ndarray
    y_trend: np.ndarray
    offset_index: np.ndarray
    offset_value: np.ndarray
    blunder_index: np.ndarray
    blunder_value: np.ndarray
    qy_true: np.ndarray


@dataclass
class SnoopingDetails:
    threshold: float
    max_abs_w: list[float]
    removed_indices: list[int]


def main() -> None:
    """Run the complete data-snooping demo."""

    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------
    # 1. Simulation settings
    # ------------------------------------------------------------------
    # Keep this moderate for Python/NumPy, because covariance matrices are
    # dense. Increase to 3*365 or 10*365 for heavier experiments.
    num_days = 90
    time = np.arange(1, num_days + 1, dtype=float) / 365.25

    frequencies = [1, 2]          # annual and semi-annual terms
    noise_index = [0, 1]          # 0 = white noise, 1 = flicker-like noise

    # x = [offset, trend, cos_annual, sin_annual, cos_semiannual, sin_semiannual]
    x_true = np.array([
        1.0,
        1.5e-3,
        np.sqrt(5) / 1000,
        np.sqrt(4) / 1000,
        np.sqrt(3) / 1000,
        np.sqrt(1) / 1000,
    ])

    # Variance components used in the simulation. These are variances, not STD.
    sigma_true = np.array([0.002, 0.004]) ** 2

    n_offset = 0
    n_blunder = 2
    alpha = 0.001                 # two-sided significance level

    # ------------------------------------------------------------------
    # 2. Build cofactor matrices and simulate observations
    # ------------------------------------------------------------------
    q_mats = build_power_law_cofactors(noise_index, time)

    sim = simulate_time_series_with_cofactor(
        num_days=num_days,
        q_mats=q_mats,
        sigma=sigma_true,
        frequencies=frequencies,
        x_true=x_true,
        n_offset=n_offset,
        n_blunder=n_blunder,
        rng=rng,
    )

    design = make_time_series_design_matrix(time, frequencies)

    # ------------------------------------------------------------------
    # 3. Data snooping with LS-VCE colored covariance model
    # ------------------------------------------------------------------
    print("\nRunning data snooping with LS-VCE covariance model...")
    detected_colored, detail_colored = data_snooping(
        y=sim.y,
        design=design,
        alpha=alpha,
        q_mats=q_mats,
    )

    # ------------------------------------------------------------------
    # 4. Data snooping with white-noise covariance model
    # ------------------------------------------------------------------
    print("\nRunning data snooping with white-noise covariance model...")
    detected_white, detail_white = data_snooping(
        y=sim.y,
        design=design,
        alpha=alpha,
        q_mats=None,
    )

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n================ DATA SNOOPING SUMMARY ================")
    print(f"True blunder indices, 1-based:        {to_one_based(sim.blunder_index)}")
    print(f"Detected, LS-VCE covariance, 1-based: {to_one_based(detected_colored)}")
    print(f"Detected, white covariance, 1-based:  {to_one_based(detected_white)}")
    print(f"Two-sided alpha:                     {alpha:g}")
    print(f"Critical |w| threshold:              {detail_colored.threshold:.3f}")
    print("=======================================================")

    # ------------------------------------------------------------------
    # 6. Plot results
    # ------------------------------------------------------------------
    plot_results(sim, detected_colored, detected_white, detail_colored, detail_white)
    plt.show()


def to_one_based(indices: Iterable[int]) -> list[int]:
    """Convert zero-based Python indices to one-based MATLAB-style indices."""
    return [int(i) + 1 for i in np.asarray(indices, dtype=int).ravel()]


def make_time_series_design_matrix(time: np.ndarray, frequencies: Iterable[float]) -> np.ndarray:
    """Design matrix for offset, trend, annual, and semi-annual terms."""
    time = np.asarray(time, dtype=float).ravel()
    frequencies = list(frequencies)

    design = np.zeros((time.size, 2 + 2 * len(frequencies)))
    design[:, 0] = 1.0
    design[:, 1] = time

    for k, freq in enumerate(frequencies):
        design[:, 2 * k + 2] = np.cos(2 * np.pi * freq * time)
        design[:, 2 * k + 3] = np.sin(2 * np.pi * freq * time)

    return design


def build_power_law_cofactors(noise_index: Iterable[float], time: np.ndarray) -> list[np.ndarray]:
    """Build one cofactor matrix for each requested power-law noise exponent."""
    time = np.asarray(time, dtype=float).ravel()
    dt = 1.0 / 365.25
    t_rel = time - time[0] + dt

    q_mats = []
    for alpha in noise_index:
        q = make_power_law_cofactor(alpha, t_rel)
        q = q * dt ** (alpha / 2.0)
        q_mats.append(q)

    return q_mats


def make_power_law_cofactor(alpha: float, time_years: np.ndarray) -> np.ndarray:
    """
    Generate a simple power-law cofactor matrix.

    alpha = 0 produces a white-noise-like identity cofactor.
    alpha = 1 produces a flicker-like cofactor.
    """
    day_index = np.rint(np.asarray(time_years).ravel() * 365.25).astype(int)
    day_index[day_index < 1] = 1
    m_full = int(day_index[-1])

    h = np.zeros(m_full)
    h[0] = 1.0
    for i in range(1, m_full):
        # MATLAB equivalent:
        # H(i) = (alpha/2 + i - 2) * H(i-1) / (i - 1), for one-based i >= 2
        h[i] = (alpha / 2.0 + i - 1) * h[i - 1] / i

    # Toeplitz matrix with first column/row h, then keep the upper triangle.
    idx = np.arange(m_full)
    toeplitz_h = h[np.abs(idx[:, None] - idx[None, :])]
    u_full = np.triu(toeplitz_h)

    keep = np.isin(np.arange(1, m_full + 1), day_index)
    u = u_full[:, keep]
    q = u.T @ u
    return 0.5 * (q + q.T)


def simulate_time_series_with_cofactor(
    num_days: int,
    q_mats: list[np.ndarray],
    sigma: np.ndarray,
    frequencies: Iterable[float],
    x_true: np.ndarray,
    n_offset: int,
    n_blunder: int,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate observations with deterministic signal, colored noise, and blunders."""
    time = np.arange(1, num_days + 1, dtype=float) / 365.25
    sigma = np.asarray(sigma, dtype=float).ravel()
    x_true = np.asarray(x_true, dtype=float).ravel()

    qy = np.zeros((num_days, num_days))
    for sk, qk in zip(sigma, q_mats):
        qy += sk * qk
    qy = symmetrize(qy) + 1e-14 * np.eye(num_days)

    # Cholesky simulation of colored Gaussian noise.
    chol = np.linalg.cholesky(qy)
    y_noise = chol @ rng.standard_normal(num_days)

    design = make_time_series_design_matrix(time, frequencies)
    y_trend = design[:, :2] @ x_true[:2]
    y_signal = design[:, 2:] @ x_true[2:]
    y = y_trend + y_signal + y_noise

    offset_index = np.array([], dtype=int)
    offset_value = np.array([], dtype=float)
    if n_offset > 0:
        offset_index = np.sort(rng.choice(num_days, size=n_offset, replace=False))
        offset_value = 10 * np.std(y_noise) * (2 * rng.random(n_offset) - 1)
        for idx, value in zip(offset_index, offset_value):
            y[idx:] += value

    blunder_index = np.array([], dtype=int)
    blunder_value = np.array([], dtype=float)
    if n_blunder > 0:
        blunder_index = np.sort(rng.choice(num_days, size=n_blunder, replace=False))
        blunder_sign = rng.choice([-1.0, 1.0], size=n_blunder)
        blunder_value = 8 * np.std(y_noise) * blunder_sign
        y[blunder_index] += blunder_value

    return SimulationResult(
        time=time,
        y=y,
        y_signal=y_signal,
        y_noise=y_noise,
        y_trend=y_trend,
        offset_index=offset_index,
        offset_value=offset_value,
        blunder_index=blunder_index,
        blunder_value=blunder_value,
        qy_true=qy,
    )


def data_snooping(
    y: np.ndarray,
    design: np.ndarray,
    alpha: float,
    q_mats: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, SnoopingDetails]:
    """
    Iterative Baarda-style data snooping.

    At each iteration, the observation with the largest absolute normalized
    residual is removed if it exceeds the critical threshold.
    """
    y_work = np.asarray(y, dtype=float).ravel().copy()
    a_work = np.asarray(design, dtype=float).copy()
    original_index = np.arange(y_work.size)

    # Equivalent to MATLAB norminv(1 - alpha/2), without SciPy dependency.
    threshold = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    use_colored_covariance = q_mats is not None and len(q_mats) > 0
    if use_colored_covariance:
        q_work = [np.asarray(q, dtype=float).copy() for q in q_mats]
        sigma0 = 1e-6 * np.ones(len(q_work))
    else:
        q_work = []
        sigma0 = np.array([])

    detected = []
    max_abs_w = []

    while True:
        m_current = y_work.size

        if use_colored_covariance:
            lsvce_result = lsvce(
                y=y_work,
                design=a_work,
                q_mats=q_work,
                sigma0=sigma0,
                threshold=1e-12,
                max_iter=12,
            )
            qy = lsvce_result["qy"]
            qy_inv = lsvce_result["qy_inv"]
            ehat = lsvce_result["residuals"]
            pao = lsvce_result["pao"]
            q_e = pao @ qy

            w = np.zeros(m_current)
            for i in range(m_current):
                b = qy_inv[i, :]
                denom = np.sqrt(max(float(b @ q_e @ b.T), np.finfo(float).eps))
                w[i] = float(b @ ehat) / denom
        else:
            qy = np.eye(m_current)
            ls_result = least_squares_estimate(y_work, a_work, qy)
            ehat = ls_result["residuals"]
            pao = ls_result["pao"]
            q_e = pao @ qy
            w = ehat / np.sqrt(np.maximum(np.diag(q_e), np.finfo(float).eps))

        local_max_index = int(np.argmax(np.abs(w)))
        this_max = float(np.abs(w[local_max_index]))
        max_abs_w.append(this_max)

        if len(detected) >= 10:
            print("Warning: data snooping stopped after 10 removals for demo safety.")
            break

        if this_max <= threshold:
            break

        detected.append(int(original_index[local_max_index]))

        y_work = np.delete(y_work, local_max_index)
        a_work = np.delete(a_work, local_max_index, axis=0)
        original_index = np.delete(original_index, local_max_index)

        if use_colored_covariance:
            q_work = [np.delete(np.delete(q, local_max_index, axis=0), local_max_index, axis=1) for q in q_work]

        if y_work.size <= a_work.shape[1] + 1:
            print("Warning: data snooping stopped because too few observations remain.")
            break

    details = SnoopingDetails(
        threshold=threshold,
        max_abs_w=max_abs_w,
        removed_indices=detected,
    )
    return np.sort(np.asarray(detected, dtype=int)), details


def lsvce(
    y: np.ndarray,
    design: np.ndarray,
    q_mats: list[np.ndarray],
    sigma0: np.ndarray,
    threshold: float = 1e-12,
    max_iter: int = 12,
) -> dict[str, np.ndarray | list[np.ndarray] | float]:
    """Least-Squares Variance Component Estimation with non-negative constraints."""
    y = np.asarray(y, dtype=float).ravel()
    design = np.asarray(design, dtype=float)
    sigma_current = np.asarray(sigma0, dtype=float).ravel().copy()

    p = len(q_mats)
    m = y.size
    history = []
    last_nvc = np.full((p, p), np.nan)

    for _ in range(max_iter):
        qy = combine_covariance(q_mats, sigma_current)
        qy_inv = stable_inverse(qy)

        if design.size == 0:
            pao = np.eye(m)
        else:
            normal = design.T @ qy_inv @ design
            pao = np.eye(m) - design @ stable_solve(normal, design.T @ qy_inv)

        residuals = pao @ y
        b_mat = qy_inv @ pao
        c_vec = qy_inv @ residuals

        l_vec = np.zeros(p)
        nvc = np.zeros((p, p))
        for i in range(p):
            qi = q_mats[i]
            l_vec[i] = 0.5 * float(c_vec.T @ qi @ c_vec)
            for j in range(p):
                qj = q_mats[j]
                nvc[i, j] = 0.5 * np.trace(b_mat @ qi @ b_mat @ qj)

        sigma_new, q_sigma = non_negative_variance_ls(nvc, l_vec)
        history.append(sigma_new.copy())
        last_nvc = nvc

        if np.max(np.abs(sigma_new - sigma_current)) <= threshold:
            sigma_current = sigma_new
            break
        sigma_current = sigma_new

    qy = combine_covariance(q_mats, sigma_current)
    qy_inv = stable_inverse(qy)

    if design.size == 0:
        pao = np.eye(m)
        xhat = np.array([])
        yhat = np.array([])
        residuals = pao @ y
    else:
        at_qinv = design.T @ qy_inv
        q_xhat = stable_inverse(at_qinv @ design)
        pao = np.eye(m) - design @ q_xhat @ at_qinv
        xhat = q_xhat @ at_qinv @ y
        yhat = design @ xhat
        residuals = pao @ y

    try:
        q_sigma = stable_inverse(last_nvc)
    except Exception:
        q_sigma = np.full((p, p), np.nan)

    return {
        "sigma": sigma_current,
        "sigma_history": np.asarray(history),
        "q_sigma": q_sigma,
        "qy": qy,
        "qy_inv": qy_inv,
        "xhat": xhat,
        "yhat": yhat,
        "residuals": residuals,
        "pao": pao,
    }


def least_squares_estimate(y: np.ndarray, design: np.ndarray, qy: np.ndarray) -> dict[str, np.ndarray | float]:
    """Weighted least-squares estimate for a known covariance matrix."""
    y = np.asarray(y, dtype=float).ravel()
    design = np.asarray(design, dtype=float)
    qy = np.asarray(qy, dtype=float)

    qy_inv = stable_inverse(qy)
    normal = design.T @ qy_inv @ design
    normal_inv = stable_inverse(normal)

    xhat = normal_inv @ design.T @ qy_inv @ y
    yhat = design @ xhat
    pao = np.eye(y.size) - design @ normal_inv @ design.T @ qy_inv
    residuals = pao @ y
    df = design.shape[0] - design.shape[1]
    sigma2 = float(residuals.T @ qy_inv @ residuals) / df

    return {
        "xhat": xhat,
        "yhat": yhat,
        "residuals": residuals,
        "sigma2": sigma2,
        "pao": pao,
        "df": float(df),
        "qy_scaled": sigma2 * qy,
    }


def non_negative_variance_ls(n_mat: np.ndarray, l_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple coordinate active-set-like non-negative variance solver.

    It solves approximately N*s = L subject to s >= 0, matching the logic of
    the original MATLAB helper function.
    """
    n_mat = np.asarray(n_mat, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    p = l_vec.size

    mu0 = -l_vec.copy()
    s0 = np.zeros(p)
    s = s0.copy()
    previous = s0 + 1.0

    while np.linalg.norm(s - previous) > 1e-12:
        for k in range(p):
            denom = n_mat[k, k] if abs(n_mat[k, k]) > np.finfo(float).eps else np.finfo(float).eps
            s[k] = max(0.0, s0[k] - mu0[k] / denom)
            mu0 = mu0 + (s[k] - s0[k]) * n_mat[:, k]
        previous = s0.copy()
        s0 = s.copy()

    try:
        q_s = stable_inverse(n_mat)
    except Exception:
        q_s = np.full((p, p), np.nan)

    return s, q_s


def combine_covariance(q_mats: list[np.ndarray], sigma: np.ndarray) -> np.ndarray:
    """Combine cofactor matrices and variance components into one covariance matrix."""
    sigma = np.asarray(sigma, dtype=float).ravel()
    qy = np.zeros_like(q_mats[0], dtype=float)
    for sk, qk in zip(sigma, q_mats):
        qy += sk * qk
    return symmetrize(qy) + 1e-14 * np.eye(qy.shape[0])


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force a matrix to be symmetric within numerical precision."""
    return 0.5 * (matrix + matrix.T)


def stable_inverse(matrix: np.ndarray) -> np.ndarray:
    """Compute an inverse, falling back to pseudo-inverse if needed."""
    try:
        return np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def stable_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a linear system, falling back to least squares if needed."""
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


# -------------------------------------------------------------------------
# Optional power-law w-test utilities from the original MATLAB toolkit.
# These are not required by the main demo, but are included for completeness.
# -------------------------------------------------------------------------

def power_law_w_test(y: np.ndarray, frequencies: Iterable[float], h0: Iterable[float]) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Test candidate power-law noise exponents against a null model H0."""
    y = np.asarray(y, dtype=float).ravel()
    m = y.size
    time = np.arange(1, m + 1, dtype=float) / 365.25
    design = make_time_series_design_matrix(time, frequencies)
    h0 = np.asarray(list(h0), dtype=float).ravel()

    if np.any(h0 != 0):
        q_mats = build_power_law_cofactors(h0, time)
        result = lsvce(y, design, q_mats, 1e-6 * np.ones(len(q_mats)))
        qy = result["qy"]
        residuals = result["residuals"]
        pao = result["pao"]
        df = m - design.shape[1]
    else:
        result = least_squares_estimate(y, design, np.eye(m))
        qy = result["qy_scaled"]
        residuals = result["residuals"]
        pao = result["pao"]
        df = int(result["df"])

    index_grid = np.round(np.arange(-1, 3.0001, 0.1), 2)
    index_grid = np.array([idx for idx in index_grid if not np.any(np.isclose(idx, h0))])
    w_values = np.zeros_like(index_grid, dtype=float)

    for k, idx in enumerate(index_grid):
        cy = build_power_law_cofactors([idx], time)[0]
        if np.any(h0 != 0):
            w_values[k] = w_test_q0(residuals, qy, pao, cy, df)
        else:
            sigma = np.sqrt(float(result["sigma2"]))
            w_values[k] = w_test_white(residuals, cy, pao, sigma, df)

    best = int(np.argmax(w_values))
    return float(index_grid[best]), float(w_values[best]), index_grid, w_values


def w_test_white(residuals: np.ndarray, cy: np.ndarray, pao: np.ndarray, sigma: float, df: int) -> float:
    """w-test under a white-noise null hypothesis."""
    a1 = df * residuals.T @ cy @ residuals
    a2 = np.trace(cy @ pao) * (residuals.T @ residuals)
    numerator = a1 - a2
    b1 = 2 * df**2 * np.trace(cy @ pao @ cy @ pao)
    b2 = 2 * df * np.trace(cy @ pao) ** 2
    denominator = sigma**2 * np.sqrt(max(float(b1 - b2), np.finfo(float).eps))
    return float(numerator / denominator)


def w_test_s2q1(residuals: np.ndarray, qy: np.ndarray, pao: np.ndarray, cy: np.ndarray, df: int) -> float:
    """Power-law w-test, following the original MATLAB helper."""
    qy_inv = stable_inverse(qy)
    qehat_r_inv = qy_inv @ pao
    term1 = 0.5 * cy - (np.trace(cy @ qehat_r_inv) / (2 * df)) * qy
    numerator = residuals.T @ qy_inv @ term1 @ qy_inv @ residuals
    denominator = np.sqrt(max(float(0.5 * np.trace(cy @ qehat_r_inv @ cy @ qehat_r_inv)
                                    - (1 / (2 * df)) * np.trace(cy @ qehat_r_inv) ** 2),
                              np.finfo(float).eps))
    return float(numerator / denominator)


def w_test_q0(residuals: np.ndarray, qy: np.ndarray, pao: np.ndarray, cy: np.ndarray, df: int) -> float:
    """Power-law w-test under a colored-noise null hypothesis."""
    qy_inv = stable_inverse(qy)
    qehat_r_inv = qy_inv @ pao
    cqr = cy @ qehat_r_inv
    wd = np.sqrt(max(float(0.5 * np.trace(cqr @ cqr)), np.finfo(float).eps))
    m_mat = (0.5 * qy_inv @ cy @ qy_inv) / wd
    m0 = (0.5 * np.trace(cqr)) / wd
    return float(residuals.T @ m_mat @ residuals - m0)


def plot_results(
    sim: SimulationResult,
    detected_colored: np.ndarray,
    detected_white: np.ndarray,
    detail_colored: SnoopingDetails,
    detail_white: SnoopingDetails,
) -> None:
    """Create the two main diagnostic figures."""
    plt.figure("Simulated time series")
    plt.plot(sim.time, sim.y, ".", label="Simulated observations")
    plt.plot(sim.time, sim.y_trend + sim.y_signal, label="True deterministic signal")

    if sim.blunder_index.size > 0:
        plt.plot(sim.time[sim.blunder_index], sim.y[sim.blunder_index], "o", label="True blunders")
    if detected_colored.size > 0:
        plt.plot(sim.time[detected_colored], sim.y[detected_colored], "x", label="Detected by LS-VCE model")
    if detected_white.size > 0:
        plt.plot(sim.time[detected_white], sim.y[detected_white], "s", label="Detected by white-noise model")

    plt.xlabel("Time (years)")
    plt.ylabel("Simulated observation")
    plt.title("Baarda-style data snooping for simulated time series")
    plt.grid(True)
    plt.legend()

    plt.figure("Maximum w-test value per iteration")
    plt.plot(np.arange(1, len(detail_colored.max_abs_w) + 1), detail_colored.max_abs_w, "-o", label="LS-VCE covariance")
    plt.plot(np.arange(1, len(detail_white.max_abs_w) + 1), detail_white.max_abs_w, "-s", label="White covariance")
    plt.axhline(detail_colored.threshold, linestyle="--", label="Critical threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum |w|")
    plt.title("Data snooping iteration history")
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":
    main()
