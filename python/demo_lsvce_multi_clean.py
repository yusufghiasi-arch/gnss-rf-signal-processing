"""
demo_lsvce_multi_clean.py

Least-Squares Variance Component Estimation (LS-VCE) demo for multiple
simulated time series.

This script is a GitHub-friendly Python version of a MATLAB LS-VCE demo.
It simulates several geodetic/remote-sensing-style time series containing
trend, seasonal signals, and multiple stochastic noise components. Then it
uses LS-VCE to estimate the contribution of each noise component.

Dependencies:
    numpy
    scipy
    matplotlib

Run:
    python demo_lsvce_multi_clean.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz


@dataclass
class SimulationResult:
    """Container for simulated multi-series time-series data."""

    time: np.ndarray
    observations: np.ndarray
    deterministic_signal: np.ndarray
    noise: np.ndarray
    trend: np.ndarray
    true_covariances: list[np.ndarray]
    offset_index: np.ndarray
    offset_value: np.ndarray
    blunder_index: np.ndarray
    blunder_value: np.ndarray


@dataclass
class LSVCEOutput:
    """Container for LS-VCE outputs."""

    sigma_history: np.ndarray
    qsigma: np.ndarray
    qy: np.ndarray
    qy_inv: np.ndarray
    x_hat: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    projection: np.ndarray
    dof: int
    series_covariance: np.ndarray
    series_correlation: np.ndarray
    log_likelihood: np.ndarray


def make_time_series_design_matrix(time: np.ndarray, frequencies: Iterable[float]) -> np.ndarray:
    """
    Build the deterministic design matrix for a time-series model.

    Model columns:
        intercept, trend, cos/sin annual terms, cos/sin semi-annual terms, etc.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    columns = [np.ones_like(t), t]

    for freq in frequencies:
        columns.append(np.cos(2.0 * np.pi * freq * t))
        columns.append(np.sin(2.0 * np.pi * freq * t))

    return np.column_stack(columns)


def make_power_law_cofactor(alpha: float, year: np.ndarray) -> np.ndarray:
    """
    Create a power-law noise cofactor matrix.

    alpha = 0 behaves like white noise.
    alpha = 1 creates a flicker-like time-correlated noise structure.
    """
    day_numbers = np.rint(np.asarray(year, dtype=float).reshape(-1) * 365.25).astype(int)
    day_numbers[day_numbers < 1] = 1
    max_day = int(day_numbers.max())

    h = np.zeros(max_day)
    h[0] = 1.0
    for i in range(1, max_day):
        # MATLAB equivalent, using one-based index i+1:
        # H(i) = (alpha/2 + i - 2) * H(i-1) / (i - 1)
        h[i] = (alpha / 2.0 + (i + 1) - 2.0) * h[i - 1] / i

    selected_days = np.isin(np.arange(1, max_day + 1), day_numbers)
    u_full = np.triu(toeplitz(h))
    u_selected = u_full[:, selected_days]
    return u_selected.T @ u_selected


def simulate_time_series_with_cofactors(
    num_series: int,
    num_days: int,
    cofactors: list[np.ndarray],
    true_sigma: np.ndarray,
    frequencies: Iterable[float],
    x_true: np.ndarray,
    num_offsets: int = 0,
    num_blunders: int = 0,
    make_plot: bool = True,
    random_seed: int = 7,
) -> SimulationResult:
    """
    Simulate multiple time series using a deterministic model plus colored noise.
    """
    rng = np.random.default_rng(random_seed)
    time = np.arange(1, num_days + 1, dtype=float) / 365.25
    num_noise_models = len(cofactors)

    observations_noise = np.zeros((num_days, num_series))
    true_covariances: list[np.ndarray] = []

    for series_index in range(num_series):
        qy = np.zeros((num_days, num_days))
        for noise_index in range(num_noise_models):
            qy += true_sigma[series_index, noise_index] * cofactors[noise_index]

        qy = symmetrize_with_jitter(qy)
        true_covariances.append(qy)

        # Generate colored noise with covariance qy.
        chol = np.linalg.cholesky(qy)
        observations_noise[:, series_index] = chol @ rng.standard_normal(num_days)

    design = make_time_series_design_matrix(time, frequencies)
    trend = design[:, :2] @ x_true[:2, :]
    deterministic_signal = design[:, 2:] @ x_true[2:, :]
    observations = trend + deterministic_signal + observations_noise

    offset_index = np.array([], dtype=int)
    offset_value = np.empty((0, num_series))
    blunder_index = np.array([], dtype=int)
    blunder_value = np.empty((0, num_series))

    noise_std = observations_noise.std(axis=0, ddof=0)

    if num_offsets > 0:
        offset_index = rng.integers(0, num_days, size=num_offsets)
        offset_value = np.zeros((num_offsets, num_series))
        offset_design = np.tril(np.ones((num_days, num_days)))

        for series_index in range(num_series):
            offset_value[:, series_index] = 3.0 * noise_std[series_index] * (
                2.0 * rng.random(num_offsets) - 1.0
            )
            for offset_number, epoch in enumerate(offset_index):
                observations[:, series_index] += offset_value[offset_number, series_index] * offset_design[:, epoch]

    if num_blunders > 0:
        blunder_index = rng.integers(0, num_days, size=num_blunders)
        blunder_value = np.zeros((num_blunders, num_series))

        for series_index in range(num_series):
            blunder_value[:, series_index] = 3.0 * noise_std[series_index] * (
                2.0 * rng.random(num_blunders) - 1.0
            )
            for blunder_number, epoch in enumerate(blunder_index):
                observations[epoch, series_index] += blunder_value[blunder_number, series_index]

    if make_plot:
        plot_simulated_time_series(time, observations, trend + deterministic_signal)

    return SimulationResult(
        time=time,
        observations=observations,
        deterministic_signal=deterministic_signal,
        noise=observations_noise,
        trend=trend,
        true_covariances=true_covariances,
        offset_index=offset_index,
        offset_value=offset_value,
        blunder_index=blunder_index,
        blunder_value=blunder_value,
    )


def lsvce_multi(
    observations: np.ndarray,
    design_matrix: np.ndarray,
    cofactors: list[np.ndarray],
    initial_sigma: np.ndarray,
    threshold: float = 1e-6,
    max_iter: int = 40,
    compute_log_likelihood: bool = True,
) -> LSVCEOutput:
    """
    Estimate variance components for multiple time series using LS-VCE.
    """
    y = np.asarray(observations, dtype=float)
    a = np.asarray(design_matrix, dtype=float)
    sigma = np.asarray(initial_sigma, dtype=float).reshape(-1)

    num_obs, num_params = a.shape
    _, num_series = y.shape
    num_components = len(cofactors)
    dof = num_obs - num_params

    if dof <= 0:
        raise ValueError("The number of observations must exceed the number of model parameters.")
    if sigma.size != num_components:
        raise ValueError("initial_sigma must have one value for each cofactor matrix.")

    # Store the initial values too. This makes convergence plots meaningful
    # even if LS-VCE converges after only one update.
    sigma_history = [sigma.copy()]
    delta = np.full(num_components, np.inf)
    normal_matrix = None

    print("Running LS-VCE iterations...")

    iteration = 0
    while np.max(delta) > threshold and iteration < max_iter:
        qy = build_covariance(cofactors, sigma)
        qy_inv = np.linalg.solve(qy, np.eye(num_obs))

        normal = a.T @ qy_inv @ a
        qx_hat = np.linalg.solve(normal, np.eye(num_params))
        projection = np.eye(num_obs) - a @ qx_hat @ a.T @ qy_inv

        residuals = projection @ y
        b = qy_inv @ projection
        c = qy_inv @ residuals

        residual_cov_inv = np.linalg.solve(residuals.T @ c / dof, np.eye(num_series))

        l_vector = np.zeros(num_components)
        normal_matrix = np.zeros((num_components, num_components))

        for i in range(num_components):
            l_vector[i] = 0.5 * trace_product(c.T @ cofactors[i] @ c, residual_cov_inv)
            for j in range(num_components):
                normal_matrix[i, j] = 0.5 * num_series * trace_product(
                    b @ cofactors[i], b @ cofactors[j]
                )

        sigma_new, qsigma = nonnegative_least_squares_vce(normal_matrix, l_vector)
        delta = np.abs(sigma_new - sigma)
        sigma = sigma_new
        sigma_history.append(sigma.copy())

        iteration += 1
        print("  Iteration {:02d}: {}".format(iteration, " ".join(f"{v:12.5e}" for v in sigma)))

    sigma_history_array = np.vstack(sigma_history)

    qy = build_covariance(cofactors, sigma)
    qy_inv = np.linalg.solve(qy, np.eye(num_obs))
    at_qinv = a.T @ qy_inv
    qx_hat = np.linalg.solve(at_qinv @ a, np.eye(num_params))
    projection = np.eye(num_obs) - a @ qx_hat @ at_qinv
    x_hat = qx_hat @ (at_qinv @ y)
    y_hat = a @ x_hat
    residuals = projection @ y

    c_final = qy_inv @ residuals
    series_covariance = residuals.T @ c_final / dof
    series_correlation = covariance_to_correlation(series_covariance)

    qsigma = np.linalg.pinv(normal_matrix) if normal_matrix is not None else np.full((num_components, num_components), np.nan)

    log_likelihood = np.array([])
    if compute_log_likelihood:
        eigenvalues = np.linalg.eigvalsh((qy + qy.T) / 2.0)
        eigenvalues[eigenvalues <= 0.0] = np.finfo(float).eps
        log_likelihood = (
            -0.5 * num_obs * np.log(2.0 * np.pi)
            - 0.5 * np.sum(np.log(eigenvalues))
            - 0.5 * np.diag(residuals.T @ qy_inv @ residuals)
        )

    return LSVCEOutput(
        sigma_history=sigma_history_array,
        qsigma=qsigma,
        qy=qy,
        qy_inv=qy_inv,
        x_hat=x_hat,
        y_hat=y_hat,
        residuals=residuals,
        projection=projection,
        dof=dof,
        series_covariance=series_covariance,
        series_correlation=series_correlation,
        log_likelihood=log_likelihood,
    )


def build_covariance(cofactors: list[np.ndarray], sigma: np.ndarray) -> np.ndarray:
    """Form Qy = sum_i sigma_i Q_i."""
    qy = np.zeros_like(cofactors[0], dtype=float)
    for component_sigma, cofactor in zip(sigma, cofactors):
        qy += component_sigma * cofactor
    return symmetrize_with_jitter(qy)


def symmetrize_with_jitter(matrix: np.ndarray, jitter: float = 1e-14) -> np.ndarray:
    """Force symmetry and add a tiny diagonal stabilizer."""
    matrix = (matrix + matrix.T) / 2.0
    return matrix + jitter * np.eye(matrix.shape[0])


def nonnegative_least_squares_vce(normal_matrix: np.ndarray, l_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the small non-negative LS problem used in the VCE update.

    This follows the simple iterative scheme used in the original MATLAB
    helper function nnls_v.m.
    """
    n = np.asarray(normal_matrix, dtype=float)
    l = np.asarray(l_vector, dtype=float).reshape(-1)

    num_components = l.size
    mu = -l.copy()
    s_old = np.zeros(num_components)
    s_test = s_old + 1.0
    s = s_old.copy()

    while np.linalg.norm(s - s_test) > 1e-12:
        for k in range(num_components):
            if abs(n[k, k]) < np.finfo(float).eps:
                s[k] = 0.0
            else:
                s[k] = max(0.0, s_old[k] - mu[k] / n[k, k])
            mu = mu + (s[k] - s_old[k]) * n[:, k]
        s_test = s_old.copy()
        s_old = s.copy()

    zero_indices = np.where(s == 0.0)[0]
    if zero_indices.size == 0:
        qs = np.linalg.pinv(n)
    else:
        ct = np.zeros((zero_indices.size, num_components))
        for row, index in enumerate(zero_indices):
            ct[row, index] = 1.0
        c = ct.T
        n_inv = np.linalg.pinv(n)
        middle = np.linalg.pinv(ct @ n_inv @ c)
        pco = np.eye(num_components) - c @ middle @ ct @ n_inv
        qs = n_inv @ pco

    return s, qs


def trace_product(a: np.ndarray, b: np.ndarray) -> float:
    """Equivalent to the original MATLAB helper trace2(A, B) = trace(A*B)."""
    return float(np.trace(a @ b))


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a standard correlation matrix."""
    diagonal_std = np.sqrt(np.abs(np.diag(covariance)))
    denominator = np.outer(diagonal_std, diagonal_std)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator

    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def covariance_to_original_corr2_display(covariance: np.ndarray) -> np.ndarray:
    """
    Reproduce the original custom MATLAB corr2.m display style.

    Off-diagonal entries are correlations.
    Diagonal entries are 10 * standard deviations.
    """
    num_series = covariance.shape[0]
    display_matrix = np.zeros_like(covariance, dtype=float)

    for i in range(num_series - 1):
        for j in range(i + 1, num_series):
            denominator = np.sqrt(covariance[i, i] * covariance[j, j])
            display_matrix[i, j] = covariance[i, j] / denominator if denominator > 0 else 0.0

    display_matrix = display_matrix + display_matrix.T
    np.fill_diagonal(display_matrix, 10.0 * np.sqrt(np.diag(covariance)))
    return display_matrix


def plot_simulated_time_series(time: np.ndarray, observations: np.ndarray, deterministic: np.ndarray) -> None:
    """Plot the simulated observations and the known deterministic signal."""
    num_series = observations.shape[1]
    fig, axes = plt.subplots(num_series, 1, sharex=True, figsize=(10, 7))
    if num_series == 1:
        axes = [axes]

    for series_index, axis in enumerate(axes):
        axis.plot(time, observations[:, series_index], label="Simulated observations")
        axis.plot(time, deterministic[:, series_index], label="True deterministic signal", linewidth=1.5)
        axis.set_ylabel(f"Series {series_index + 1} [m]")
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time [year]")
    fig.suptitle("Simulated multi-series time series")
    fig.tight_layout()


def plot_lsvce_convergence(sigma_history: np.ndarray, noise_exponents: Iterable[float]) -> None:
    """Plot the convergence of the estimated variance components."""
    plt.figure(figsize=(8, 5))
    for component_index, exponent in enumerate(noise_exponents):
        plt.plot(
            np.arange(0, sigma_history.shape[0]),
            sigma_history[:, component_index],
            linewidth=1.5,
            marker="o",
            label=f"Noise exponent {exponent:g}",
        )
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.xlim(0, max(1, sigma_history.shape[0] - 1))
    plt.ylabel("Estimated variance component")
    plt.title("LS-VCE variance-component convergence")
    plt.legend(loc="best")
    plt.tight_layout()


def main() -> None:
    """Run the LS-VCE demo."""
    np.set_printoptions(precision=6, suppress=False)

    # ------------------------------------------------------------------
    # User settings
    # ------------------------------------------------------------------
    num_series = 3
    num_days = 180              # Increase to 365, 3*365, or 10*365 for a larger/full-size demo.
    frequencies = [1, 2]        # Annual and semi-annual cycles [cycles/year].
    noise_exponents = [0, 1]    # 0 = white noise, 1 = flicker-like noise.
    make_plots = True
    save_figures = True       # Saves PNG figures for GitHub/README use.

    time = np.arange(1, num_days + 1, dtype=float) / 365.25
    dt = 1.0 / 365.25

    # ------------------------------------------------------------------
    # Build cofactor matrices for the selected power-law noise models.
    # ------------------------------------------------------------------
    cofactors = []
    for exponent in noise_exponents:
        cofactor = make_power_law_cofactor(exponent, time - time[0] + dt)
        cofactor = cofactor * dt ** (exponent / 2.0)
        cofactors.append(cofactor)

    # Rows are time series; columns are variance components.
    # These are variances, not standard deviations.
    true_sigma = np.array(
        [
            [0.0015, 0.0030],
            [0.0020, 0.0040],
            [0.0030, 0.0060],
        ],
        dtype=float,
    ) ** 2

    # Model columns are:
    # intercept, trend, cos(annual), sin(annual), cos(semiannual), sin(semiannual).
    x_true = np.zeros((2 + 2 * len(frequencies), num_series))
    x_true[:, 0] = np.array([100, 5, np.sqrt(2), np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]) / 1000.0
    x_true[:, 1] = np.array([100, 5, np.sqrt(2), np.sqrt(3), 1, 1 / np.sqrt(2)]) / 1000.0
    x_true[:, 2] = np.array([100, 1, np.sqrt(4), np.sqrt(5), np.sqrt(2), np.sqrt(2)]) / 1000.0

    # ------------------------------------------------------------------
    # Simulate observations.
    # ------------------------------------------------------------------
    simulation = simulate_time_series_with_cofactors(
        num_series=num_series,
        num_days=num_days,
        cofactors=cofactors,
        true_sigma=true_sigma,
        frequencies=frequencies,
        x_true=x_true,
        num_offsets=0,
        num_blunders=0,
        make_plot=make_plots,
        random_seed=7,
    )

    # ------------------------------------------------------------------
    # Run LS-VCE.
    # ------------------------------------------------------------------
    design_matrix = make_time_series_design_matrix(simulation.time, frequencies)
    initial_sigma = np.repeat((0.001) ** 2, len(noise_exponents))

    output = lsvce_multi(
        observations=simulation.observations,
        design_matrix=design_matrix,
        cofactors=cofactors,
        initial_sigma=initial_sigma,
        # Variance components are around 1e-6, so 1e-6 is too loose
        # and can stop the iteration after only one update.
        threshold=1e-12,
        max_iter=40,
        compute_log_likelihood=True,
    )

    final_sigma = output.sigma_history[-1, :]
    estimated_std = np.sqrt(np.kron(np.diag(output.series_covariance), final_sigma).reshape(num_series, -1))
    true_std = np.sqrt(true_sigma)

    print("\nFinal estimated variance components:")
    print(final_sigma)

    print("\nEstimated noise standard deviations by series and noise component:")
    print(estimated_std)

    print("\nTrue noise standard deviations used in the simulation:")
    print(true_std)

    print("\nEstimated inter-series covariance matrix S:")
    print(output.series_covariance)

    print("\nEstimated inter-series correlation matrix, diagonal = 1:")
    print(output.series_correlation)

    print("\nOriginal corr2-style display matrix: off-diagonal = correlation, diagonal = 10*std:")
    print(covariance_to_original_corr2_display(output.series_covariance))

    if output.log_likelihood.size:
        print("\nLog-likelihood by time series:")
        print(output.log_likelihood)

    if make_plots:
        plot_lsvce_convergence(output.sigma_history, noise_exponents)
        if save_figures:
            output_dir = Path(__file__).resolve().parent / "figures"
            output_dir.mkdir(exist_ok=True)
            for number in plt.get_fignums():
                plt.figure(number)
                plt.savefig(output_dir / f"lsvce_demo_figure_{number}.png", dpi=200, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
