#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.backend import CUPY_AVAILABLE, Timer, get_backend, sync, to_cpu

try:
    import cupy as cp  # noqa: F401
except Exception:
    cp = None


# ----------------------------
# Generic helpers
# ----------------------------

def ensure_dir(path: Path) -> None:
    """Create directory and all parent directories if they don't exist.
    
    Args:
        path: Path object for the directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, header: str, arr: np.ndarray) -> None:
    """Save a NumPy array to CSV file with header.
    
    Args:
        path: Path object for output CSV file.
        header: Header string for the CSV file.
        arr: NumPy array to save.
    """
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def setup_matplotlib() -> None:
    """Configure matplotlib plotting parameters for consistent appearance."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.8),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def slope_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ~ exp(intercept) * x^slope in log-log space."""
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept


def write_text(path: Path, text: str) -> None:
    """Write text to a file with UTF-8 encoding.
    
    Args:
        path: Path object for output file.
        text: Text content to write.
    """
    path.write_text(text, encoding="utf-8")


# ----------------------------
# Problem 1: explicit IVP solvers
# ----------------------------

def rhs_decay(t, y):
    """Right-hand side of exponential decay ODE: dy/dt = -y.
    
    Args:
        t: Time (unused for autonomous system).
        y: Solution value(s).
        
    Returns:
        Time derivative -y.
    """
    return -y


def euler_integrate(rhs, y0, t0, tf, dt, xp):
    """Integrate ODE using forward Euler method.
    
    Args:
        rhs: Right-hand side function rhs(t, y).
        y0: Initial condition.
        t0: Initial time.
        tf: Final time.
        dt: Time step size.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution at final time tf.
    """
    nsteps = int(round((tf - t0) / dt))
    t = float(t0)
    y = xp.array(y0, dtype=xp.float64, copy=True)
    for _ in range(nsteps):
        y = y + dt * rhs(t, y)
        t += dt
    return y


def rk2_midpoint_integrate(rhs, y0, t0, tf, dt, xp):
    """Integrate ODE using 2nd-order Runge-Kutta (midpoint) method.
    
    Args:
        rhs: Right-hand side function rhs(t, y).
        y0: Initial condition.
        t0: Initial time.
        tf: Final time.
        dt: Time step size.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution at final time tf.
    """
    nsteps = int(round((tf - t0) / dt))
    t = float(t0)
    y = xp.array(y0, dtype=xp.float64, copy=True)
    for _ in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        y = y + dt * k2
        t += dt
    return y


def rk4_integrate(rhs, y0, t0, tf, dt, xp):
    """Integrate ODE using 4th-order Runge-Kutta method.
    
    Args:
        rhs: Right-hand side function rhs(t, y).
        y0: Initial condition.
        t0: Initial time.
        tf: Final time.
        dt: Time step size.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution at final time tf.
    """
    nsteps = int(round((tf - t0) / dt))
    t = float(t0)
    y = xp.array(y0, dtype=xp.float64, copy=True)
    for _ in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
    return y


P1_METHODS: Dict[str, Callable] = {
    "euler": euler_integrate,
    "rk2": rk2_midpoint_integrate,
    "rk4": rk4_integrate,
}


def problem1_validate(outdir: Path) -> None:
    """Validate CPU and GPU results agree for exponential decay problem.
    
    Compares solutions from CPU and GPU backends for all RK methods
    and writes validation report to validation.txt.
    
    Args:
        outdir: Output directory for results.
    """
    backend_cpu = get_backend(prefer_gpu=False)
    backend_gpu = get_backend(prefer_gpu=True)
    lines = []
    if not backend_gpu.has_gpu:
        lines.append("CuPy/GPU unavailable. Validation skipped on GPU; CPU-only fallback used.")
    else:
        for name, method in P1_METHODS.items():
            y_cpu = float(to_cpu(method(rhs_decay, 1.0, 0.0, 1.0, 2.0 ** -8, backend_cpu.xp)))
            y_gpu = float(to_cpu(method(rhs_decay, 1.0, 0.0, 1.0, 2.0 ** -8, backend_gpu.xp)))
            diff = abs(y_cpu - y_gpu)
            lines.append(f"{name}: cpu={y_cpu:.16e}, gpu={y_gpu:.16e}, abs_diff={diff:.3e}")
    write_text(outdir / "validation.txt", "\n".join(lines) + "\n")


def problem1_convergence(outdir: Path) -> None:
    """Analyze convergence of Euler, RK2, and RK4 methods.
    
    Computes global error as function of time step for exponential decay
    problem and fits observed convergence order. Generates convergence plot
    and CSV files with error data.
    
    Args:
        outdir: Output directory for results.
    """
    setup_matplotlib()
    backend = get_backend(prefer_gpu=True)
    ns = np.arange(4, 11)
    dts = 2.0 ** (-ns)
    exact = math.exp(-1.0)

    table_rows = []
    plt.figure()

    ref_consts = {"euler": 1.0, "rk2": 0.2, "rk4": 0.03}

    for name, method in P1_METHODS.items():
        errs = []
        for dt in dts:
            y = float(to_cpu(method(rhs_decay, 1.0, 0.0, 1.0, float(dt), backend.xp)))
            errs.append(abs(y - exact))
        errs = np.asarray(errs)
        slope, intercept = slope_fit(dts, errs)
        table_rows.append([name, slope, math.exp(intercept)])
        save_csv(
            outdir / f"convergence_{name}.csv",
            "dt,error",
            np.column_stack([dts, errs]),
        )
        plt.loglog(dts, errs, "o-", label=f"{name} (fit={slope:.3f})")

    # reference slopes
    plt.loglog(dts, ref_consts["euler"] * dts, "--", label="O(dt)")
    plt.loglog(dts, ref_consts["rk2"] * dts**2, "--", label="O(dt^2)")
    plt.loglog(dts, ref_consts["rk4"] * dts**4, "--", label="O(dt^4)")
    plt.xlabel("dt")
    plt.ylabel("global error at t=1")
    plt.title(f"Problem 1 convergence ({backend.name})")
    plt.legend()
    plt.savefig(outdir / "problem1_error_vs_dt.png")
    plt.close()

    # Save table with method names and numeric values
    methods = [row[0] for row in table_rows]
    numeric_data = np.asarray([row[1:] for row in table_rows], dtype=np.float64)
    
    # Write CSV with method names in first column
    csv_path = outdir / "problem1_orders.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "observed_order", "fit_prefactor"])
        for method, row in zip(methods, numeric_data):
            writer.writerow([method, row[0], row[1]])


def problem1_timing(outdir: Path) -> None:
    """Benchmark CPU and GPU performance for Problem 1 solvers.
    
    Measures wall time and computes GPU speedup for all RK methods
    across varying time steps. Generates timing plots and CSV files.
    
    Args:
        outdir: Output directory for results.
    """
    setup_matplotlib()
    ns = np.arange(4, 11)
    dts = 2.0 ** (-ns)

    backend_cpu = get_backend(prefer_gpu=False)
    backend_gpu = get_backend(prefer_gpu=True)

    summary_rows = []
    plt.figure()

    for name, method in P1_METHODS.items():
        cpu_times = []
        gpu_times = []
        for dt in dts:
            with Timer(backend_cpu) as tcpu:
                _ = method(rhs_decay, 1.0, 0.0, 1.0, float(dt), backend_cpu.xp)
            cpu_times.append(tcpu.dt)

            if backend_gpu.has_gpu:
                with Timer(backend_gpu) as tgpu: #with looks for __enter__ and __exit__ methods in Timer class, which handle GPU synchronization
                    _ = method(rhs_decay, 1.0, 0.0, 1.0, float(dt), backend_gpu.xp)
                gpu_times.append(tgpu.dt)
            else:
                gpu_times.append(np.nan)

        cpu_times = np.asarray(cpu_times)
        gpu_times = np.asarray(gpu_times)

        save_csv(
            outdir / f"timing_{name}.csv",
            "dt,cpu_time_s,gpu_time_s,speedup_cpu_over_gpu",
            np.column_stack(
                [dts, cpu_times, gpu_times, cpu_times / gpu_times]
            ),
        )

        plt.loglog(dts, cpu_times, "o-", label=f"{name} CPU")
        if backend_gpu.has_gpu:
            plt.loglog(dts, gpu_times, "s--", label=f"{name} GPU")

        idx = -1  # smallest dt
        speedup = cpu_times[idx] / gpu_times[idx] if backend_gpu.has_gpu else np.nan
        summary_rows.append([name, cpu_times[idx], gpu_times[idx], speedup])

    plt.xlabel("dt")
    plt.ylabel("wall time (s)")
    plt.title("Problem 1 timing")
    plt.legend(ncol=2)
    plt.savefig(outdir / "problem1_timing_vs_dt.png")
    plt.close()

    save_csv(
        outdir / "problem1_timing_summary_smallest_dt.csv",
        "method,cpu_time,gpu_time,speedup_cpu_over_gpu",
        np.asarray(summary_rows, dtype=object),
    )


def run_problem1(outdir: Path) -> None:
    """Run all analyses for Problem 1 (explicit RK methods).
    
    Performs validation, convergence analysis, and timing benchmarks.
    
    Args:
        outdir: Output directory for all Problem 1 results.
    """
    ensure_dir(outdir)
    problem1_validate(outdir)
    problem1_convergence(outdir)
    problem1_timing(outdir)


# ----------------------------
# Problem 2: stability regions
# ----------------------------

def R_euler(z):
    """Stability function for Euler method: R(z) = 1 + z.
    
    Args:
        z: Complex argument.
        
    Returns:
        Stability function value.
    """
    return 1.0 + z


def R_rk2(z):
    """Stability function for RK2 (midpoint) method: R(z) = 1 + z + z²/2.
    
    Args:
        z: Complex argument.
        
    Returns:
        Stability function value.
    """
    return 1.0 + z + 0.5 * z**2


def R_rk4(z):
    """Stability function for RK4 method: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24.
    
    Args:
        z: Complex argument.
        
    Returns:
        Stability function value.
    """
    return 1.0 + z + 0.5 * z**2 + (1.0 / 6.0) * z**3 + (1.0 / 24.0) * z**4


P2_R = {
    "euler": R_euler,
    "rk2": R_rk2,
    "rk4": R_rk4,
}


def estimate_negative_real_extent(fun, x_left=-5.0, x_right=0.0, n=20001):
    """Estimate the extent of stability region on negative real axis.
    
    Finds the leftmost point where |fun(x)| <= 1 on the negative real axis.
    
    Args:
        fun: Stability function to evaluate.
        x_left: Left boundary of search interval.
        x_right: Right boundary of search interval.
        n: Number of sample points.
        
    Returns:
        Leftmost point where |fun(x)| <= 1, or NaN if not found.
    """
    x = np.linspace(x_left, x_right, n)
    vals = np.abs(fun(x))
    mask = vals <= 1.0 + 1e-12
    if not np.any(mask):
        return np.nan
    return float(x[mask][0])


def run_problem2(outdir: Path, grid_n: int = 801) -> None:
    """Analyze stability regions of RK methods in the complex plane.
    
    Plots contours where |R(z)| = 1 for Euler, RK2, and RK4 methods
    and estimates stability region extent on negative real axis.
    
    Args:
        outdir: Output directory for results.
        grid_n: Grid resolution for stability region computation.
    """
    ensure_dir(outdir)
    setup_matplotlib()

    x = np.linspace(-5.0, 5.0, grid_n)
    y = np.linspace(-5.0, 5.0, grid_n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    plt.figure(figsize=(7.5, 6.0))
    extents = []

    for name, fun in P2_R.items():
        absR = np.abs(fun(Z))
        np.save(outdir / f"{name}_absR.npy", absR)
        plt.contour(X, Y, absR, levels=[1.0], linewidths=2.0)
        xmin = estimate_negative_real_extent(fun)
        extents.append([name, xmin])

    # overlay legends using proxies
    import matplotlib.lines as mlines
    proxies = [
        mlines.Line2D([], [], linewidth=2, label="Euler"),
        mlines.Line2D([], [], linewidth=2, label="RK2 midpoint"),
        mlines.Line2D([], [], linewidth=2, label="RK4"),
    ]
    plt.legend(handles=proxies, loc="upper right")
    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    plt.axvline(0.0, color="k", linewidth=0.8, alpha=0.5)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Problem 2 stability boundaries |R(z)| = 1")
    plt.savefig(outdir / "problem2_stability_regions.png")
    plt.close()

    save_csv(
        outdir / "problem2_negative_real_extent.csv",
        "method,estimated_xmin",
        np.asarray(extents, dtype=object),
    )

    theory = (
        "Stability functions used:\n"
        "Euler: R(z) = 1 + z\n"
        "RK2(midpoint): R(z) = 1 + z + z^2/2\n"
        "RK4: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24\n\n"
        "On the negative real axis the approximate stability endpoints are:\n"
        f"Euler: {estimate_negative_real_extent(R_euler):.6f} (theory: -2)\n"
        f"RK2:   {estimate_negative_real_extent(R_rk2):.6f} (theory: -2)\n"
        f"RK4:   {estimate_negative_real_extent(R_rk4):.6f} (theory: about -2.7853)\n\n"
        "Explicit RK stability functions are polynomials. As x -> -infinity, |R(x)| -> infinity,\n"
        "so the entire left half-plane cannot lie inside the stability region. Hence explicit methods cannot be A-stable.\n"
    )
    write_text(outdir / "problem2_notes.txt", theory)


# ----------------------------
# Problem 3: batched logistic RK4
# ----------------------------

def logistic_rhs(t, y, r=2.0):
    """Right-hand side of logistic growth ODE: dy/dt = r*y*(1-y).
    
    Args:
        t: Time (unused for autonomous system).
        y: Solution value(s).
        r: Growth rate parameter.
        
    Returns:
        Time derivative r*y*(1-y).
    """
    return r * y * (1.0 - y)


def batched_rk4_logistic(y0, dt, tf, xp, r=2.0):
    """Integrate batched logistic ODE trajectories using RK4 method.
    
    Solves multiple independent logistic growth trajectories simultaneously,
    leveraging GPU parallelization for better performance with large batches.
    
    Args:
        y0: Initial conditions (batch of starting values).
        dt: Time step size.
        tf: Final time.
        xp: Numerical backend (numpy or cupy).
        r: Growth rate parameter.
        
    Returns:
        Solution values at final time tf for all trajectories.
    """
    y = xp.array(y0, copy=True)
    nsteps = int(round(tf / dt))
    t = 0.0
    for _ in range(nsteps):
        k1 = logistic_rhs(t, y, r=r)
        k2 = logistic_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, r=r)
        k3 = logistic_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, r=r)
        k4 = logistic_rhs(t + dt, y + dt * k3, r=r)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return y


def run_problem3(outdir: Path, Ns: Iterable[int], dt: float = 1e-3, tf: float = 10.0) -> None:
    """Benchmark batched GPU computation with varying problem sizes.
    
    Solves N independent logistic growth trajectories on CPU and GPU,
    measuring wall time and computing GPU speedup as function of batch size.
    
    Args:
        outdir: Output directory for results.
        Ns: List of batch sizes to benchmark.
        dt: Time step size.
        tf: Final integration time.
    """
    ensure_dir(outdir)
    setup_matplotlib()

    backend_cpu = get_backend(prefer_gpu=False)
    backend_gpu = get_backend(prefer_gpu=True)

    rows = []

    for N in Ns:
        y0_cpu = np.linspace(0.01, 0.99, int(N), dtype=np.float64)

        with Timer(backend_cpu) as tcpu:
            _ = batched_rk4_logistic(y0_cpu, dt=dt, tf=tf, xp=backend_cpu.xp)

        if backend_gpu.has_gpu:
            y0_gpu = backend_gpu.xp.linspace(0.01, 0.99, int(N), dtype=backend_gpu.xp.float64)
            with Timer(backend_gpu) as tgpu:
                _ = batched_rk4_logistic(y0_gpu, dt=dt, tf=tf, xp=backend_gpu.xp)
            gpu_t = tgpu.dt
        else:
            gpu_t = np.nan

        steps = int(round(tf / dt))
        throughput_cpu = N * steps / tcpu.dt
        throughput_gpu = N * steps / gpu_t if backend_gpu.has_gpu else np.nan
        speedup = tcpu.dt / gpu_t if backend_gpu.has_gpu else np.nan
        rows.append([N, tcpu.dt, gpu_t, speedup, throughput_cpu, throughput_gpu])

    arr = np.asarray(rows, dtype=object)
    save_csv(
        outdir / "problem3_benchmark.csv",
        "N,cpu_time_s,gpu_time_s,speedup_cpu_over_gpu,throughput_cpu,throughput_gpu",
        arr,
    )

    arr_num = np.array(rows, dtype=float)
    N = arr_num[:, 0]
    cpu_t = arr_num[:, 1]
    gpu_t = arr_num[:, 2]
    speedup = arr_num[:, 3]
    throughput_cpu = arr_num[:, 4]
    throughput_gpu = arr_num[:, 5]

    plt.figure()
    plt.loglog(N, cpu_t, "o-", label="CPU")
    if backend_gpu.has_gpu:
        plt.loglog(N, gpu_t, "s--", label="GPU")
    plt.xlabel("N trajectories")
    plt.ylabel("wall time (s)")
    plt.title("Problem 3 runtime")
    plt.legend()
    plt.savefig(outdir / "problem3_runtime.png")
    plt.close()

    if backend_gpu.has_gpu:
        plt.figure()
        plt.semilogx(N, speedup, "o-")
        plt.xlabel("N trajectories")
        plt.ylabel("speedup = tCPU / tGPU")
        plt.title("Problem 3 GPU speedup")
        plt.savefig(outdir / "problem3_speedup.png")
        plt.close()

    plt.figure()
    plt.loglog(N, throughput_cpu, "o-", label="CPU")
    if backend_gpu.has_gpu:
        plt.loglog(N, throughput_gpu, "s--", label="GPU")
    plt.xlabel("N trajectories")
    plt.ylabel("trajectory-steps / second")
    plt.title("Problem 3 throughput")
    plt.legend()
    plt.savefig(outdir / "problem3_throughput.png")
    plt.close()

    notes = [
        f"Backend GPU available: {backend_gpu.has_gpu}",
        "This problem is vectorized over independent trajectories.",
        "GPU wins only once N is large enough that kernel-launch and transfer overheads are amortized.",
        "For small N, CPU often wins because the work per launch is too small.",
    ]
    write_text(outdir / "problem3_notes.txt", "\n".join(notes) + "\n")


# ----------------------------
# Problem 4: stiff diagonal system
# ----------------------------

def explicit_euler_linear_decay(y0, alpha, dt, tf, xp):
    """Integrate diagonal linear system y' = -alpha*y using explicit Euler.
    
    Args:
        y0: Initial conditions.
        alpha: Decay rates (diagonal elements).
        dt: Time step size.
        tf: Final time.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution at final time tf.
    """
    y = xp.array(y0, copy=True)
    alpha = xp.array(alpha, copy=False)
    nsteps = int(round(tf / dt))
    for _ in range(nsteps):
        y = y - dt * alpha * y
    return y


def tr_step_linear(y, alpha, dt, xp):
    """One step of trapezoidal (TR) method for y' = -alpha*y.
    
    Args:
        y: Current solution values.
        alpha: Decay rates (diagonal elements).
        dt: Time step size.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution after one time step.
    """
    num = 1.0 - 0.5 * dt * alpha
    den = 1.0 + 0.5 * dt * alpha
    return (num / den) * y


def tr_integrate_linear(y0, alpha, dt, tf, xp):
    """Integrate diagonal linear system y' = -alpha*y using trapezoidal method.
    
    Args:
        y0: Initial conditions.
        alpha: Decay rates (diagonal elements).
        dt: Time step size.
        tf: Final time.
        xp: Numerical backend (numpy or cupy).
        
    Returns:
        Solution at final time tf.
    """
    y = xp.array(y0, copy=True)
    alpha = xp.array(alpha, copy=False)
    nsteps = int(round(tf / dt))
    for _ in range(nsteps):
        y = tr_step_linear(y, alpha, dt, xp)
    return y


def trbdf2_step_linear(y, alpha, dt, xp, gamma=2.0 - math.sqrt(2.0)):
    """One step of TR-BDF2 (composite) method for y' = -alpha*y.
    
    Two-stage method combining trapezoidal and BDF2 for L-stability.
    
    Args:
        y: Current solution values.
        alpha: Decay rates (diagonal elements).
        dt: Time step size.
        xp: Numerical backend (numpy or cupy).
        gamma: Stage parameter (default: 2 - sqrt(2)).
        
    Returns:
        Solution after one time step.
    """
    alpha = xp.array(alpha, copy=False)
    # Stage 1: TR over gamma*dt
    y_gamma = ((1.0 - 0.5 * gamma * dt * alpha) / (1.0 + 0.5 * gamma * dt * alpha)) * y
    # Stage 2: variable-step BDF2 from t_n, t_n+gamma*dt, t_n+dt
    a = 1.0 / (gamma * (2.0 - gamma))
    b = ((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))
    den = 1.0 + ((1.0 - gamma) / (2.0 - gamma)) * dt * alpha
    y_next = (a * y_gamma - b * y) / den
    return y_next


def trbdf2_integrate_linear(y0, alpha, dt, tf, xp, gamma=2.0 - math.sqrt(2.0)):
    """Integrate diagonal linear system y' = -alpha*y using TR-BDF2 method.
    
    Args:
        y0: Initial conditions.
        alpha: Decay rates (diagonal elements).
        dt: Time step size.
        tf: Final time.
        xp: Numerical backend (numpy or cupy).
        gamma: Stage parameter (default: 2 - sqrt(2)).
        
    Returns:
        Solution at final time tf.
    """
    y = xp.array(y0, copy=True)
    alpha = xp.array(alpha, copy=False)
    nsteps = int(round(tf / dt))
    for _ in range(nsteps):
        y = trbdf2_step_linear(y, alpha, dt, xp, gamma=gamma)
    return y


def run_problem4(outdir: Path, d: int = 10**6, dt_small: float = 1e-6, dt_large: float = 1e-2, tf: float = 1.0) -> None:
    """Analyze stiffness and stability of implicit methods.
    
    Demonstrates explicit Euler instability for stiff problems and compares
    damping properties of trapezoidal (A-stable) and TR-BDF2 (L-stable) methods.
    Benchmarks TR-BDF2 on CPU and GPU.
    
    Args:
        outdir: Output directory for results.
        d: Dimension of diagonal system.
        dt_small: Small time step for accurate integration.
        dt_large: Large time step demonstrating stiffness.
        tf: Final integration time.
    """
    ensure_dir(outdir)
    setup_matplotlib()

    backend_cpu = get_backend(prefer_gpu=False)
    backend_gpu = get_backend(prefer_gpu=True)

    alpha_cpu = np.logspace(0, 6, int(d), dtype=np.float32)
    y0_cpu = np.ones(int(d), dtype=np.float32)
    exact_cpu = np.exp(-alpha_cpu * tf)

    # Accuracy and stability summary
    y_tr_small = to_cpu(tr_integrate_linear(y0_cpu, alpha_cpu, dt_small, tf, backend_cpu.xp))
    y_trb_small = to_cpu(trbdf2_integrate_linear(y0_cpu, alpha_cpu, dt_small, tf, backend_cpu.xp))
    y_tr_large = to_cpu(tr_integrate_linear(y0_cpu, alpha_cpu, dt_large, tf, backend_cpu.xp))
    y_trb_large = to_cpu(trbdf2_integrate_linear(y0_cpu, alpha_cpu, dt_large, tf, backend_cpu.xp))

    def max_rel_err(y):
        denom = np.maximum(np.abs(exact_cpu), 1e-30)
        return float(np.max(np.abs(y - exact_cpu) / denom))

    # explicit instability demonstration
    alpha_demo = np.array([1.0, 1e3, 1e6], dtype=np.float64)
    y0_demo = np.ones_like(alpha_demo)
    dt_explicit_good = 1e-6
    dt_explicit_bad = 5e-6  # unstable for alpha=1e6 because |1-dt*alpha| > 1
    times_good = np.linspace(0.0, tf, int(round(tf / dt_explicit_good)) + 1)
    times_bad = np.linspace(0.0, tf, int(round(tf / dt_explicit_bad)) + 1)

    def explicit_history(alpha_vals, dt):
        y = np.ones_like(alpha_vals, dtype=np.float64)
        hist = [y.copy()]
        nsteps = int(round(tf / dt))
        for _ in range(nsteps):
            y = y - dt * alpha_vals * y
            hist.append(y.copy())
        return np.asarray(hist)

    good_hist = explicit_history(alpha_demo, dt_explicit_good)
    bad_hist = explicit_history(alpha_demo, dt_explicit_bad)

    labels = ["alpha=1", "alpha=1e3", "alpha=1e6"]
    plt.figure()
    for j, lab in enumerate(labels):
        plt.semilogy(times_good, np.abs(good_hist[:, j]), label=f"{lab}, stable dt")
        plt.semilogy(times_bad, np.abs(bad_hist[:, j]), "--", label=f"{lab}, unstable dt")
    plt.xlabel("t")
    plt.ylabel("|y(t)|")
    plt.title("Problem 4 explicit Euler stiffness demo")
    plt.legend(ncol=2, fontsize=9)
    plt.savefig(outdir / "problem4_explicit_stiffness_demo.png")
    plt.close()

    # TR vs TRBDF2 damping demo
    alpha_select = np.array([1.0, 1e3, 1e6], dtype=np.float64)
    y_tr = np.ones_like(alpha_select)
    y_tb = np.ones_like(alpha_select)
    times = [0.0]
    hist_tr = [y_tr.copy()]
    hist_tb = [y_tb.copy()]
    nsteps_large = int(round(tf / dt_large))
    for k in range(nsteps_large):
        y_tr = to_cpu(tr_step_linear(y_tr, alpha_select, dt_large, np))
        y_tb = to_cpu(trbdf2_step_linear(y_tb, alpha_select, dt_large, np))
        hist_tr.append(y_tr.copy())
        hist_tb.append(y_tb.copy())
        times.append((k + 1) * dt_large)
    hist_tr = np.asarray(hist_tr)
    hist_tb = np.asarray(hist_tb)
    times = np.asarray(times)

    plt.figure()
    for j, lab in enumerate(labels):
        plt.semilogy(times, np.abs(hist_tr[:, j]), "o-", markevery=max(1, len(times)//10), label=f"TR {lab}")
        plt.semilogy(times, np.abs(hist_tb[:, j]), "--", label=f"TRBDF2 {lab}")
    plt.xlabel("t")
    plt.ylabel("|y(t)|")
    plt.title("Problem 4 damping: TR vs TRBDF2")
    plt.legend(ncol=2, fontsize=9)
    plt.savefig(outdir / "problem4_tr_vs_trbdf2_damping.png")
    plt.close()

    # GPU benchmark for TRBDF2
    if backend_gpu.has_gpu:
        alpha_gpu = backend_gpu.xp.logspace(0, 6, int(d), dtype=backend_gpu.xp.float32)
        y0_gpu = backend_gpu.xp.ones(int(d), dtype=backend_gpu.xp.float32)
        with Timer(backend_cpu) as tcpu:
            _ = trbdf2_integrate_linear(y0_cpu, alpha_cpu, dt_large, tf, backend_cpu.xp)
        with Timer(backend_gpu) as tgpu:
            _ = trbdf2_integrate_linear(y0_gpu, alpha_gpu, dt_large, tf, backend_gpu.xp)
        gpu_summary = f"CPU TRBDF2 time: {tcpu.dt:.6f} s\nGPU TRBDF2 time: {tgpu.dt:.6f} s\nSpeedup: {tcpu.dt / tgpu.dt:.3f}\n"
    else:
        gpu_summary = "CuPy/GPU unavailable. Benchmark skipped on GPU.\n"

    summary_arr = np.asarray(
        [
            ["TR", dt_small, max_rel_err(y_tr_small)],
            ["TRBDF2", dt_small, max_rel_err(y_trb_small)],
            ["TR", dt_large, max_rel_err(y_tr_large)],
            ["TRBDF2", dt_large, max_rel_err(y_trb_large)],
        ],
        dtype=object,
    )
    save_csv(
        outdir / "problem4_accuracy_summary.csv",
        "method,dt,max_relative_error_at_t1",
        summary_arr,
    )

    notes = (
        "TR step factor for y'=-alpha y: (1 - alpha dt / 2) / (1 + alpha dt / 2)\n"
        "TRBDF2 gamma used: 2 - sqrt(2)\n"
        + gpu_summary +
        "\nTR is A-stable but not L-stable, so very stiff modes are not strongly damped for large dt.\n"
        "TRBDF2 is L-stable, so large-alpha modes are strongly damped.\n"
    )
    write_text(outdir / "problem4_notes.txt", notes)


# ----------------------------
# Orchestration / CLI
# ----------------------------

def run_all(outdir: Path) -> None:
    """Run all four problems and generate all results.
    
    Args:
        outdir: Root output directory for all results.
    """
    run_problem1(outdir / "problem1")
    run_problem2(outdir / "problem2")
    run_problem3(outdir / "problem3", Ns=[10**3, 10**4, 10**5, 10**6])
    run_problem4(outdir / "problem4")


def parse_args():
    """Parse command-line arguments for running specific problems.
    
    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="HW-6 CuPy ODE IVP solution code")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("problem1", help="Run Problem 1")
    p1.add_argument("--outdir", type=Path, default=Path("results/problem1"))

    p2 = sub.add_parser("problem2", help="Run Problem 2")
    p2.add_argument("--outdir", type=Path, default=Path("results/problem2"))
    p2.add_argument("--grid-n", type=int, default=801)

    p3 = sub.add_parser("problem3", help="Run Problem 3")
    p3.add_argument("--outdir", type=Path, default=Path("results/problem3"))
    p3.add_argument("--Ns", type=int, nargs="+", default=[10**3, 10**4, 10**5, 10**6])
    p3.add_argument("--dt", type=float, default=1e-3)
    p3.add_argument("--tf", type=float, default=10.0)

    p4 = sub.add_parser("problem4", help="Run Problem 4 (grad)")
    p4.add_argument("--outdir", type=Path, default=Path("results/problem4"))
    p4.add_argument("--d", type=int, default=10**6)
    p4.add_argument("--dt-small", type=float, default=1e-6)
    p4.add_argument("--dt-large", type=float, default=1e-2)
    p4.add_argument("--tf", type=float, default=1.0)

    pall = sub.add_parser("all", help="Run all problems")
    pall.add_argument("--outdir", type=Path, default=Path("results"))

    return parser.parse_args()


def main():
    """Main entry point for command-line execution.
    
    Parses arguments and runs the requested problem(s).
    """
    args = parse_args()
    if args.command == "problem1":
        run_problem1(args.outdir)
    elif args.command == "problem2":
        run_problem2(args.outdir, grid_n=args.grid_n)
    elif args.command == "problem3":
        run_problem3(args.outdir, Ns=args.Ns, dt=args.dt, tf=args.tf)
    elif args.command == "problem4":
        run_problem4(args.outdir, d=args.d, dt_small=args.dt_small, dt_large=args.dt_large, tf=args.tf)
    elif args.command == "all":
        run_all(args.outdir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
