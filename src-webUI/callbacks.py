"""
Calculator callback functions for web UI.
Handles data conversion and computation logic.
"""

import base64
import os
import sys
from io import StringIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.bearing_calculations import (
    solve_eccentricity,
    solve_K_and_C_AlBender,
    solve_K_and_C_Friswell,
)
from src.compressible_flow import compressible_flow
from src.modal_analysis import Bearing, RotorSystem
from src.pressure_distribution import pressure_distribution


def calculate_friswell_K_and_C(data):
    """
    Calculate stiffness and damping matrices using Friswell method.

    Parameters
    ----------
    data : dict
        Input data containing:
        - omega: Shaft speed in RPM
        - D: Diameter in mm
        - L: Length in mm
        - c: Radial clearance in mm
        - eta: Viscosity in Pa·s
        - f: Static load in N

    Returns
    -------
    dict
        Result containing K matrix, C matrix, and computed epsilon
    """
    # Unit conversions
    omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
    D = data["D"] / 1000  # mm to m
    c = data["c"] / 1000  # mm to m
    L = data["L"] / 1000  # mm to m
    eta = data["eta"]  # Pa·s
    f = data["f"]  # N

    # Compute eccentricity and matrices
    epsilon = solve_eccentricity(D, omega, eta, L, f, c)
    K, C = solve_K_and_C_Friswell(omega, f, c, epsilon)

    return {"K": K.tolist(), "C": C.tolist(), "epsilon": epsilon}


def plot_friswell_results(data):
    """
    Generate plot data for Friswell K and C across a parameter sweep.

    Parameters
    ----------
    data : dict
        Input data containing:
        - sweep_param: Parameter to sweep ('omega', 'f', 'D', 'L', 'c', or 'eta')
        - sweep_min: Minimum value for sweep
        - sweep_max: Maximum value for sweep
        - sweep_points: Number of points in sweep
        - omega: Shaft speed in RPM (fixed if not sweep param)
        - D: Diameter in mm (fixed if not sweep param)
        - L: Length in mm (fixed if not sweep param)
        - c: Radial clearance in mm (fixed if not sweep param)
        - eta: Viscosity in Pa·s (fixed if not sweep param)
        - f: Static load in N (fixed if not sweep param)

    Returns
    -------
    dict
        Result containing sweep values and K/C coefficients
    """
    # Extract parameters
    sweep_param = data.get("sweep_param", "omega")
    sweep_min = data.get("sweep_min", 500)
    sweep_max = data.get("sweep_max", 3000)
    sweep_points = data.get("sweep_points", 50)

    # Generate sweep values
    sweep_vals = np.linspace(sweep_min, sweep_max, sweep_points)

    # Initialize result arrays for all 8 coefficients
    K_xx, K_xy, K_yx, K_yy = [], [], [], []
    C_xx, C_xy, C_yx, C_yy = [], [], [], []

    for val in sweep_vals:
        # Set sweep parameter and get fixed parameters with unit conversions
        if sweep_param == "omega":
            omega = val * (2 * np.pi) / 60  # RPM to rad/s
            f = data["f"]  # N
            D = data["D"] / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            eta = data["eta"]  # Pa·s
        elif sweep_param == "f":
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            f = val  # N
            D = data["D"] / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            eta = data["eta"]  # Pa·s
        elif sweep_param == "D":
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            f = data["f"]  # N
            D = val / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            eta = data["eta"]  # Pa·s
        elif sweep_param == "L":
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            f = data["f"]  # N
            D = data["D"] / 1000  # mm to m
            L = val / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            eta = data["eta"]  # Pa·s
        elif sweep_param == "c":
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            f = data["f"]  # N
            D = data["D"] / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            c = val / 1000  # mm to m
            eta = data["eta"]  # Pa·s
        elif sweep_param == "eta":
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            f = data["f"]  # N
            D = data["D"] / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            eta = val  # Pa·s

        # Compute eccentricity and matrices
        epsilon = solve_eccentricity(D, omega, eta, L, f, c)
        K, C = solve_K_and_C_Friswell(omega, f, c, epsilon)

        # Store coefficients
        K_xx.append(K[0, 0])
        K_xy.append(K[0, 1])
        K_yx.append(K[1, 0])
        K_yy.append(K[1, 1])
        C_xx.append(C[0, 0])
        C_xy.append(C[0, 1])
        C_yx.append(C[1, 0])
        C_yy.append(C[1, 1])

    # Determine x-axis label based on sweep parameter
    x_labels = {
        "omega": "ω [RPM]",
        "f": "f [N]",
        "D": "D [mm]",
        "L": "L [mm]",
        "c": "c [mm]",
        "eta": "η [Pa·s]",
    }
    x_label = x_labels.get(sweep_param, "Value")

    return {
        "sweep_vals": sweep_vals.tolist(),
        "sweep_param": sweep_param,
        "x_label": x_label,
        "K_xx": K_xx,
        "K_xy": K_xy,
        "K_yx": K_yx,
        "K_yy": K_yy,
        "C_xx": C_xx,
        "C_xy": C_xy,
        "C_yx": C_yx,
        "C_yy": C_yy,
    }


def calculate_albender_K_and_C(data):
    """
    Calculate stiffness and damping matrices using Al-Bender method.

    Parameters
    ----------
    data : dict
        Input data containing:
        - lambda_vals: List of frequency parameters (uses first value)
        - sigma_min: Minimum sigma value (log scale)
        - sigma_max: Maximum sigma value (log scale)
        - sigma_points: Number of sigma points
        - L_over_D: Length to diameter ratio

    Returns
    -------
    dict
        Result containing K matrix, C matrix, and sigma values array
    """
    # Extract parameters with defaults
    lambda_val = data.get("lambda_vals", [1.0])[0]
    sigma_min = data.get("sigma_min", -1)
    sigma_max = data.get("sigma_max", 3)
    sigma_points = data.get("sigma_points", 10000)
    L_over_D = data.get("L_over_D", 1.0)

    # Generate sigma values
    sigma_vals = np.logspace(sigma_min, sigma_max, sigma_points)

    # Compute matrices
    K, C = solve_K_and_C_AlBender(
        lambda_val,
        sigma_vals,
        L_over_D,
    )

    return {
        "K": K.tolist(),
        "C": C.tolist(),
        "sigma_vals": sigma_vals.tolist(),
    }


def plot_albender_results(data):
    """
    Generate raw data for Al-Bender results with multiple lambda values.

    Parameters
    ----------
    data : dict
        Parameters including:
        - lambda_vals: list of frequency parameters
        - sigma_min, sigma_max, sigma_points: Sommerfeld number range
        - L_over_D: Length to diameter ratio

    Returns
    -------
    dict
        Raw data for plotting
    """
    # Extract parameters with defaults
    lambda_vals = data.get("lambda_vals", [1.0])
    sigma_min = data.get("sigma_min", -1)
    sigma_max = data.get("sigma_max", 3)
    sigma_points = data.get("sigma_points", 10000)
    L_over_D = data.get("L_over_D", 1.0)

    # Generate sigma values
    sigma_vals = np.logspace(sigma_min, sigma_max, sigma_points)

    # Calculate K and C for all lambda values and store in DataFrames
    dfs = []
    for lambda_val in lambda_vals:
        K, C = solve_K_and_C_AlBender(
            lambda_val,
            sigma_vals,
            L_over_D,
        )

        # Extract coefficients - K and C are (2,2) arrays where each element is an array
        K_xx = np.array(K[0, 0]).flatten()
        K_xy = np.array(K[0, 1]).flatten()
        C_xx = np.array(C[0, 0]).flatten()
        C_xy = np.array(C[0, 1]).flatten()

        # Create DataFrame for this lambda value
        df = pd.DataFrame(
            {
                "sigma": sigma_vals,
                "K_xx": K_xx,
                "K_xy": K_xy,
                "C_xx": C_xx,
                "C_xy": C_xy,
                "lambda": lambda_val,
            }
        )
        dfs.append(df)

    # Combine all DataFrames
    results_df = pd.concat(dfs, ignore_index=True)

    # Return raw data, including all coefficients
    return {
        "sigma": results_df["sigma"].tolist(),
        "K_xx": [
            results_df[results_df["lambda"] == lambda_val]["K_xx"].tolist()
            for lambda_val in lambda_vals
        ],
        "K_xy": [
            results_df[results_df["lambda"] == lambda_val]["K_xy"].tolist()
            for lambda_val in lambda_vals
        ],
        "C_xx": [
            results_df[results_df["lambda"] == lambda_val]["C_xx"].tolist()
            for lambda_val in lambda_vals
        ],
        "C_xy": [
            results_df[results_df["lambda"] == lambda_val]["C_xy"].tolist()
            for lambda_val in lambda_vals
        ],
        "lambdas": lambda_vals,
    }


def calculate_pressure_distribution(data):
    """
    Calculate pressure distribution for tilted pad thrust bearing.

    Parameters
    ----------
    data : dict
        Input parameters for pressure distribution calculation

    Returns
    -------
    dict
        Result containing image path and statistics
    """

    # Extract parameters
    mu = data.get("mu", 0.04)
    omega = data.get("omega", 2 * np.pi)
    r_in = data.get("r_in", 0.05)
    r_out = data.get("r_out", 0.10)
    n_sector = data.get("n_sector", 6)
    Delta_theta = data.get("Delta_theta", np.pi / 4)
    hL = data.get("hL", 50e-6)
    hT = data.get("hT", 10e-6)

    # Create a temporary file path
    import tempfile

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "pressure_plot.png")

    # Capture stdout to get statistics
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Call the pressure distribution function
        pressure_distribution(
            mu=mu,
            omega=omega,
            r_in=r_in,
            r_out=r_out,
            n_sector=n_sector,
            Delta_theta=Delta_theta,
            hL=hL,
            hT=hT,
            plot_title=True,
            out_figure=output_path,
            savefig=True,
        )

        # Get the captured output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Read the image and convert to base64
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()

        # Extract statistics from output
        stats = {}
        for line in output.split("\n"):
            if "Peak pressure" in line:
                stats["peak_pressure"] = line.split(":")[1].strip()
            elif "Mean pressure" in line:
                stats["mean_pressure"] = line.split(":")[1].strip()

        return {
            "image": f"data:image/png;base64,{img_data}",
            "stats": stats,
        }

    except Exception as e:
        sys.stdout = old_stdout
        raise e
    finally:
        plt.close("all")


def calculate_compressible_flow(data):
    """
    Calculate pressure distribution for compressible flow.

    Parameters
    ----------
    data : dict
        Input data containing:
        - mu: Dynamic viscosity in Pa·s
        - omega: Angular velocity in rad/s
        - r_in: Inner radius in m
        - r_out: Outer radius in m
        - n_sector: Number of sectors
        - Delta_theta: Sector angle in rad
        - hL: Leading edge film thickness in m
        - hT: Trailing edge film thickness in m
        - rho_a: Reference density in kg/m³
        - p_a: Ambient pressure in Pa
        - beta: Compressibility parameter in Pa
        - max_iter: Maximum iterations
        - tol: Convergence tolerance
        - relax: Relaxation factor

    Returns
    -------
    dict
        Result containing base64 encoded plot image and statistics
    """
    # Extract parameters
    mu = data.get("mu", 0.001)
    omega = data.get("omega", 157.08)
    r_in = data.get("r_in", 0.030)
    r_out = data.get("r_out", 0.050)
    n_sector = data.get("n_sector", 12)
    Delta_theta = data.get("Delta_theta", 26 * np.pi / 180)
    hL = data.get("hL", 50e-6)
    hT = data.get("hT", 10e-6)
    rho_a = data.get("rho_a", 1.2)
    p_a = data.get("p_a", 101325)
    beta = data.get("beta", 1e6)
    max_iter = data.get("max_iter", 500)
    tol = data.get("tol", 1e-6)
    relax = data.get("relax", 0.4)

    # Create a temporary file path
    import tempfile

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "compressible_flow_plot.png")

    # Capture stdout to get statistics
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Call the compressible flow function
        compressible_flow(
            mu=mu,
            omega=omega,
            r_in=r_in,
            r_out=r_out,
            n_sector=n_sector,
            Delta_theta=Delta_theta,
            hL=hL,
            hT=hT,
            rho_a=rho_a,
            p_a=p_a,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            relax=relax,
            plot_title=True,
            out_figure=output_path,
            savefig=True,
        )

        # Get the captured output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Read the image and convert to base64
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()

        # Extract statistics from output
        stats = {}
        for line in output.split("\n"):
            if "Peak pressure" in line:
                stats["peak_pressure"] = line.split(":")[1].strip()
            elif "Mean pressure" in line:
                stats["mean_pressure"] = line.split(":")[1].strip()
            elif "Iterations" in line or "iterations" in line:
                stats["iterations"] = (
                    line.split(":")[1].strip() if ":" in line else line.strip()
                )

        return {
            "image": f"data:image/png;base64,{img_data}",
            "stats": stats,
        }

    except Exception as e:
        sys.stdout = old_stdout
        raise e
    finally:
        plt.close("all")


def calculate_campbell(data):
    """
    Calculate Campbell diagram data for rotor system.

    Parameters
    ----------
    data : dict
        Input data containing:
        - bearing1_kxx: Stiffness in MN/m
        - bearing1_kyy: Stiffness in MN/m
        - bearing1_cxx: Damping in kNs/m
        - bearing1_cyy: Damping in kNs/m
        - bearing2_kxx: Stiffness in MN/m
        - bearing2_kyy: Stiffness in MN/m
        - bearing2_cxx: Damping in kNs/m
        - bearing2_cyy: Damping in kNs/m
        - speed_min: Minimum speed in RPM
        - speed_max: Maximum speed in RPM
        - speed_points: Number of points

    Returns
    -------
    dict
        Result containing speeds and eigenvalue frequencies for D3 plotting
    """
    # Unit conversions: MN/m to N/m, kNs/m to Ns/m
    bearing1_kxx = data.get("bearing1_kxx", 0.2) * 1e6  # MN/m to N/m
    bearing1_kyy = data.get("bearing1_kyy", 0.4) * 1e6
    bearing1_cxx = data.get("bearing1_cxx", 0.0) * 1e3  # kNs/m to Ns/m
    bearing1_cyy = data.get("bearing1_cyy", 0.0) * 1e3
    bearing2_kxx = data.get("bearing2_kxx", 0.2) * 1e6
    bearing2_kyy = data.get("bearing2_kyy", 0.4) * 1e6
    bearing2_cxx = data.get("bearing2_cxx", 0.0) * 1e3
    bearing2_cyy = data.get("bearing2_cyy", 0.0) * 1e3

    speed_min = data.get("speed_min", 0)
    speed_max = data.get("speed_max", 3000)
    speed_points = data.get("speed_points", 100)

    # Create bearings at fixed nodes 0 and 4 (Friswell example 6.8.1)
    bearing_1 = Bearing(
        node=0,
        kxx=bearing1_kxx,
        kyy=bearing1_kyy,
        cxx=bearing1_cxx,
        cyy=bearing1_cyy,
    )
    bearing_2 = Bearing(
        node=4,
        kxx=bearing2_kxx,
        kyy=bearing2_kyy,
        cxx=bearing2_cxx,
        cyy=bearing2_cyy,
    )
    bearings = [bearing_1, bearing_2]

    # Create rotor system
    rotor = RotorSystem(bearings=bearings)

    # Generate speed array
    speeds = np.linspace(speed_min, speed_max, int(speed_points))

    # Calculate eigenvalues at each speed
    eig_1 = np.zeros_like(speeds)
    eig_2 = np.zeros_like(speeds)
    eig_3 = np.zeros_like(speeds)
    eig_4 = np.zeros_like(speeds)
    eig_5 = np.zeros_like(speeds)
    eig_6 = np.zeros_like(speeds)

    for i, speed in enumerate(speeds):
        eig, _ = rotor.eigenvalues(Omega=speed / 60 * 2 * np.pi)
        eig_1[i] = np.abs(eig[0]) / (2 * np.pi)  # Convert to Hz
        eig_2[i] = np.abs(eig[2]) / (2 * np.pi)
        eig_3[i] = np.abs(eig[4]) / (2 * np.pi)
        eig_4[i] = np.abs(eig[6]) / (2 * np.pi)
        eig_5[i] = np.abs(eig[8]) / (2 * np.pi)
        eig_6[i] = np.abs(eig[10]) / (2 * np.pi)

    # Rotor speed line (1X synchronous)
    rotor_speed_line = speeds / 60  # RPM to Hz

    # Calculate eigenvalues at key speeds for table display
    key_speeds = [0, 1000, 3000]
    eigenvalue_table = []
    for key_speed in key_speeds:
        eig_raw, _ = rotor.eigenvalues(Omega=key_speed / 60 * 2 * np.pi)
        # Get first 8 eigenvalues (4 pairs of complex conjugates)
        row = {
            "speed": key_speed,
            "eigenvalues": [
                {"real": float(np.real(eig_raw[i])), "imag": float(np.imag(eig_raw[i]))}
                for i in range(8)
            ],
        }
        eigenvalue_table.append(row)

    return {
        "speeds": speeds.tolist(),
        "eig_1": eig_1.tolist(),
        "eig_2": eig_2.tolist(),
        "eig_3": eig_3.tolist(),
        "eig_4": eig_4.tolist(),
        "eig_5": eig_5.tolist(),
        "eig_6": eig_6.tolist(),
        "rotor_speed_line": rotor_speed_line.tolist(),
        "eigenvalue_table": eigenvalue_table,
    }
