"""
Calculator callback functions for web UI.
Handles data conversion and computation logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.bearing_calculations import (
    solve_eccentricity,
    solve_K_and_C_Friswell,
    solve_K_and_C_AlBender,
)
from src.pressure_distribution import pressure_distribution
from src.compressible_flow import compressible_flow
from plotting import create_albender_plot, create_multiple_lambda_plot


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
    import os
    import base64
    from io import BytesIO
    from src.pressure_distribution import pressure_distribution
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

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
    import sys
    from io import StringIO

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
    import os
    import base64
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

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
    import sys
    from io import StringIO

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
