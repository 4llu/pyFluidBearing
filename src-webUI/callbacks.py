"""
Calculator callback functions for web UI.
Handles data conversion and computation logic.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.bearing_calculations import (
    solve_eccentricity,
    solve_K_and_C_Friswell,
    solve_K_and_C_AlBender,
)
from plotting import create_albender_plot


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
    Generate a Plotly figure for Al-Bender results.

    Parameters
    ----------
    data : dict
        Same parameters as calculate_albender_K_and_C

    Returns
    -------
    dict
        Plotly figure as JSON
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

    # Create plot
    plot_json = create_albender_plot(sigma_vals, K, C, lambda_val)

    return plot_json
