"""
Plotting functions for bearing calculations using Plotly.
"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json


def create_albender_plot(sigma_vals, K, C, lambda_val):
    """
    Create a Plotly figure showing Al-Bender dynamic coefficients for a single lambda.

    Parameters
    ----------
    sigma_vals : array-like
        Sigma values (x-axis)
    K : numpy.ndarray
        Stiffness matrix (2x2) where each element is an array
    C : numpy.ndarray
        Damping matrix (2x2) where each element is an array
    lambda_val : float
        Lambda value used for calculation

    Returns
    -------
    dict
        Plotly figure as dict
    """
    # Extract coefficients - K and C are (2,2) arrays where each element is an array
    K_xx = np.array(K[0, 0]).flatten()
    K_xy = np.array(K[0, 1]).flatten()
    C_xx = np.array(C[0, 0]).flatten()
    C_xy = np.array(C[0, 1]).flatten()

    # Ensure sigma_vals is a flat array
    sigma_vals = np.array(sigma_vals).flatten()

    # Create DataFrame for cleaner data handling
    df = pd.DataFrame(
        {"sigma": sigma_vals, "K_xx": K_xx, "K_xy": K_xy, "C_xx": C_xx, "C_xy": C_xy}
    )

    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Stiffness K<sub>xx</sub>",
            "Stiffness K<sub>xy</sub>",
            "Damping C<sub>xx</sub>",
            "Damping C<sub>xy</sub>",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Add K_xx
    fig.add_trace(
        go.Scatter(
            x=df["sigma"],
            y=df["K_xx"],
            mode="lines",
            name=f"K<sub>xx</sub> (Λ={lambda_val})",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    # Add K_xy
    fig.add_trace(
        go.Scatter(
            x=df["sigma"],
            y=df["K_xy"],
            mode="lines",
            name=f"K<sub>xy</sub> (Λ={lambda_val})",
            line=dict(width=2),
        ),
        row=1,
        col=2,
    )

    # Add C_xx
    fig.add_trace(
        go.Scatter(
            x=df["sigma"],
            y=df["C_xx"],
            mode="lines",
            name=f"C<sub>xx</sub> (Λ={lambda_val})",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    # Add C_xy
    fig.add_trace(
        go.Scatter(
            x=df["sigma"],
            y=df["C_xy"],
            mode="lines",
            name=f"C<sub>xy</sub> (Λ={lambda_val})",
            line=dict(width=2),
        ),
        row=2,
        col=2,
    )

    # Update x-axes to log scale
    fig.update_xaxes(type="log", title_text="σ", row=1, col=1)
    fig.update_xaxes(type="log", title_text="σ", row=1, col=2)
    fig.update_xaxes(type="log", title_text="σ", row=2, col=1)
    fig.update_xaxes(type="log", title_text="σ", row=2, col=2)

    # Update y-axes labels
    fig.update_yaxes(title_text="K<sub>xx</sub>", row=1, col=1)
    fig.update_yaxes(title_text="K<sub>xy</sub>", row=1, col=2)
    fig.update_yaxes(title_text="C<sub>xx</sub>", row=2, col=1)
    fig.update_yaxes(title_text="C<sub>xy</sub>", row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text=f"Al-Bender Dynamic Coefficients (Λ = {lambda_val})",
        showlegend=False,
        height=700,
        template="plotly_white",
    )

    # Return the figure as a dict (not JSON string) so it can be serialized properly
    # Use json.loads(fig.to_json()) to ensure all numpy arrays are converted to lists
    # and NaNs/Infs are handled correctly by Plotly's encoder
    return json.loads(fig.to_json())


def create_multiple_lambda_plot(results_df):
    """
    Create a Plotly figure showing Al-Bender coefficients for multiple lambda values.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing columns: sigma, K_xx, K_xy, C_xx, C_xy, lambda

    Returns
    -------
    dict
        Plotly figure as dict
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Stiffness K<sub>xx</sub>",
            "Stiffness K<sub>xy</sub>",
            "Damping C<sub>xx</sub>",
            "Damping C<sub>xy</sub>",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Get unique lambda values and plot each
    lambda_values = results_df["lambda"].unique()

    for idx, lambda_val in enumerate(lambda_values):
        # Filter DataFrame for this lambda value
        df_lambda = results_df[results_df["lambda"] == lambda_val]

        color = colors[idx % len(colors)]

        # Extract data from DataFrame
        sigma_vals = df_lambda["sigma"].values
        K_xx = df_lambda["K_xx"].values
        K_xy = df_lambda["K_xy"].values
        C_xx = df_lambda["C_xx"].values
        C_xy = df_lambda["C_xy"].values

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=sigma_vals,
                y=K_xx,
                mode="lines",
                name=f"Λ={lambda_val}",
                line=dict(width=2, color=color),
                legendgroup=f"lambda_{lambda_val}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sigma_vals,
                y=K_xy,
                mode="lines",
                name=f"Λ={lambda_val}",
                line=dict(width=2, color=color),
                legendgroup=f"lambda_{lambda_val}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=sigma_vals,
                y=C_xx,
                mode="lines",
                name=f"Λ={lambda_val}",
                line=dict(width=2, color=color),
                legendgroup=f"lambda_{lambda_val}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sigma_vals,
                y=C_xy,
                mode="lines",
                name=f"Λ={lambda_val}",
                line=dict(width=2, color=color),
                legendgroup=f"lambda_{lambda_val}",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Update axes
    fig.update_xaxes(type="log", title_text="σ", row=1, col=1)
    fig.update_xaxes(type="log", title_text="σ", row=1, col=2)
    fig.update_xaxes(type="log", title_text="σ", row=2, col=1)
    fig.update_xaxes(type="log", title_text="σ", row=2, col=2)

    fig.update_yaxes(title_text="K<sub>xx</sub>", row=1, col=1)
    fig.update_yaxes(title_text="K<sub>xy</sub>", row=1, col=2)
    fig.update_yaxes(title_text="C<sub>xx</sub>", row=2, col=1)
    fig.update_yaxes(title_text="C<sub>xy</sub>", row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text="Al-Bender Dynamic Coefficients",
        height=700,
        template="plotly_white",
        legend=dict(
            title="Lambda (Λ)",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    # Return the figure as a dict (not JSON string) so it can be serialized properly
    # Use json.loads(fig.to_json()) to ensure all numpy arrays are converted to lists
    return json.loads(fig.to_json())
