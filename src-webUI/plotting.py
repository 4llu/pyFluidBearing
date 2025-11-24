"""
Plotting functions for bearing calculations using Plotly.
"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


def create_albender_plot(sigma_vals, K, C, lambda_val):
    """
    Create a Plotly figure showing Al-Bender dynamic coefficients.

    Parameters
    ----------
    sigma_vals : array-like
        Sigma values (x-axis)
    K : numpy.ndarray
        Stiffness matrix (2x2xN)
    C : numpy.ndarray
        Damping matrix (2x2xN)
    lambda_val : float
        Lambda value used for calculation

    Returns
    -------
    dict
        Plotly figure as JSON
    """
    # Extract coefficients
    K_xx = K[0, 0, :]
    K_xy = K[0, 1, :]
    C_xx = C[0, 0, :]
    C_xy = C[0, 1, :]

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
            x=sigma_vals,
            y=K_xx,
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
            x=sigma_vals,
            y=K_xy,
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
            x=sigma_vals,
            y=C_xx,
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
            x=sigma_vals,
            y=C_xy,
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

    return fig.to_json()


def create_multiple_lambda_plot(results_list):
    """
    Create a Plotly figure showing Al-Bender coefficients for multiple lambda values.

    Parameters
    ----------
    results_list : list of dict
        List of results, each containing 'sigma_vals', 'K', 'C', 'lambda_val'

    Returns
    -------
    dict
        Plotly figure as JSON
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

    for idx, result in enumerate(results_list):
        sigma_vals = np.array(result["sigma_vals"])
        K = np.array(result["K"])
        C = np.array(result["C"])
        lambda_val = result["lambda_val"]

        color = colors[idx % len(colors)]

        # Extract coefficients
        K_xx = K[0, 0, :]
        K_xy = K[0, 1, :]
        C_xx = C[0, 0, :]
        C_xy = C[0, 1, :]

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

    return fig.to_json()
