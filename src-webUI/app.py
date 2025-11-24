from flask import Flask, render_template, request, jsonify
import json
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

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate/<calculator_type>", methods=["POST"])
def calculate(calculator_type):
    try:
        data = request.get_json()

        if calculator_type == "friswell_K_and_C":
            # Conversions
            omega = data["omega"] * (2 * np.pi) / 60  # RPM to rad/s
            D = data["D"] / 1000  # mm to m
            c = data["c"] / 1000  # mm to m
            L = data["L"] / 1000  # mm to m
            eta = data["eta"]  # Pa·s
            f = data["f"]  # N

            epsilon = solve_eccentricity(D, omega, eta, L, f, c)
            K, C = solve_K_and_C_Friswell(omega, f, c, epsilon)

            result = {"K": K.tolist(), "C": C.tolist(), "epsilon": epsilon}

        elif calculator_type == "AlBender_K_and_C":
            lambda_val = data.get("lambda_vals", [1.0])[0]
            sigma_min = data.get("sigma_min", -1)
            sigma_max = data.get("sigma_max", 3)
            sigma_points = data.get("sigma_points", 10000)
            L_over_D = data.get("L_over_D", 1.0)

            sigma_vals = np.logspace(sigma_min, sigma_max, sigma_points)

            K, C = solve_K_and_C_AlBender(
                lambda_val,
                sigma_vals,
                L_over_D,
            )
            result = {
                "K": K.tolist(),
                "C": C.tolist(),
                "sigma_vals": sigma_vals.tolist(),
            }
        else:
            return jsonify({"error": "Unknown calculator type"}), 400

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
