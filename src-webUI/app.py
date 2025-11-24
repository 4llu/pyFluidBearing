from flask import Flask, render_template, request, jsonify
from callbacks import (
    calculate_friswell_K_and_C,
    calculate_albender_K_and_C,
    plot_albender_results,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/damping_and_stiffness")
def damping_and_stiffness():
    return render_template("damping_and_stiffness.html")


@app.route("/calculate/<calculator_type>", methods=["POST"])
def calculate(calculator_type):
    try:
        data = request.get_json()

        if calculator_type == "friswell_K_and_C":
            result = calculate_friswell_K_and_C(data)
        elif calculator_type == "AlBender_K_and_C":
            result = calculate_albender_K_and_C(data)
        else:
            return jsonify({"error": "Unknown calculator type"}), 400

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/plot/albender", methods=["POST"])
def plot_albender():
    try:
        data = request.get_json()
        plot_json = plot_albender_results(data)
        return jsonify({"plot": plot_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
