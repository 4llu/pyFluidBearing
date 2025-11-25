from flask import Flask, render_template, request, jsonify
from callbacks import (
    calculate_friswell_K_and_C,
    calculate_albender_K_and_C,
    plot_albender_results,
    plot_friswell_results,
    calculate_pressure_distribution,
    calculate_compressible_flow,
    calculate_campbell,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/damping_and_stiffness")
def damping_and_stiffness():
    return render_template("damping_and_stiffness.html")


@app.route("/pressure_fields")
def pressure_fields():
    return render_template("pressure_fields.html")


@app.route("/modal_analysis")
def modal_analysis():
    return render_template("modal_analysis.html")


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
        print(f"Received plot request with data: {data}")
        plot_json = plot_albender_results(data)
        print(f"Plot JSON length: {len(plot_json)}")
        print(plot_json)
        return jsonify({"plot": plot_json})
    except Exception as e:
        print(f"Error in plot_albender: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/plot/friswell", methods=["POST"])
def plot_friswell():
    try:
        data = request.get_json()
        print(f"Received Friswell plot request with data: {data}")
        plot_json = plot_friswell_results(data)
        return jsonify({"plot": plot_json})
    except Exception as e:
        print(f"Error in plot_friswell: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/calculate/pressure_distribution", methods=["POST"])
def pressure_distribution_route():
    try:
        data = request.get_json()
        result = calculate_pressure_distribution(data)
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in pressure_distribution: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/calculate/compressible_flow", methods=["POST"])
def compressible_flow_route():
    try:
        data = request.get_json()
        result = calculate_compressible_flow(data)
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in compressible_flow: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/campbell", methods=["POST"])
def campbell_route():
    try:
        data = request.get_json()
        result = calculate_campbell(data)
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in campbell: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
