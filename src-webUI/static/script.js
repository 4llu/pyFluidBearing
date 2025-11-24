class CalculatorApp {
    constructor() {
        this.initEventListeners();
    }

    initEventListeners() {
        // Friswell form
        document
            .getElementById("friswell-form")
            .addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateFriswell();
            });

        // Al-Bender form
        document
            .getElementById("albender-form")
            .addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateAlBender();
            });

        // Al-Bender plot button
        const plotButton = document.getElementById("albender-plot-btn");
        if (plotButton) {
            plotButton.addEventListener("click", (e) => {
                e.preventDefault();
                this.plotAlBender();
            });
        }
    }

    async calculateFriswell() {
        const formData = this.getFriswellData();
        if (!formData) return;

        try {
            this.setLoading("friswell-form", true);

            const response = await fetch("/calculate/friswell_K_and_C", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayFriswellResult(result.result);
            } else {
                this.displayError("friswell-result-content", result.error);
            }
        } catch (error) {
            this.displayError(
                "friswell-result-content",
                "Network error: " + error.message
            );
        } finally {
            this.setLoading("friswell-form", false);
        }
    }

    async calculateAlBender() {
        const formData = this.getAlBenderData();
        if (!formData) return;

        try {
            this.setLoading("albender-form", true);

            const response = await fetch("/calculate/AlBender_K_and_C", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayAlBenderResult(result.result);
            } else {
                this.displayError("albender-result-content", result.error);
            }
        } catch (error) {
            this.displayError(
                "albender-result-content",
                "Network error: " + error.message
            );
        } finally {
            this.setLoading("albender-form", false);
        }
    }

    getFriswellData() {
        const omega = parseFloat(
            document.getElementById("friswell-omega").value
        );
        const D = parseFloat(document.getElementById("friswell-D").value);
        const L = parseFloat(document.getElementById("friswell-L").value);
        const f = parseFloat(document.getElementById("friswell-f").value);
        const c = parseFloat(document.getElementById("friswell-c").value);
        const eta = parseFloat(document.getElementById("friswell-eta").value);

        if (
            isNaN(omega) ||
            isNaN(D) ||
            isNaN(L) ||
            isNaN(f) ||
            isNaN(c) ||
            isNaN(eta)
        ) {
            alert("Please fill in all required fields with valid numbers");
            return null;
        }

        return { omega, D, L, f, c, eta };
    }

    getAlBenderData() {
        const lambda_val = parseFloat(
            document.getElementById("albender-lambda").value
        );
        const sigma_min = parseFloat(
            document.getElementById("albender-sigma-min").value
        );
        const sigma_max = parseFloat(
            document.getElementById("albender-sigma-max").value
        );
        const sigma_points = parseInt(
            document.getElementById("albender-sigma-points").value
        );
        const L_over_D = parseFloat(
            document.getElementById("albender-ld").value
        );

        if (isNaN(lambda_val)) {
            alert("Please fill in the frequency parameter (Λ)");
            return null;
        }

        return {
            lambda_vals: [lambda_val],
            sigma_min: isNaN(sigma_min) ? -1 : sigma_min,
            sigma_max: isNaN(sigma_max) ? 3 : sigma_max,
            sigma_points: isNaN(sigma_points) ? 10000 : sigma_points,
            L_over_D: isNaN(L_over_D) ? 1.0 : L_over_D,
        };
    }

    displayFriswellResult(result) {
        const resultDiv = document.getElementById("friswell-result");
        const contentDiv = document.getElementById("friswell-result-content");

        const K = result.K;
        const C = result.C;
        const epsilon = result.epsilon;

        contentDiv.innerHTML = `
            <div class="success">
                <h4>Computed Eccentricity (ε):</h4>
                <p>${epsilon.toFixed(6)}</p>
                
                <h4>Stiffness Matrix K (N/m):</h4>
                <pre>[${K[0][0].toExponential(4)}, ${K[0][1].toExponential(4)}]
[${K[1][0].toExponential(4)}, ${K[1][1].toExponential(4)}]</pre>
                
                <h4>Damping Matrix C (N·s/m):</h4>
                <pre>[${C[0][0].toExponential(4)}, ${C[0][1].toExponential(4)}]
[${C[1][0].toExponential(4)}, ${C[1][1].toExponential(4)}]</pre>
            </div>
        `;

        resultDiv.classList.remove("hidden");
    }

    displayAlBenderResult(result) {
        const resultDiv = document.getElementById("albender-result");
        const contentDiv = document.getElementById("albender-result-content");

        const K = result.K;
        const C = result.C;
        const sigma_vals = result.sigma_vals;

        // Note: K and C are arrays with shape matching sigma_vals
        // Display dimensions and first few values
        const K_shape = `${K.length}x${K[0].length}x${K[0][0].length}`;
        const C_shape = `${C.length}x${C[0].length}x${C[0][0].length}`;

        contentDiv.innerHTML = `
            <div class="success">
                <p><strong>Note:</strong> Results are arrays with ${
                    sigma_vals.length
                } sigma points (from ${sigma_vals[0].toExponential(
            2
        )} to ${sigma_vals[sigma_vals.length - 1].toExponential(2)}).</p>
                <h4>Stiffness Matrix K (shape: ${K_shape}):</h4>
                <p>First value at σ[0] = ${sigma_vals[0].toExponential(4)}:</p>
                <pre>[${K[0][0][0].toExponential(
                    4
                )}, ${K[0][1][0].toExponential(4)}]
[${K[1][0][0].toExponential(4)}, ${K[1][1][0].toExponential(4)}]</pre>
                
                <h4>Damping Matrix C (shape: ${C_shape}):</h4>
                <p>First value at σ[0] = ${sigma_vals[0].toExponential(4)}:</p>
                <pre>[${C[0][0][0].toExponential(
                    4
                )}, ${C[0][1][0].toExponential(4)}]
[${C[1][0][0].toExponential(4)}, ${C[1][1][0].toExponential(4)}]</pre>
            </div>
        `;

        resultDiv.classList.remove("hidden");
    }

    displayError(elementId, message) {
        const element = document.getElementById(elementId);

        element.innerHTML = `
            <div class="error">
                <p><strong>Error:</strong> ${message}</p>
            </div>
        `;

        element.parentElement.classList.remove("hidden");
    }

    async plotAlBender() {
        const formData = this.getAlBenderData();
        if (!formData) return;

        try {
            const plotButton = document.getElementById("albender-plot-btn");
            plotButton.disabled = true;
            plotButton.textContent = "Generating Plot...";

            const response = await fetch("/plot/albender", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayPlot(result.plot);
            } else {
                alert("Error generating plot: " + result.error);
            }
        } catch (error) {
            alert("Network error: " + error.message);
        } finally {
            const plotButton = document.getElementById("albender-plot-btn");
            plotButton.disabled = false;
            plotButton.textContent = "Generate Plot";
        }
    }

    displayPlot(plotJson) {
        const plotContainer = document.getElementById(
            "albender-plot-container"
        );
        const plotDiv = document.getElementById("albender-plot");

        // Parse and display the Plotly figure
        const figure = JSON.parse(plotJson);
        Plotly.newPlot(plotDiv, figure.data, figure.layout, {
            responsive: true,
        });

        plotContainer.classList.remove("hidden");
    }

    setLoading(formId, isLoading) {
        const form = document.getElementById(formId);
        const buttons = form.querySelectorAll("button");

        buttons.forEach((button) => {
            button.disabled = isLoading;
            if (isLoading) {
                button.textContent = "Calculating...";
            } else {
                button.textContent = "Calculate K and C";
            }
        });
    }
}

// Initialize the app when the page loads
document.addEventListener("DOMContentLoaded", () => {
    new CalculatorApp();
});
