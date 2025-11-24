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
        const lambda_input = document
            .getElementById("albender-lambda")
            .value.trim();
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

        // Parse comma-separated lambda values
        const lambda_vals = lambda_input
            .split(",")
            .map((val) => parseFloat(val.trim()))
            .filter((val) => !isNaN(val));

        if (lambda_vals.length === 0) {
            alert("Please enter at least one valid frequency parameter (Λ)");
            return null;
        }

        return {
            lambda_vals: lambda_vals,
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

        // Helper function to format matrix with common exponent
        const formatMatrix = (matrix) => {
            const allValues = [
                matrix[0][0],
                matrix[0][1],
                matrix[1][0],
                matrix[1][1],
            ];

            // Find the smallest non-zero absolute value
            const nonZeroValues = allValues
                .filter((v) => v !== 0)
                .map((v) => Math.abs(v));
            const minAbsValue = Math.min(...nonZeroValues);

            // Determine exponent so smallest value doesn't start with 0
            const exponent = Math.floor(Math.log10(minAbsValue));
            const scale = Math.pow(10, exponent);

            const scaled = allValues.map((v) => (v / scale).toFixed(4));

            return {
                values: scaled,
                exponent: exponent,
            };
        };

        const kFormatted = formatMatrix(K);
        const cFormatted = formatMatrix(C);

        contentDiv.innerHTML = `
            <div class="space-y-6">
                <div class="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                    <h4 class="text-lg font-semibold text-gray-800 mb-2">Eccentricity Ratio (ε)</h4>
                    <p class="text-2xl font-bold text-blue-700">${epsilon.toFixed(
                        6
                    )}</p>
                </div>
                
                <div class="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                    <h4 class="text-lg font-semibold text-gray-800 mb-3">Stiffness Matrix <strong>K</strong> (N/m)</h4>
                    <div class="flex items-center justify-start gap-2 font-mono">
                        <span class="text-4xl font-light text-gray-400">[</span>
                        <div class="flex flex-col gap-1">
                            <div class="flex gap-4">
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[0]
                                }</span>
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[1]
                                }</span>
                            </div>
                            <div class="flex gap-4">
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[2]
                                }</span>
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[3]
                                }</span>
                            </div>
                        </div>
                        <span class="text-4xl font-light text-gray-400">]</span>
                        <span class="text-sm text-gray-600 ml-2">× 10<sup>${
                            kFormatted.exponent
                        }</sup></span>
                    </div>
                </div>
                
                <div class="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                    <h4 class="text-lg font-semibold text-gray-800 mb-3">Damping Matrix <strong>C</strong> (N·s/m)</h4>
                    <div class="flex items-center justify-start gap-2 font-mono">
                        <span class="text-4xl font-light text-gray-400">[</span>
                        <div class="flex flex-col gap-1">
                            <div class="flex gap-4">
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[0]
                                }</span>
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[1]
                                }</span>
                            </div>
                            <div class="flex gap-4">
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[2]
                                }</span>
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[3]
                                }</span>
                            </div>
                        </div>
                        <span class="text-4xl font-light text-gray-400">]</span>
                        <span class="text-sm text-gray-600 ml-2">× 10<sup>${
                            cFormatted.exponent
                        }</sup></span>
                    </div>
                </div>
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
        const lambda_vals = result.lambda_vals || [result.lambda_val];

        // Use middle indices for sigma and lambda
        const midSigmaIdx = Math.floor(sigma_vals.length / 2);
        const midLambdaIdx = Math.floor(lambda_vals.length / 2);

        // Extract 2x2 matrices at middle sigma value
        // K and C are [2][2][sigma_points] arrays
        const K_matrix = [
            [K[0][0][midSigmaIdx], K[0][1][midSigmaIdx]],
            [K[1][0][midSigmaIdx], K[1][1][midSigmaIdx]],
        ];
        const C_matrix = [
            [C[0][0][midSigmaIdx], C[0][1][midSigmaIdx]],
            [C[1][0][midSigmaIdx], C[1][1][midSigmaIdx]],
        ];

        // Helper function to format matrix with common exponent
        const formatMatrix = (matrix) => {
            const allValues = [
                matrix[0][0],
                matrix[0][1],
                matrix[1][0],
                matrix[1][1],
            ];

            // Find the smallest non-zero absolute value
            const nonZeroValues = allValues
                .filter((v) => v !== 0)
                .map((v) => Math.abs(v));
            const minAbsValue = Math.min(...nonZeroValues);

            // Determine exponent so smallest value doesn't start with 0
            const exponent = Math.floor(Math.log10(minAbsValue));
            const scale = Math.pow(10, exponent);

            const scaled = allValues.map((v) => (v / scale).toFixed(4));

            return {
                values: scaled,
                exponent: exponent,
            };
        };

        const kFormatted = formatMatrix(K_matrix);
        const cFormatted = formatMatrix(C_matrix);

        contentDiv.innerHTML = `
            <div class="space-y-6">
                <div class="bg-gray-50 border-l-4 border-gray-500 p-4 rounded">
                    <p class="text-sm text-gray-700">
                        <strong>Note:</strong> Matrices shown at middle values: 
                        σ = ${sigma_vals[midSigmaIdx].toExponential(3)}${
            lambda_vals.length > 1 ? `, Λ = ${lambda_vals[midLambdaIdx]}` : ""
        }
                    </p>
                    <p class="text-xs text-gray-600 mt-1">
                        Computed for ${
                            sigma_vals.length
                        } σ points from ${sigma_vals[0].toExponential(
            2
        )} to ${sigma_vals[sigma_vals.length - 1].toExponential(2)}
                    </p>
                </div>

                <div class="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                    <h4 class="text-lg font-semibold text-gray-800 mb-3">Stiffness Matrix <strong>K</strong> (N/m)</h4>
                    <div class="flex items-center justify-start gap-2 font-mono">
                        <span class="text-4xl font-light text-gray-400">[</span>
                        <div class="flex flex-col gap-1">
                            <div class="flex gap-4">
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[0]
                                }</span>
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[1]
                                }</span>
                            </div>
                            <div class="flex gap-4">
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[2]
                                }</span>
                                <span class="text-green-700 min-w-[100px] text-right">${
                                    kFormatted.values[3]
                                }</span>
                            </div>
                        </div>
                        <span class="text-4xl font-light text-gray-400">]</span>
                        <span class="text-sm text-gray-600 ml-2">× 10<sup>${
                            kFormatted.exponent
                        }</sup></span>
                    </div>
                </div>
                
                <div class="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                    <h4 class="text-lg font-semibold text-gray-800 mb-3">Damping Matrix <strong>C</strong> (N·s/m)</h4>
                    <div class="flex items-center justify-start gap-2 font-mono">
                        <span class="text-4xl font-light text-gray-400">[</span>
                        <div class="flex flex-col gap-1">
                            <div class="flex gap-4">
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[0]
                                }</span>
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[1]
                                }</span>
                            </div>
                            <div class="flex gap-4">
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[2]
                                }</span>
                                <span class="text-purple-700 min-w-[100px] text-right">${
                                    cFormatted.values[3]
                                }</span>
                            </div>
                        </div>
                        <span class="text-4xl font-light text-gray-400">]</span>
                        <span class="text-sm text-gray-600 ml-2">× 10<sup>${
                            cFormatted.exponent
                        }</sup></span>
                    </div>
                </div>
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
                this.displayD3Plot(result.plot);
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

    displayD3Plot(plotData) {
        const svg = d3.select("#albender-plot");
        svg.selectAll("*").remove(); // Clear previous content

        const svgWidth = +svg.attr("width");
        const svgHeight = +svg.attr("height");
        const margin = { top: 30, right: 60, bottom: 60, left: 60 };
        const rowGap = 50; // Space between rows for top row's x-axis
        // Calculate plot height: (total - top margin - bottom margin - gap) / 2
        // This ensures bottom margin is AFTER the second row's plot area
        const availableHeight = svgHeight - margin.top - margin.bottom;
        const plotHeight = (availableHeight - rowGap) / 2;
        const plotWidth = (svgWidth - margin.left - margin.right) / 2;

        const coefficients = ["K_xx", "K_xy", "C_xx", "C_xy"];
        const coefficientData = {
            K_xx: plotData.K_xx,
            K_xy: plotData.K_xy,
            C_xx: plotData.C_xx,
            C_xy: plotData.C_xy,
        };

        const lambdas = plotData.lambdas;
        const sigma = plotData.sigma;

        // Define color scale
        const color = d3.scaleOrdinal(d3.schemeCategory10);

        // Calculate independent min/max for each coefficient with proper scaling
        const yScales = {};
        coefficients.forEach((coeff) => {
            const maxVal = d3.max(coefficientData[coeff].flat());
            const minVal = d3.min(coefficientData[coeff].flat());
            // Add 10% padding to the top and ensure min includes 0
            const yMax = maxVal * 1.1;
            const yMin = Math.min(minVal, 0);
            yScales[coeff] = [yMin, yMax];
        });

        // Plot two rows separately - first row then second row
        const rows = [
            { coeffs: ["K_xx", "K_xy"], row: 0 },
            { coeffs: ["C_xx", "C_xy"], row: 1 },
        ];

        rows.forEach(({ coeffs, row }) => {
            coeffs.forEach((coeff, colIndex) => {
                const x0 = margin.left + colIndex * (plotWidth + margin.right);
                // Row 0 starts at margin.top, row 1 starts at margin.top + plotHeight + rowGap
                const y0 = margin.top + row * (plotHeight + rowGap);

                const g = svg
                    .append("g")
                    .attr("transform", `translate(${x0},${y0})`);

                // Create scales
                const x = d3
                    .scaleLog()
                    .domain(d3.extent(sigma))
                    .range([0, plotWidth]);
                const y = d3
                    .scaleLinear()
                    .domain(yScales[coeff])
                    .range([plotHeight, 0]);

                // Add grid lines for x-axis (log scale with major ticks at multiples of 10)
                // Generate ticks at powers of 10
                const [minSigma, maxSigma] = d3.extent(sigma);
                const minPower = Math.floor(Math.log10(minSigma));
                const maxPower = Math.ceil(Math.log10(maxSigma));
                const majorTicks = [];
                for (let i = minPower; i <= maxPower; i++) {
                    majorTicks.push(Math.pow(10, i));
                }

                const gridXMajor = g
                    .append("g")
                    .attr("class", "grid-x-major")
                    .attr("transform", `translate(0,${plotHeight})`)
                    .call(
                        d3
                            .axisBottom(x)
                            .tickValues(majorTicks)
                            .tickSize(-plotHeight)
                            .tickFormat("")
                    );
                gridXMajor
                    .selectAll("line")
                    .style("stroke", "#aaaaaa")
                    .style("opacity", "0.9");
                gridXMajor.selectAll("text").style("display", "none");
                gridXMajor.select(".domain").style("display", "none");

                // Add minor grid lines for x-axis
                const gridXMinor = g
                    .append("g")
                    .attr("class", "grid-x-minor")
                    .attr("transform", `translate(0,${plotHeight})`)
                    .call(
                        d3
                            .axisBottom(x)
                            .ticks(50, "")
                            .tickSize(-plotHeight)
                            .tickFormat("")
                    );
                gridXMinor
                    .selectAll("line")
                    .style("stroke", "#dddddd")
                    .style("stroke-dasharray", "2,3")
                    .style("stroke-linecap", "round")
                    .style("opacity", "0.7");
                gridXMinor.selectAll("text").style("display", "none");
                gridXMinor.select(".domain").style("display", "none");

                // Add grid lines for y-axis
                const gridY = g
                    .append("g")
                    .attr("class", "grid-y")
                    .call(
                        d3
                            .axisLeft(y)
                            .ticks(5)
                            .tickSize(-plotWidth)
                            .tickFormat("")
                    );
                gridY
                    .selectAll("line")
                    .style("stroke", "#aaaaaa")
                    .style("opacity", "0.9");
                gridY.selectAll("text").style("display", "none");
                gridY.select(".domain").style("display", "none");

                // Add actual axes with labels
                const xAxis = g
                    .append("g")
                    .attr("class", "x-axis")
                    .attr("transform", `translate(0,${plotHeight})`)
                    .call(d3.axisBottom(x).ticks(5, ".0e"))
                    .style("font-size", "10px");

                xAxis.selectAll("text").style("fill", "black");
                xAxis.selectAll("line").style("stroke", "black");
                xAxis.select(".domain").style("stroke", "black");

                const yAxis = g
                    .append("g")
                    .attr("class", "y-axis")
                    .call(d3.axisLeft(y).ticks(5))
                    .style("font-size", "10px");

                yAxis.selectAll("text").style("fill", "black");
                yAxis.selectAll("line").style("stroke", "black");
                yAxis.select(".domain").style("stroke", "black");

                // Add x-axis label (only on bottom row)
                if (row === 1) {
                    g.append("text")
                        .attr("x", plotWidth / 2)
                        .attr("y", plotHeight + 35)
                        .attr("text-anchor", "middle")
                        .attr("fill", "black")
                        .style("font-size", "12px")
                        .text("σ");
                }

                // Add y-axis label on the left side of all subplots
                g.append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", -45)
                    .attr("x", -plotHeight / 2)
                    .attr("text-anchor", "middle")
                    .attr("fill", "black")
                    .style("font-size", "12px")
                    .text(coeff);

                // Draw lines for each lambda
                lambdas.forEach((lambda, lambdaIndex) => {
                    g.append("path")
                        .datum(
                            sigma.map((d, i) => ({
                                x: d,
                                y: coefficientData[coeff][lambdaIndex][i],
                            }))
                        )
                        .attr("fill", "none")
                        .attr("stroke", color(lambda))
                        .attr("stroke-width", 2)
                        .attr(
                            "d",
                            d3
                                .line()
                                .x((d) => x(d.x))
                                .y((d) => y(d.y))
                        )
                        .attr(
                            "class",
                            `lambda-line lambda-${lambda} coeff-${coeff}`
                        );
                });
            });
        });

        // Add legend in the lower right corner of the first subplot (K_xx)
        const legendX = margin.left + plotWidth - 10;
        const legendY = margin.top + plotHeight - 10;

        const legend = svg
            .append("g")
            .attr("transform", `translate(${legendX}, ${legendY})`);

        lambdas.forEach((lambda, index) => {
            const legendRow = legend
                .append("g")
                .attr(
                    "transform",
                    `translate(0, ${-(lambdas.length - index) * 16})`
                );

            legendRow
                .append("rect")
                .attr("x", -12)
                .attr("width", 12)
                .attr("height", 12)
                .attr("fill", color(lambda));

            legendRow
                .append("text")
                .attr("x", -18)
                .attr("y", 10)
                .attr("text-anchor", "end")
                .style("font-size", "11px")
                .text(index === 0 ? `Λ=${lambda}` : `${lambda}`);
        });

        // Show the plot container
        document
            .getElementById("albender-plot-container")
            .classList.remove("hidden");
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
