class CalculatorApp {
    constructor() {
        this.friswellMode = "single";
        this.initEventListeners();
        // Store reference globally for inline event handlers
        window.calculatorApp = this;
    }

    initEventListeners() {
        // Friswell form
        const friswellForm = document.getElementById("friswell-form");
        if (friswellForm) {
            friswellForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateFriswell();
            });
        }

        // Friswell plot button
        const friswellPlotButton = document.getElementById("friswell-plot-btn");
        if (friswellPlotButton) {
            friswellPlotButton.addEventListener("click", (e) => {
                e.preventDefault();
                this.plotFriswell();
            });
        }

        // Al-Bender form
        const albenderForm = document.getElementById("albender-form");
        if (albenderForm) {
            albenderForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateAlBender();
            });
        }

        // Al-Bender plot button
        const plotButton = document.getElementById("albender-plot-btn");
        if (plotButton) {
            plotButton.addEventListener("click", (e) => {
                e.preventDefault();
                this.plotAlBender();
            });
        }

        // Pressure distribution form
        const pressureForm = document.getElementById("pressure-form");
        if (pressureForm) {
            pressureForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculatePressureDistribution();
            });
        }

        // Compressible flow form
        const compressibleForm = document.getElementById("compressible-form");
        if (compressibleForm) {
            compressibleForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateCompressibleFlow();
            });
        }

        // Campbell diagram form
        const campbellForm = document.getElementById("campbell-form");
        if (campbellForm) {
            campbellForm.addEventListener("submit", (e) => {
                e.preventDefault();
                this.calculateCampbell();
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

    // Friswell Mode Toggle
    toggleFriswellMode(mode) {
        this.friswellMode = mode;
        const rangeOptions = document.getElementById("friswell-range-options");
        const calcBtn = document.getElementById("friswell-calc-btn");
        const plotBtn = document.getElementById("friswell-plot-btn");
        const resultDiv = document.getElementById("friswell-result");
        const plotContainer = document.getElementById(
            "friswell-plot-container"
        );

        if (mode === "range") {
            rangeOptions.classList.remove("hidden");
            calcBtn.classList.add("hidden");
            plotBtn.classList.remove("hidden");
            resultDiv.classList.add("hidden");
            // Set default values for the first sweep parameter
            this.updateSweepDefaults();
        } else {
            rangeOptions.classList.add("hidden");
            calcBtn.classList.remove("hidden");
            plotBtn.classList.add("hidden");
            plotContainer.classList.add("hidden");
        }
    }

    updateSweepDefaults() {
        const paramSelect = document.getElementById("friswell-sweep-param");
        const selectedOption = paramSelect.options[paramSelect.selectedIndex];
        const minValue = selectedOption.getAttribute("data-min");
        const maxValue = selectedOption.getAttribute("data-max");
        const unit = selectedOption.getAttribute("data-unit");
        const sweepParam = selectedOption.value;

        document.getElementById("friswell-sweep-min").value = minValue;
        document.getElementById("friswell-sweep-max").value = maxValue;
        document.getElementById(
            "friswell-sweep-min-label"
        ).textContent = `Min Value [${unit}]`;
        document.getElementById(
            "friswell-sweep-max-label"
        ).textContent = `Max Value [${unit}]`;

        // Disable/enable form inputs based on sweep parameter
        const omegaInput = document.getElementById("friswell-omega");
        const fInput = document.getElementById("friswell-f");
        const DInput = document.getElementById("friswell-D");
        const LInput = document.getElementById("friswell-L");
        const cInput = document.getElementById("friswell-c");
        const etaInput = document.getElementById("friswell-eta");

        // Helper function to disable/enable input
        const setInputState = (input, isDisabled) => {
            input.disabled = isDisabled;
            if (isDisabled) {
                input.classList.add("opacity-50", "cursor-not-allowed");
            } else {
                input.classList.remove("opacity-50", "cursor-not-allowed");
            }
        };

        // Reset all inputs to enabled
        setInputState(omegaInput, false);
        setInputState(fInput, false);
        setInputState(DInput, false);
        setInputState(LInput, false);
        setInputState(cInput, false);
        setInputState(etaInput, false);

        // Disable the parameter being swept
        switch (sweepParam) {
            case "omega":
                setInputState(omegaInput, true);
                break;
            case "f":
                setInputState(fInput, true);
                break;
            case "D":
                setInputState(DInput, true);
                break;
            case "L":
                setInputState(LInput, true);
                break;
            case "c":
                setInputState(cInput, true);
                break;
            case "eta":
                setInputState(etaInput, true);
                break;
        }
    }

    async plotFriswell() {
        const formData = this.getFriswellPlotData();
        if (!formData) return;

        try {
            const plotButton = document.getElementById("friswell-plot-btn");
            plotButton.disabled = true;
            plotButton.textContent = "Generating Plot...";

            const response = await fetch("/plot/friswell", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayFriswellD3Plot(result.plot);
            } else {
                alert("Error generating plot: " + result.error);
            }
        } catch (error) {
            alert("Network error: " + error.message);
        } finally {
            const plotButton = document.getElementById("friswell-plot-btn");
            plotButton.disabled = false;
            plotButton.textContent = "Generate Plot";
        }
    }

    getFriswellPlotData() {
        const formData = this.getFriswellData();
        if (!formData) return null;

        const sweepParam = document.getElementById(
            "friswell-sweep-param"
        ).value;
        const sweepMin = parseFloat(
            document.getElementById("friswell-sweep-min").value
        );
        const sweepMax = parseFloat(
            document.getElementById("friswell-sweep-max").value
        );
        const sweepPoints = parseInt(
            document.getElementById("friswell-sweep-points").value
        );

        if (isNaN(sweepMin) || isNaN(sweepMax) || isNaN(sweepPoints)) {
            alert("Please fill in all range parameters with valid numbers");
            return null;
        }

        return {
            ...formData,
            sweep_param: sweepParam,
            sweep_min: sweepMin,
            sweep_max: sweepMax,
            sweep_points: sweepPoints,
        };
    }

    displayFriswellD3Plot(plotData) {
        const svg = d3.select("#friswell-plot");
        svg.selectAll("*").remove(); // Clear previous content

        const svgWidth = +svg.attr("width");
        const svgHeight = +svg.attr("height");
        const margin = { top: 30, right: 15, bottom: 50, left: 55 };
        const rowGap = 50; // Space between rows
        const colGap = 55; // Space between columns for y-axis ticks

        // 2x4 grid: 2 rows (K, C), 4 columns (xx, xy, yx, yy)
        const plotWidth =
            (svgWidth - margin.left - margin.right - 3 * colGap) / 4;
        const plotHeight =
            (svgHeight - margin.top - margin.bottom - rowGap) / 2;

        const coefficients = [
            ["K_xx", "K_xy", "K_yx", "K_yy"],
            ["C_xx", "C_xy", "C_yx", "C_yy"],
        ];

        const coefficientData = {
            K_xx: plotData.K_xx,
            K_xy: plotData.K_xy,
            K_yx: plotData.K_yx,
            K_yy: plotData.K_yy,
            C_xx: plotData.C_xx,
            C_xy: plotData.C_xy,
            C_yx: plotData.C_yx,
            C_yy: plotData.C_yy,
        };

        const sweepVals = plotData.sweep_vals;
        const xLabel = plotData.x_label;

        // Calculate y-scales for each coefficient
        const yScales = {};
        Object.keys(coefficientData).forEach((coeff) => {
            const data = coefficientData[coeff];
            const maxVal = d3.max(data);
            const minVal = d3.min(data);
            const padding =
                (maxVal - minVal) * 0.1 || Math.abs(maxVal) * 0.1 || 1;
            yScales[coeff] = [minVal - padding, maxVal + padding];
        });

        // Plot 2 rows x 4 columns
        coefficients.forEach((row, rowIndex) => {
            row.forEach((coeff, colIndex) => {
                const x0 = margin.left + colIndex * (plotWidth + colGap);
                const y0 = margin.top + rowIndex * (plotHeight + rowGap);

                const g = svg
                    .append("g")
                    .attr("transform", `translate(${x0},${y0})`);

                // Create scales
                const x = d3
                    .scaleLinear()
                    .domain(d3.extent(sweepVals))
                    .range([0, plotWidth]);
                const y = d3
                    .scaleLinear()
                    .domain(yScales[coeff])
                    .range([plotHeight, 0]);

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
                    .style("stroke", "#ddd")
                    .style("opacity", "0.8");
                gridY.select(".domain").style("display", "none");

                // Add grid lines for x-axis
                const gridX = g
                    .append("g")
                    .attr("class", "grid-x")
                    .attr("transform", `translate(0,${plotHeight})`)
                    .call(
                        d3
                            .axisBottom(x)
                            .ticks(4)
                            .tickSize(-plotHeight)
                            .tickFormat("")
                    );
                gridX
                    .selectAll("line")
                    .style("stroke", "#ddd")
                    .style("opacity", "0.8");
                gridX.select(".domain").style("display", "none");

                // Add axes
                const xAxis = g
                    .append("g")
                    .attr("class", "x-axis")
                    .attr("transform", `translate(0,${plotHeight})`)
                    .call(d3.axisBottom(x).ticks(4, "~s"))
                    .style("font-size", "9px");

                const yAxis = g
                    .append("g")
                    .attr("class", "y-axis")
                    .call(d3.axisLeft(y).ticks(5, ".2s"))
                    .style("font-size", "9px");

                // Add x-axis label (only on bottom row)
                if (rowIndex === 1) {
                    g.append("text")
                        .attr("x", plotWidth / 2)
                        .attr("y", plotHeight + 35)
                        .attr("text-anchor", "middle")
                        .attr("fill", "black")
                        .style("font-size", "11px")
                        .text(xLabel);
                }

                // Add subplot title
                g.append("text")
                    .attr("x", plotWidth / 2)
                    .attr("y", -10)
                    .attr("text-anchor", "middle")
                    .attr("fill", "black")
                    .style("font-size", "12px")
                    .style("font-weight", "bold")
                    .text(coeff);

                // Draw line
                const lineColor = coeff.startsWith("K") ? "#2563eb" : "#7c3aed";
                g.append("path")
                    .datum(
                        sweepVals.map((d, i) => ({
                            x: d,
                            y: coefficientData[coeff][i],
                        }))
                    )
                    .attr("fill", "none")
                    .attr("stroke", lineColor)
                    .attr("stroke-width", 2)
                    .attr(
                        "d",
                        d3
                            .line()
                            .x((d) => x(d.x))
                            .y((d) => y(d.y))
                    );
            });
        });

        // Add row labels
        svg.append("text")
            .attr("x", 15)
            .attr("y", margin.top + plotHeight / 2)
            .attr("text-anchor", "middle")
            .attr(
                "transform",
                `rotate(-90, 15, ${margin.top + plotHeight / 2})`
            )
            .attr("fill", "#2563eb")
            .style("font-size", "14px")
            .style("font-weight", "bold")
            .text("K [N/m]");

        svg.append("text")
            .attr("x", 15)
            .attr("y", margin.top + plotHeight + rowGap + plotHeight / 2)
            .attr("text-anchor", "middle")
            .attr(
                "transform",
                `rotate(-90, 15, ${
                    margin.top + plotHeight + rowGap + plotHeight / 2
                })`
            )
            .attr("fill", "#7c3aed")
            .style("font-size", "14px")
            .style("font-weight", "bold")
            .text("C [N·s/m]");

        // Show the plot container
        const plotContainer = document.getElementById(
            "friswell-plot-container"
        );
        plotContainer.classList.remove("hidden");
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

    async calculatePressureDistribution() {
        const formData = this.getPressureData();
        if (!formData) return;

        try {
            this.setPressureLoading(true);

            const response = await fetch("/calculate/pressure_distribution", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayPressureResult(result.result);
            } else {
                this.displayError("pressure-stats", result.error);
            }
        } catch (error) {
            this.displayError(
                "pressure-stats",
                "Network error: " + error.message
            );
        } finally {
            this.setPressureLoading(false);
        }
    }

    getPressureData() {
        const mu = parseFloat(document.getElementById("mu").value);
        const omega = parseFloat(document.getElementById("omega").value);
        const r_in = parseFloat(document.getElementById("r_in").value);
        const r_out = parseFloat(document.getElementById("r_out").value);
        const n_sector = parseInt(document.getElementById("n_sector").value);
        const Delta_theta = parseFloat(
            document.getElementById("Delta_theta").value
        );
        const hL = parseFloat(document.getElementById("hL").value) * 1e-6; // Convert μm to m
        const hT = parseFloat(document.getElementById("hT").value) * 1e-6; // Convert μm to m

        return {
            mu,
            omega,
            r_in,
            r_out,
            n_sector,
            Delta_theta,
            hL,
            hT,
        };
    }

    displayPressureResult(result) {
        const resultDiv = document.getElementById("pressure-result");
        const statsDiv = document.getElementById("pressure-stats");
        const plotImg = document.getElementById("pressure-plot");

        // Display statistics
        statsDiv.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                    <h4 class="text-sm font-semibold text-gray-700 mb-1">Peak Pressure</h4>
                    <p class="text-lg font-bold text-blue-700">${
                        result.stats.peak_pressure || "N/A"
                    }</p>
                </div>
                <div class="bg-green-50 border-l-4 border-green-500 p-3 rounded">
                    <h4 class="text-sm font-semibold text-gray-700 mb-1">Mean Pressure</h4>
                    <p class="text-lg font-bold text-green-700">${
                        result.stats.mean_pressure || "N/A"
                    }</p>
                </div>
            </div>
        `;

        // Display the plot image
        plotImg.src = result.image;

        resultDiv.classList.remove("hidden");
    }

    setPressureLoading(isLoading) {
        const form = document.getElementById("pressure-form");
        const button = form.querySelector("button[type='submit']");

        button.disabled = isLoading;
        button.textContent = isLoading
            ? "Calculating..."
            : "Calculate Pressure Distribution";
    }

    // Compressible Flow Methods
    async calculateCompressibleFlow() {
        const formData = this.getCompressibleData();
        if (!formData) return;

        try {
            this.setCompressibleLoading(true);

            const response = await fetch("/calculate/compressible_flow", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (result.error) {
                alert(`Error: ${result.error}`);
            } else {
                this.displayCompressibleResult(result.result);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            this.setCompressibleLoading(false);
        }
    }

    getCompressibleData() {
        const form = document.getElementById("compressible-form");
        const formData = new FormData(form);

        return {
            mu: parseFloat(formData.get("mu")),
            omega: parseFloat(formData.get("omega")),
            r_in: parseFloat(formData.get("r_in")),
            r_out: parseFloat(formData.get("r_out")),
            n_sector: parseInt(formData.get("n_sector")),
            Delta_theta:
                parseFloat(formData.get("Delta_theta")) * (Math.PI / 180), // Convert to radians
            hL: parseFloat(formData.get("hL")) * 1e-6, // Convert μm to m
            hT: parseFloat(formData.get("hT")) * 1e-6, // Convert μm to m
            rho_a: parseFloat(formData.get("rho_a")),
            p_a: parseFloat(formData.get("p_a")),
            beta: parseFloat(formData.get("beta")),
            max_iter: parseInt(formData.get("max_iter")),
            tol: parseFloat(formData.get("tol")),
            relax: parseFloat(formData.get("relax")),
            boundary: form.querySelector('input[name="boundary"]').checked,
        };
    }

    displayCompressibleResult(result) {
        const resultDiv = document.getElementById("compressible-result");
        const statsDiv = document.getElementById("compressible-stats");
        const plotImg = document.getElementById("compressible-plot");

        // Display statistics
        let statsHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                    <h4 class="text-sm font-semibold text-gray-700 mb-1">Peak Pressure</h4>
                    <p class="text-lg font-bold text-blue-700">${
                        result.stats.peak_pressure || "N/A"
                    }</p>
                </div>
                <div class="bg-green-50 border-l-4 border-green-500 p-3 rounded">
                    <h4 class="text-sm font-semibold text-gray-700 mb-1">Mean Pressure</h4>
                    <p class="text-lg font-bold text-green-700">${
                        result.stats.mean_pressure || "N/A"
                    }</p>
                </div>
        `;

        // Add iterations info if available
        if (result.stats.iterations) {
            statsHTML += `
                <div class="bg-purple-50 border-l-4 border-purple-500 p-3 rounded">
                    <h4 class="text-sm font-semibold text-gray-700 mb-1">Convergence</h4>
                    <p class="text-lg font-bold text-purple-700">${result.stats.iterations}</p>
                </div>
            `;
        }

        statsHTML += `</div>`;
        statsDiv.innerHTML = statsHTML;

        // Display the plot image
        plotImg.src = result.image;

        resultDiv.classList.remove("hidden");
    }

    setCompressibleLoading(isLoading) {
        const form = document.getElementById("compressible-form");
        const button = form.querySelector("button[type='submit']");

        button.disabled = isLoading;
        button.textContent = isLoading
            ? "Calculating..."
            : "Calculate Compressible Flow";
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

    // Campbell Diagram Methods
    async calculateCampbell() {
        const formData = this.getCampbellData();
        if (!formData) return;

        try {
            this.setCampbellLoading(true);

            const response = await fetch("/api/campbell", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (response.ok) {
                this.displayCampbellD3Plot(result.result);
            } else {
                alert("Error calculating Campbell diagram: " + result.error);
            }
        } catch (error) {
            alert("Network error: " + error.message);
        } finally {
            this.setCampbellLoading(false);
        }
    }

    getCampbellData() {
        return {
            bearing1_kxx: parseFloat(
                document.getElementById("bearing1_kxx").value
            ),
            bearing1_kyy: parseFloat(
                document.getElementById("bearing1_kyy").value
            ),
            bearing1_cxx: parseFloat(
                document.getElementById("bearing1_cxx").value
            ),
            bearing1_cyy: parseFloat(
                document.getElementById("bearing1_cyy").value
            ),
            bearing2_kxx: parseFloat(
                document.getElementById("bearing2_kxx").value
            ),
            bearing2_kyy: parseFloat(
                document.getElementById("bearing2_kyy").value
            ),
            bearing2_cxx: parseFloat(
                document.getElementById("bearing2_cxx").value
            ),
            bearing2_cyy: parseFloat(
                document.getElementById("bearing2_cyy").value
            ),
            speed_min: parseFloat(document.getElementById("speed_min").value),
            speed_max: parseFloat(document.getElementById("speed_max").value),
            speed_points: parseInt(
                document.getElementById("speed_points").value
            ),
        };
    }

    setCampbellLoading(isLoading) {
        const button = document.getElementById("campbell-btn");
        if (button) {
            button.disabled = isLoading;
            button.textContent = isLoading
                ? "Calculating..."
                : "Generate Campbell Diagram";
        }
    }

    displayCampbellD3Plot(plotData) {
        const svg = d3.select("#campbell-plot");
        svg.selectAll("*").remove();

        const svgWidth = +svg.attr("width");
        const svgHeight = +svg.attr("height");
        const margin = { top: 60, right: 150, bottom: 60, left: 70 };
        const plotWidth = svgWidth - margin.left - margin.right;
        const plotHeight = svgHeight - margin.top - margin.bottom;

        const speeds = plotData.speeds;
        const eigenvalues = [
            { name: "First", data: plotData.eig_1, color: "#e41a1c" },
            { name: "Second", data: plotData.eig_2, color: "#377eb8" },
            { name: "Third", data: plotData.eig_3, color: "#4daf4a" },
            { name: "Fourth", data: plotData.eig_4, color: "#984ea3" },
            { name: "Fifth", data: plotData.eig_5, color: "#ff7f00" },
            { name: "Sixth", data: plotData.eig_6, color: "#a65628" },
        ];
        const rotorSpeedLine = plotData.rotor_speed_line;

        // Calculate scales
        const xExtent = d3.extent(speeds);
        const allFreqs = eigenvalues
            .flatMap((e) => e.data)
            .concat(rotorSpeedLine);
        const yMax = d3.max(allFreqs) * 1.1;

        const x = d3.scaleLinear().domain(xExtent).range([0, plotWidth]);

        const y = d3.scaleLinear().domain([0, yMax]).range([plotHeight, 0]);

        // Create main group
        const g = svg
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Add grid lines
        g.append("g")
            .attr("class", "grid-y")
            .call(d3.axisLeft(y).ticks(10).tickSize(-plotWidth).tickFormat(""))
            .selectAll("line")
            .style("stroke", "#e0e0e0")
            .style("stroke-dasharray", "2,2");

        g.append("g")
            .attr("class", "grid-x")
            .attr("transform", `translate(0,${plotHeight})`)
            .call(
                d3.axisBottom(x).ticks(10).tickSize(-plotHeight).tickFormat("")
            )
            .selectAll("line")
            .style("stroke", "#e0e0e0")
            .style("stroke-dasharray", "2,2");

        // Remove grid domain lines
        g.selectAll(".grid-x .domain, .grid-y .domain").style(
            "display",
            "none"
        );

        // Add axes
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${plotHeight})`)
            .call(d3.axisBottom(x).ticks(10))
            .style("font-size", "11px");

        g.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(y).ticks(10))
            .style("font-size", "11px");

        // Add axis labels
        g.append("text")
            .attr("x", plotWidth / 2)
            .attr("y", plotHeight + 45)
            .attr("text-anchor", "middle")
            .style("font-size", "13px")
            .style("font-weight", "500")
            .text("Speed (RPM)");

        g.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -plotHeight / 2)
            .attr("y", -50)
            .attr("text-anchor", "middle")
            .style("font-size", "13px")
            .style("font-weight", "500")
            .text("Natural Frequency (Hz)");

        // Add title
        svg.append("text")
            .attr("x", svgWidth / 2)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .style("font-weight", "600")
            .text("Campbell Diagram");

        // Line generator
        const line = d3
            .line()
            .x((d, i) => x(speeds[i]))
            .y((d) => y(d));

        // Draw rotor speed line (dashed)
        g.append("path")
            .datum(rotorSpeedLine)
            .attr("fill", "none")
            .attr("stroke", "#2563eb")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "6,4")
            .attr("d", line);

        // Draw eigenvalue lines
        eigenvalues.forEach((eig) => {
            g.append("path")
                .datum(eig.data)
                .attr("fill", "none")
                .attr("stroke", eig.color)
                .attr("stroke-width", 2)
                .attr("d", line);
        });

        // Add legend
        const legend = svg
            .append("g")
            .attr(
                "transform",
                `translate(${svgWidth - margin.right + 20}, ${margin.top})`
            );

        // Rotor speed legend entry
        legend
            .append("line")
            .attr("x1", 0)
            .attr("x2", 25)
            .attr("y1", 0)
            .attr("y2", 0)
            .attr("stroke", "#2563eb")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "6,4");

        legend
            .append("text")
            .attr("x", 30)
            .attr("y", 4)
            .style("font-size", "11px")
            .text("Rotor Speed");

        // Eigenvalue legend entries
        eigenvalues.forEach((eig, i) => {
            const yOffset = (i + 1) * 22;

            legend
                .append("line")
                .attr("x1", 0)
                .attr("x2", 25)
                .attr("y1", yOffset)
                .attr("y2", yOffset)
                .attr("stroke", eig.color)
                .attr("stroke-width", 2);

            legend
                .append("text")
                .attr("x", 30)
                .attr("y", yOffset + 4)
                .style("font-size", "11px")
                .text(eig.name);
        });

        // Show the plot container
        const plotContainer = document.getElementById(
            "campbell-plot-container"
        );
        plotContainer.classList.remove("hidden");

        // Display eigenvalue table
        this.displayEigenvalueTable(plotData.eigenvalue_table);
    }

    displayEigenvalueTable(tableData) {
        const container = document.getElementById("eigenvalue-tables");
        if (!container || !tableData) return;

        container.innerHTML = "";

        tableData.forEach((row) => {
            const tableWrapper = document.createElement("div");
            tableWrapper.className = "overflow-x-auto";

            const speedLabel = document.createElement("h3");
            speedLabel.className = "text-md font-medium text-slate-700 mb-2";
            speedLabel.innerHTML = `Ω = <span class="text-violet-600 font-semibold">${row.speed}</span> RPM`;
            tableWrapper.appendChild(speedLabel);

            const table = document.createElement("table");
            table.className =
                "min-w-full divide-y divide-slate-200 border border-slate-200 rounded-lg overflow-hidden";

            // Table header
            const thead = document.createElement("thead");
            thead.className = "bg-slate-50";
            thead.innerHTML = `
                <tr>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Mode</th>
                    <th class="px-4 py-3 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">Real Part (σ)</th>
                    <th class="px-4 py-3 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">Imag Part (ω)</th>
                    <th class="px-4 py-3 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">Frequency (Hz)</th>
                    <th class="px-4 py-3 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">Damping Ratio</th>
                </tr>
            `;
            table.appendChild(thead);

            // Table body
            const tbody = document.createElement("tbody");
            tbody.className = "bg-white divide-y divide-slate-100";

            row.eigenvalues.forEach((eig, idx) => {
                const magnitude = Math.sqrt(
                    eig.real * eig.real + eig.imag * eig.imag
                );
                const freqHz = Math.abs(eig.imag) / (2 * Math.PI);
                // Damping ratio: -σ / |λ| (positive means stable)
                const dampingRatio = magnitude > 0 ? -eig.real / magnitude : 0;

                const tr = document.createElement("tr");
                tr.className = idx % 2 === 0 ? "bg-white" : "bg-slate-50/50";
                tr.innerHTML = `
                    <td class="px-4 py-2 text-sm font-medium text-slate-700">${
                        idx + 1
                    }</td>
                    <td class="px-4 py-2 text-sm text-right font-mono ${
                        eig.real >= 0 ? "text-red-600" : "text-slate-600"
                    }">${eig.real.toFixed(2)}</td>
                    <td class="px-4 py-2 text-sm text-right font-mono text-slate-600">${eig.imag.toFixed(
                        2
                    )}</td>
                    <td class="px-4 py-2 text-sm text-right font-mono text-violet-600 font-medium">${freqHz.toFixed(
                        2
                    )}</td>
                    <td class="px-4 py-2 text-sm text-right font-mono ${
                        dampingRatio < 0 ? "text-red-600" : "text-emerald-600"
                    }">${(dampingRatio * 100).toFixed(2)}%</td>
                `;
                tbody.appendChild(tr);
            });

            table.appendChild(tbody);
            tableWrapper.appendChild(table);
            container.appendChild(tableWrapper);
        });

        // Show the table container
        const tableContainer = document.getElementById(
            "eigenvalue-table-container"
        );
        tableContainer.classList.remove("hidden");
    }
}

// Initialize the app when the page loads
document.addEventListener("DOMContentLoaded", () => {
    new CalculatorApp();
});
