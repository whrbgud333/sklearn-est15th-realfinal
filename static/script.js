const API_URL = "http://localhost:8000";

document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = "Uploading...";

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            statusDiv.innerHTML = `<span class="text-success">${data.message}</span>`;
            document.getElementById('analysisSection').classList.remove('d-none');
        } else {
            statusDiv.innerHTML = `<span class="text-danger">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="text-danger">Connection Error: ${error.message}</span>`;
    }
});

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const resultDiv = document.getElementById('analysisResult');
    resultDiv.innerHTML = "Analyzing...";

    try {
        const response = await fetch(`${API_URL}/analyze`);
        const data = await response.json();

        if (response.ok) {
            // Render Head
            let html = "<h6>First 5 Rows:</h6><div class='table-responsive'><table class='table table-sm table-striped table-bordered'><thead><tr>";
            data.head.columns.forEach(col => html += `<th>${col}</th>`);
            html += "</tr></thead><tbody>";
            data.head.data.forEach(row => {
                html += "<tr>";
                row.forEach(cell => html += `<td>${cell}</td>`);
                html += "</tr>";
            });
            html += "</tbody></table></div>";

            // Render Info
            html += "<h6 class='mt-3'>Dataset Info:</h6><ul>";
            html += `<li>Shape: ${data.info.shape[0]} rows, ${data.info.shape[1]} columns</li>`;
            html += "</ul>";

            resultDiv.innerHTML = html;
            document.getElementById('vizSection').classList.remove('d-none');
        } else {
            resultDiv.innerHTML = `<span class="text-danger">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<span class="text-danger">Connection Error: ${error.message}</span>`;
    }
});

document.getElementById('vizBtn').addEventListener('click', async () => {
    const resultDiv = document.getElementById('vizResult');
    resultDiv.innerHTML = "Generating Plots...";

    try {
        const response = await fetch(`${API_URL}/visualize`);
        const data = await response.json();

        if (response.ok) {
            let html = "";
            data.plots.forEach(plot => {
                html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6>${plot.name}</h6>
                            <img src="data:image/png;base64,${plot.image}" class="img-fluid">
                        </div>
                    </div>
                </div>`;
            });
            resultDiv.innerHTML = html;
            document.getElementById('preprocessSection').classList.remove('d-none');
        } else {
            resultDiv.innerHTML = `<span class="text-danger">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<span class="text-danger">Connection Error: ${error.message}</span>`;
    }
});

document.getElementById('preprocessBtn').addEventListener('click', async () => {
    const resultDiv = document.getElementById('preprocessResult');
    resultDiv.innerHTML = "Preprocessing...";

    try {
        const response = await fetch(`${API_URL}/preprocess`, { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    ${data.message}<br>
                    Initial Shape: ${data.initial_shape}<br>
                    Final Shape: ${data.final_shape}<br>
                    Saved as: ${data.processed_file}
                </div>`;
            document.getElementById('modelSection').classList.remove('d-none');
        } else {
            resultDiv.innerHTML = `<span class="text-danger">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<span class="text-danger">Connection Error: ${error.message}</span>`;
    }
});

document.getElementById('modelBtn').addEventListener('click', async () => {
    const target = document.getElementById('targetColumn').value;
    if (!target) {
        alert("Please enter a target column name!");
        return;
    }

    const resultDiv = document.getElementById('modelResult');
    const codeBlock = document.getElementById('codeBlock');
    const generatedCode = document.getElementById('generatedCode');

    resultDiv.innerHTML = "Generating Code & Training Model... (This may take a moment)";
    codeBlock.classList.add('d-none');

    const formData = new FormData();
    formData.append("target_column", target);

    try {
        const response = await fetch(`${API_URL}/model`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            // Show Results
            let metricsHtml = "<h6>Model Results:</h6><ul>";
            for (const [key, value] of Object.entries(data.results)) {
                metricsHtml += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            metricsHtml += "</ul>";

            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    ${data.message}
                </div>
                ${metricsHtml}
            `;

            // Show Code
            generatedCode.textContent = data.code_preview;
            codeBlock.classList.remove('d-none');

        } else {
            resultDiv.innerHTML = `<span class="text-danger">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<span class="text-danger">Connection Error: ${error.message}</span>`;
    }
});
