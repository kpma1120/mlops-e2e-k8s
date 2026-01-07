// Handle form submission and call FastAPI API
const form = document.getElementById("prediction-form");
const resultText = document.getElementById("prediction-text");

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // Prevent default form submission

  // Collect form data and convert to JSON
  const formData = new FormData(form);
  const jsonData = {};
  formData.forEach((value, key) => {
    jsonData[key] = value;
  });

  try {
    // Call FastAPI's /predict endpoint
    // IMPORTANT: Use full URL because Flask and FastAPI run on different ports
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(jsonData)
    });

    const data = await response.json();

    if (response.ok) {
      // Display prediction result
      resultText.textContent = "The prediction is: " + data.prediction;
    } else {
      // Display error message if API returns error
      resultText.textContent = "Error: " + (data.error || "Unknown error");
    }
  } catch (err) {
    // Handle network or request failure
    resultText.textContent = "Request failed: " + err.message;
  }
});
