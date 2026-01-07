// Handle form submission and call FastAPI API
const form = document.getElementById("prediction-form");
const resultText = document.getElementById("prediction-text");

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // Prevent default form submission

  // Collect form data and convert to JSON
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    // Call FastAPI's /predict endpoint
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    resultText.textContent = response.ok
      // Display prediction result
      ? `The prediction is: ${data.prediction}`
      // Display error message if API returns error
      : `Error: ${data.error || "Unknown error"}`;
  } catch (err) {
    // Handle network or request failure
    resultText.textContent = "Request failed: " + err.message;
  }
});
