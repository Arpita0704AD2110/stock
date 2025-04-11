document.getElementById('stockForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const ticker = document.getElementById('ticker').value;
    const days = document.getElementById('days').value;
    const resultDiv = document.getElementById('result');

    if (ticker.trim() === "" || days.trim() === "") {
        resultDiv.innerHTML = '<p style="color:red;">Please enter valid stock ticker and days.</p>';
        return;
    }

    resultDiv.innerHTML = '<p>Loading prediction...</p>';

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker, days: days })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<p style='color:red;'>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p><strong>${data.ticker}</strong> predicted price after <strong>${data.days}</strong> day(s): <strong>$${data.prediction}</strong></p>`;
        }
    })
    .catch(err => {
        resultDiv.innerHTML = `<p style='color:red;'>Request failed. Try again later.</p>`;
    });
});