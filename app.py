from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import io
import base64
import os

app = Flask(__name__)

# Define model directory
model_dir = "notebook/"

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def load_stock_model(stock_symbol):
    """Load the corresponding model for a given stock symbol."""
    model_path = os.path.join(model_dir, f'stock_price_model_{stock_symbol}.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

def fetch_data(stock_symbol):
    """Fetch historical stock data for the given symbol."""
    stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    return stock_data

def fetch_live_price(stock_symbol):
    """Fetch the current stock price for the given symbol."""
    live_data = yf.Ticker(stock_symbol)
    current_price = live_data.history(period='1d')
    return current_price['Close'].iloc[0]

def preprocess_data(stock_data):
    """Preprocess the stock data for model prediction."""
    data = stock_data[['Open', 'Close']]
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def generate_graph(x_data, y_data_list, title, xlabel, ylabel, labels, colors):
    """Generate a graph and return it as a base64 encoded string."""
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    for y_data, label, color in zip(y_data_list, labels, colors):
        plt.plot(x_data, y_data, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def home():
    return render_template('stock_index.html')

@app.route('/start')
def start():
    return render_template('stock_index_1.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['symbol']
    model = load_stock_model(stock_symbol)
    
    if model is None:
        return render_template('predict.html', error='Model not found for this stock.')

    stock_data = fetch_data(stock_symbol)
    if stock_data.empty:
        return render_template('predict.html', error='Invalid stock symbol or no data available')

    scaled_data = preprocess_data(stock_data)
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))

    predicted_scaled = model.predict(last_60_days)
    predicted_prices = scaler.inverse_transform(predicted_scaled)

    next_day_open_price = predicted_prices[0, 0]
    next_day_close_price = predicted_prices[0, 1]

    return render_template('predict.html', 
                           stock_symbol=stock_symbol.upper(),
                           next_day_open=next_day_open_price, 
                           next_day_close=next_day_close_price)

@app.route('/today_price', methods=['POST'])
def today_price():
    stock_symbol = request.form['symbol']
    stock_data = fetch_data(stock_symbol)
    
    if stock_data.empty:
        return render_template('today.html', error='No data available for today.')

    today_open_price = stock_data['Open'].iloc[0]
    today_close_price = stock_data['Close'].iloc[-1]
    live_price = fetch_live_price(stock_symbol)

    close_chart_url = generate_graph(
        stock_data.index, [stock_data['Close']],
        title=f"{stock_symbol} - Today's Close Price History",
        xlabel="Time", ylabel="Close Price",
        labels=["Close Price"], colors=['blue']
    )

    past_30_days = stock_data.tail(30)
    open_close_chart_url = generate_graph(
        past_30_days.index, [past_30_days['Open'], past_30_days['Close']],
        title=f"{stock_symbol} - Opening and Closing Prices (Last 30 Days)",
        xlabel="Date", ylabel="Price",
        labels=["Opening Price", "Closing Price"], colors=['green', 'red']
    )

    return render_template('today.html', 
                           stock_symbol=stock_symbol,
                           today_open=today_open_price,
                           today_close=today_close_price,
                           live_price=live_price,
                           close_chart_url=close_chart_url,
                           open_close_chart_url=open_close_chart_url)

@app.route('/historical', methods=['POST'])
def historical():
    stock_symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    if start_date >= end_date:
        return render_template('historical.html', error='Start date must be before end date.')

    historical_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if historical_data.empty:
        return render_template('historical.html', error='No data available for the given date range.')

    historical_chart_url = generate_graph(
        historical_data.index, 
        [historical_data['Close'], historical_data['Open']],
        title=f"{stock_symbol} - Historical Open and Close Prices ({start_date} to {end_date})",
        xlabel="Date", 
        ylabel="Price",
        labels=["Closing Price", "Opening Price"],
        colors=['red', 'green']
    )

    return render_template('historical.html', 
                           stock_symbol=stock_symbol.upper(),
                           start_date=start_date,
                           end_date=end_date,
                           historical_chart_url=historical_chart_url)

if __name__ == '__main__':
    app.run(debug=True)
