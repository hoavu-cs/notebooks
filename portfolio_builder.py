import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import contextlib
import os

def find_combinations(k, total, increment=5):
    """
    Recursive function to find all combinations of k integers that are multiples of increment
    and sum to 'total'.
    """
    if k == 1:
        return [[total]]
    
    combinations = []
    for i in range(0, total + 1, increment):
        sub_combinations = find_combinations(k - 1, total - i, increment)
        for sub_combination in sub_combinations:
            combinations.append([i] + sub_combination)
    
    return combinations
    

def calculate_portfolio_return(tickers, weights, start_date, end_date):
    # Check that weights sum to 1
    # if sum(weights) != 1:
    #     print(weights)
    #     raise ValueError("Weights must sum to 1")

    # weights.sort()
    # tickers.sort()
    
    # Download adjusted close prices for the tickers
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    data = data[tickers]
    
    start_price = data.iloc[0].values
    end_price = data.iloc[-1].values

    start_portfolio = weights / start_price
    gain = np.sum(end_price * start_portfolio / 100) - 1

    return gain


def calculate_portfolio_gain_monthly_investment(tickers, weights, start_date, end_date):
    
    # Normalize weights to percentage
    weights = np.array(weights) / 100
    monthly_investment = 1000  # Monthly investment amount
    
    # Download adjusted close prices for the tickers
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    data = data[tickers]

    # Resample data to monthly (taking the price on the first trading day of each month)
    monthly_prices = data.resample('MS').first()

    #print(monthly_prices)

    # Initialize the portfolio to track the number of shares for each stock
    total_shares = np.zeros(len(tickers))

    # Loop through each month and compute the number of shares bought
    for i in range(len(monthly_prices)):
        monthly_price = monthly_prices.iloc[i].values  # Price at the beginning of the month
        
        # Allocate the monthly investment according to the weights
        investment_per_stock = weights * monthly_investment
        
        # Calculate the number of shares bought for each stock
        shares_bought = investment_per_stock / monthly_price
        
        # Accumulate the shares bought
        total_shares += shares_bought

        #print(f"Month {i + 1}, total value: {np.sum(total_shares * monthly_price)}, investment: {monthly_investment * (i + 1)}")

    # Calculate the portfolio value at the end of the period
    final_prices = monthly_prices.iloc[-1].values
    portfolio_value = np.sum(total_shares * final_prices)

    # print(f"Portfolio value: {portfolio_value}")
    # print(f"Total invested: {monthly_investment * len(monthly_prices)}")

    # Total amount invested over the period
    total_invested = monthly_investment * len(monthly_prices)

    # Calculate the gain
    gain = (portfolio_value - total_invested) / total_invested

    return gain

# Example usage
tickers = ['UPRO',  'SSO']
#tickers = ['UPRO', 'GLD']
start_date = '2022-01-01'
end_date = '2024-10-01'
best_return = -1
best_weights = []

weights = find_combinations(len(tickers), 100, 5)
#weights = [[100, 0]]

for weight in weights:
    #print(weight)
    portfolio_return = calculate_portfolio_gain_monthly_investment(tickers, weight, start_date, end_date)

    if portfolio_return > 0.5:
        print(f"Portfolio return: {portfolio_return * 100:.2f}% with weights {weight}")

    if portfolio_return > best_return:
        best_return = portfolio_return
        best_weights = weight
        #print(f"New best return: {best_return * 100:.2f}% with weights {best_weights}")

print(f"Best return: {best_return * 100:.2f}%")
print("Best weights:", best_weights)
