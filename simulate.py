import plotly.express as px
import pandas as pd 
import yfinance as yf
import numpy as np
from tqdm import tqdm

def get_data(ticker):
    data = yf.Ticker(ticker + "-USD").history(period = "max").reset_index()[["Date","Open"]]
    data = data.rename(columns = {"Open":ticker})
    return data

def get_matrix(assets, upto = "0"):
    asset_data = get_data(assets[0])
    for a in range(1,len(assets)):
        asset_data = pd.merge(asset_data, get_data(assets[a]), on="Date", how="inner")
    asset_data = asset_data.set_index("Date")
    if(upto != "0"):
        asset_data = asset_data.truncate(after="2016-04-02")
    asset_data = asset_data/asset_data.iloc[0]
    return asset_data

def simulate(assets, asset_data, iterations = 10000):
    df_returns = asset_data.pct_change()
    mean_returns = df_returns.mean()
    cov_matrix = df_returns.cov()
    simulation_data = np.zeros((4+len(assets)-1, iterations))

    for i in tqdm(range(iterations)):
        weights = np.array(np.random.random(len(assets)))
        weights /= np.sum(weights)

        pf_returns = np.sum(mean_returns * weights)
        pf_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))

        simulation_data[0,i] = pf_returns
        simulation_data[1,i] = pf_std_dev
        simulation_data[2,i] = simulation_data[0,i]/simulation_data[1,i]

        for j in range(len(weights)):
            simulation_data[j+3,i] = weights[j]

    return simulation_data

def plot_simulation(assets, simulation_data):
    df_plot = pd.DataFrame(simulation_data.T, columns = ["avg-ret", "std", "Sharpe"] + assets)
    df_plot["size"] = 1
    optimum = df_plot.iloc[df_plot.idxmax(axis=0)['Sharpe']]
    fig = px.scatter(df_plot, x = "std", y = "avg-ret", color = "Sharpe",
                        template = "plotly_dark", size = "size",
                        hover_data = assets)
    fig.update_xaxes(title_text = "Volatility", title_font_size = 20)
    fig.update_yaxes(title_text = "Average Returns", title_font_size = 20)
    print("Optimal portfolio", optimum)
    fig.show()

assets = ["BTC","ETH","XRP"]
data = simulate(assets, get_matrix(assets), 10000)
plot_simulation(assets,data)

