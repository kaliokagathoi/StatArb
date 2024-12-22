import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import numpy.lib.recfunctions as rf

# Define time range
start_date = '2021-01-01 00:00:00'
end_date = '2022-12-31 23:59:00'
date_rng = pd.date_range(start=start_date, end=end_date, freq='min')

# Parameters for the hsi_fm price model
np.random.seed(16)  # for reproducibility
n = len(date_rng)
dt = 1/525600  # minute increments in years (1 year = 525600 minutes)
initial_price_hsi_fm = 20000
annual_drift_hsi_fm = -0.05
annual_volatility_hsi_fm = 0.3

# Generate prices for hsi_fm with drift and volatility
hsi_fm_price = np.zeros(n)  # initialize price array
hsi_fm_price[0] = initial_price_hsi_fm
for i in range(1, n):
    hsi_fm_price[i] = hsi_fm_price[i-1] * np.exp((annual_drift_hsi_fm - 0.5 * annual_volatility_hsi_fm**2) * dt + annual_volatility_hsi_fm * np.sqrt(dt) * np.random.normal())
hsi_fm_price = np.round(hsi_fm_price).astype(int)  # ensure hsi_fm_price is an integer

# Parameters for the correlated asset (hhi_fm)
correlation = 0.90
initial_price_hhi_fm = 20000
annual_drift_hhi_fm = -0.06
annual_volatility_hhi_fm = 0.4

# Generate correlated asset prices
hhi_fm_price = np.zeros(n)
hhi_fm_price[0] = initial_price_hhi_fm
for i in range(1, n):
    epsilon = np.random.normal()
    hhi_fm_price[i] = (
        correlation * hsi_fm_price[i] +
        (1 - correlation) * (hhi_fm_price[i-1] * np.exp((annual_drift_hhi_fm - 0.5 * annual_volatility_hhi_fm**2) * dt + annual_volatility_hhi_fm * np.sqrt(dt) * epsilon))
    )
hhi_fm_price = np.round(hhi_fm_price).astype(int)  # ensure hhi_fm_price is an integer

# Randomly distributed variable for 'LAST_BID_VOLUME_0' and 'LAST_ASK_VOLUME_0'
bid_volumes = np.clip(np.random.normal(loc=5, scale=2, size=n), 1, None).round().astype(int)
ask_volumes = np.clip(np.random.normal(loc=5, scale=2, size=n), 1, None).round().astype(int)

# Random variable for 'SPREAD'
spreads = np.random.normal(loc=6, scale=2, size=n)
spreads = np.clip(spreads, 2, None)
spreads = (2 * np.round(spreads / 2)).astype(int)

# Construct DataFrames
hsi_fm = pd.DataFrame({
    'LAST_PRICE': hsi_fm_price,
    'LAST_BID_VOLUME_0': bid_volumes,
    'SPREAD': spreads,
    'LAST_BID_PRICE_0': hsi_fm_price - 0.5 * spreads,
    'LAST_ASK_VOLUME_0': ask_volumes,
    'LAST_ASK_PRICE_0': hsi_fm_price + 0.5 * spreads
}, index=date_rng)

# Ensure 'LAST_BID_PRICE_0' and 'LAST_ASK_PRICE_0' are integers for hsi_fm
hsi_fm['LAST_BID_PRICE_0'] = hsi_fm['LAST_BID_PRICE_0'].round().astype(int)
hsi_fm['LAST_ASK_PRICE_0'] = hsi_fm['LAST_ASK_PRICE_0'].round().astype(int)

hhi_fm = pd.DataFrame({
    'LAST_PRICE': hhi_fm_price,
    'LAST_BID_VOLUME_0': bid_volumes,
    'SPREAD': spreads,
    'LAST_BID_PRICE_0': hhi_fm_price - 0.5 * spreads,
    'LAST_ASK_VOLUME_0': ask_volumes,
    'LAST_ASK_PRICE_0': hhi_fm_price + 0.5 * spreads
}, index=date_rng)

# Ensure 'LAST_BID_PRICE_0' and 'LAST_ASK_PRICE_0' are integers for hhi_fm
hhi_fm['LAST_BID_PRICE_0'] = hhi_fm['LAST_BID_PRICE_0'].round().astype(int)
hhi_fm['LAST_ASK_PRICE_0'] = hhi_fm['LAST_ASK_PRICE_0'].round().astype(int)

hsi_fm.to_csv('hsi_fm.csv')
hhi_fm.to_csv('hhi_fm.csv')