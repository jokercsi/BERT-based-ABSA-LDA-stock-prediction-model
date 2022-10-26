import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import specific stock prices 
SnP = yf.download('^GSPC', start='2018-01-01', end='2022-08-01', progress=False)

# 종가만 가져오기
SnP = SnP[["Close"]] 
SnP = SnP.reset_index()
SnP.columns = ['day', 'price']
SnP['day'] = pd.to_datetime(SnP['day'])


SnP.index = SnP['day']
SnP.set_index('day', inplace=True)

fig, ax = plt.subplots(figsize=(15, 8))
SnP.plot(ax=ax)


