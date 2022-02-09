"""
  Working Code:
      ATR and ADX with pandas.DataFrame 
"""

import pandas as pd
import numpy as np

def _adxatrcom(self):
  """
    ATR and ADX - Third Approach 
  """
  n=14
  timeframe_list = []
  open_list = []
  high_list = []
  low_list = []
  close_list = []
  volume_list = []

  for candle in self.candles:
    timeframe_list.append(candle.timestamp)
    open_list.append(candle.open)
    high_list.append(candle.high)
    low_list.append(candle.low)
    close_list.append(candle.close)
    volume_list.append(candle.volume)


  try:
    df = pd.DataFrame(timeframe_list, columns=['timeframe'])
    df['Open'] = pd.DataFrame(open_list)
    df['High'] = pd.DataFrame(high_list)
    df['Low'] = pd.DataFrame(low_list)
    df['Close'] = pd.DataFrame(close_list)
    df['Volume'] = pd.DataFrame(volume_list)
    
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()

    adxdf = df.copy()
    adxdf["upmove"] = df["High"] - df["High"].shift(1)
    adxdf["downmove"] = df["Low"].shift(1) - df["Low"]
    adxdf["+dm"] = np.where((adxdf["upmove"]>adxdf["downmove"]) & (adxdf["upmove"] >0), adxdf["upmove"], 0)
    adxdf["-dm"] = np.where((adxdf["downmove"]>adxdf["upmove"]) & (adxdf["downmove"] >0), adxdf["downmove"], 0)
    adxdf["+di"] = 100 * (adxdf["+dm"]/adxdf["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    adxdf["-di"] = 100 * (adxdf["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    adxdf["ADX"] = 100* abs((adxdf["+di"] - adxdf["-di"])/(adxdf["+di"] + adxdf["-di"])).ewm(alpha=1/n, min_periods=n).mean()

    
    # print('Dataframe:_________ \n', df)
    # print('ADXDF:_________ \n', adxdf)
    # print('adxdf["ADX"]: ', adxdf["ADX"][998])
  except Exception as e:
    print("Error:", e)


_adxatrcom()

