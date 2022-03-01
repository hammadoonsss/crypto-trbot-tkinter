"""
  ADX and ATR (All 3 Approach)
"""
import pandas as pd
import numpy as np

def _adx(self):
  """
    Average Directional Index (ADX) - 1st Approach
  """
  high_list = []
  low_list = []
  close_list = []
  lookback = 14

  for candle in self.candles:
    high_list.append(candle.high)
    low_list.append(candle.low)
    close_list.append(candle.close)

  try:

    highes = pd.Series(high_list)
    lows = pd.Series(low_list)
    closes = pd.Series(close_list)

    plus_dm = highes.diff().dropna()
    minus_dm = lows.diff().dropna()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.Series(highes - lows).dropna()
    tr2 = pd.Series(abs(highes - closes.shift(1)).dropna())
    tr3 = pd.Series(abs(lows - closes.shift(1).dropna()))

    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    # atr = tr.rolling(lookback).mean().dropna()
    atr2 = tr.ewm(com=lookback, min_periods=lookback).mean().dropna()

    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr2).dropna()
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr2)).dropna()

    dx = 100 * (abs(plus_di - minus_di) / abs(plus_di + minus_di)).dropna()
    adx = (((dx.shift(1) * (lookback - 1)) + dx) / lookback).dropna()
    adx_smooth = adx.ewm(alpha = 1 / lookback, min_periods=lookback ).mean().dropna()

    return atr2.iloc[-2], plus_di.iloc[-2], minus_di.iloc[-2], adx_smooth.iloc[-2]

  except Exception as e:
    print("Error in ADXATR1:", e)


def _adxatr(self):
  """
    True Range and Average True Range (ATR)
  """
  n=14
  high_list = []
  low_list = []
  close_list = []

  for candle in self.candles:
    high_list.append(candle.high)
    low_list.append(candle.low)
    close_list.append(candle.close)

  try:

    highes = pd.Series(high_list)
    lows = pd.Series(low_list)
    close = pd.Series(close_list)

    hl = (highes - lows)
    hpc = (abs(highes - close.shift(1)))
    lpc = (abs(lows - close.shift(1)))

    frames = [hl, hpc, lpc]

    tr = pd.concat(frames, axis=1, join='inner').max(axis=1, skipna=False)
    atr = tr.ewm(com=n, min_periods=n).mean()

    """
    Average Directional Index (ADX) - 2nd Approach
    """

    upmove = (highes - highes.shift(1))
    downmove = (lows.shift(1) - lows)

    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)

    plus_di = 100 * (plus_dm / atr).ewm(com=n, min_periods=n).mean().dropna()
    minus_di = 100 * (minus_dm / atr).ewm(com=n, min_periods=n).mean().dropna()

    adx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di)).ewm(com=n, min_periods=n).mean().dropna()
    # print("adx2_______", adx.to_json())

    return atr.iloc[-2], plus_di.iloc[-2], minus_di.iloc[-2], adx.iloc[-2]

  except Exception as e:
    print("Error in ADXATR2: ", e)
    

def _adxatrcom(self):
  """
    ATR and ADX - 3rd Approach 
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

    return adxdf["ADX"].iloc[-2]

  except Exception as e:
    print("Error ADXATR3:", e)


# self._adxatr()
# self._adx()
# self._adxatrcom()
