import numpy as np
from .atr import ATR

def ADX(DF, n=20):
  df = DF.copy()
  df["ATR"] = ATR(df, n)
  df["upmove"] = df["High"] - df["High"].shift(1)
  df["downmove"] = df["Low"].shift(1) - df["Low"]
  df["+dm"] = np.where((df["upmove"] > df["downmove"]) & (df["upmove"]>0), df["upmove"],0)
  df["-dm"] = np.where((df["downmove"] > df["upmove"]) & (df["downmove"]>0), df["downmove"],0)
  df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(com = n, min_periods=n).mean()
  df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(com = n, min_periods=n).mean()
  df["ADX"] = 100 * abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(com = n, min_periods=n).mean()
  return df["ADX"]