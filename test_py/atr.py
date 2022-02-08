def ATR(DF, n=14):
  """
    Function to calclate True Range and Average True Range
  """
  df = DF.copy()
  df["H-L"] = df["High"] - df['Low']
  df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
  df["L-PC"] = abs(df['Low'] - df["Adj Close"].shift(1))
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
  df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
  return df["ATR"]