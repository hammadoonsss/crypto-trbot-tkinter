"""
  Lists of Historical Candle Data
"""
import pandas as pd

def _candle_list(self):
  
  timeframe_list, open_list, high_list, low_list, close_list, volume_list = ([], ) * 6 

  for candle in self.candles:
    timeframe_list.append(candle.timeframe)
    open_list.append(candle.open)
    high_list.append(candle.high)
    low_list.append(candle.low)
    close_list.append(candle.close)
    volume_list.append(candle.volume)

  df = pd.DataFrame(timeframe_list, columns=["timeframe"])
  df["Open"] = pd.DataFrame(open_list)
  df["High"] = pd.DataFrame(high_list)
  df["Low"] = pd.DataFrame(low_list)
  df["Close"] = pd.DataFrame(close_list)
  df["Volume"] = pd.DataFrame(volume_list)

  return df
