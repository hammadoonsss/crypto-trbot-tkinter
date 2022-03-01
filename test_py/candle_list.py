"""

import datetime
import pandas as pd


def _candle_list(self):
  
    # Candle Lists of Hisorical Candle:
    #      - Timeframe, Open, High, Low, Close, Volume
  

  timeframe_list = []
  open_list = []
  high_list = []
  low_list=[]
  close_list = []
  volume_list = [] 
  datetime_list = []

  try:
    for candle in self.candles:
      timeframe_list.append(candle.timestamp)
      open_list.append(candle.open)
      high_list.append(candle.high)
      low_list.append(candle.low)
      close_list.append(candle.close)
      volume_list.append(candle.volume)
      date_time = datetime.datetime.fromtimestamp(candle.timestamp / 1000).strftime("%H:%M")
      datetime_list.append(date_time)

    df = pd.DataFrame(timeframe_list, columns=["timeframe"])
    df["Open"] = pd.DataFrame(open_list)
    df["High"] = pd.DataFrame(high_list)
    df["Low"] = pd.DataFrame(low_list)
    df["Close"] = pd.DataFrame(close_list)
    df["Volume"] = pd.DataFrame(volume_list)
    df["DateTime"] = pd.DataFrame(datetime_list)
    # print("Candle Lists ---------DF: \n", df)

    return df  
  except Exception as e:
    print("Error in Candle List: ", e)

# self._candle_list()

"""