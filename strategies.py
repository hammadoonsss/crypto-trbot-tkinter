import time
import logging
import numpy as np
import pandas as pd
import tkinter as tk

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from typing import *

from models import *

from threading import Timer

from stocktrends import Renko

if TYPE_CHECKING:
  from connectors.bitmex import BitmexClient
  from connectors.binance_futures import BinanceFuturesClient

logger = logging.getLogger()

#timeframe equivalent
TF_EQUIV = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}


class Strategy:
  def __init__(self, client: Union["BitmexClient", "BinanceFuturesClient"], contract: Contract, exchange: str,
               timeframe: str, balance_pct: float, take_profit: float, stop_loss: float, strat_name):

    self.client = client

    self.contract = contract
    self.exchange = exchange
    self.tf = timeframe
    self.tf_equiv = TF_EQUIV[timeframe] * 1000
    self.balance_pct = balance_pct
    self.take_profit = take_profit
    self.stop_loss = stop_loss

    self.strat_name = strat_name

    self.ongoing_position = False

    self.candles: List[Candle] = []
    self.trades: List[Trade] = []
    self.logs = []

  def _add_log(self, msg: str):
    logger.info("%s", msg)
    self.logs.append({"log": msg, "displayed": False})

  def parse_trade(self, price: float, size: float, timestamp: int) -> str:

    #To monitor Time Difference between the current time and the trade time
    timestamp_diff = int(time.time() * 1000) - timestamp

    if timestamp_diff >= 2000:
      logger.warning("%s %s: %s milliseconds of difference between the current time and the trade time.",
                     self.exchange, self.contract.symbol, timestamp_diff)
    
    last_candle = self.candles[-1]

    # Same Candle
    if timestamp < last_candle.timestamp + self.tf_equiv:
      
      last_candle.close = price
      last_candle.volume += size

      if price > last_candle.high:
        last_candle.high = price
      elif price < last_candle.low:
        last_candle.low = price

      # Check Take Profit / Stop Loss (Every-time Trade Update for the same Candle). 
      for trade in self.trades:
        if trade.status == "open" and trade.entry_price is not None:
          self._check_tp_sl(trade)

      return "same_candle"

    # Missing Candle(s)
    elif timestamp >= last_candle.timestamp + 2 * self.tf_equiv:

      missing_candles = int((timestamp - last_candle.timestamp) / self.tf_equiv) - 1

      logger.info("%s Missing %s candles for %s %s (%s %s)", self.exchange, missing_candles,
                  self.contract.symbol, self.tf, timestamp, last_candle.timestamp)

      for missing in missing_candles:
        new_ts = last_candle.timestamp + self.tf_equiv
        candle_info = {'ts': new_ts, 'open':last_candle.close, 'high':last_candle.close,
                       'low':last_candle.close, 'close': last_candle.close, 'volume': 0}
        new_candle = Candle(candle_info, self.tf, "parse_trade")
        
        self.candles.append(new_candle)

        last_candle = new_candle
      
      new_ts = last_candle.timestamp + self.tf_equiv
      candle_info = {'ts': new_ts, 'open':price , 'high':price , 'low':price , 'close': price, 'volume': size}
      new_candle = Candle(candle_info, self.tf, "parse_trade")

      self.candles.append(new_candle)

      return "new candle"

    # New Candle
    elif timestamp >= last_candle.timestamp + self.tf_equiv:

      new_ts = last_candle.timestamp + self.tf_equiv
      candle_info = {'ts': new_ts, 'open':price , 'high':price , 'low':price , 'close': price, 'volume': size}
      new_candle = Candle(candle_info, self.tf, "parse_trade")

      self.candles.append(new_candle)

      logger.info("%s New candle for %s %s", self.exchange, self.contract.symbol, self.tf)

      return "new candle"

  def _check_order_status(self, order_id):

    order_status = self.client.get_order_status(self.contract, order_id)

    if order_status is not None:

      logger.info("%s order status: %s", self.exchange, order_status.status)

      if order_status.status == "filled":
        for trade in self.trades:
          if trade.entry_id == order_id:
            trade.entry_price = order_status.avg_price
            break
        return

    # Check _check_order_status() function every 2 seconds until the order_status is "filled"
    t = Timer(2.0, lambda:self._check_order_status(order_id))
    t.start()

  def _open_position(self, signal_result: int):
    
    trade_size = self.client.get_trade_size(self.contract, self.candles[-1].close, self.balance_pct)
    if trade_size is None:
      return

    order_side = "buy" if signal_result == 1 else "sell"
    position_side = "long" if signal_result == 1 else "short"

    self._add_log(f"{position_side.capitalize()} signal on {self.contract.symbol} {self.tf}")

    order_status = self.client.place_order(self.contract, "MARKET", trade_size, order_side)

    if order_status is not None:
      self._add_log(f"{order_side.capitalize()} order placed on {self.exchange} | Status: {order_status.status}")

      self.ongoing_position = True

      avg_fill_price = None

      if order_status.status == "filled":
        avg_fill_price = order_status.avg_price
      else:
        t = Timer(2.0, lambda:self._check_order_status(order_status.order_id))
        t.start()

      new_trade = Trade({"time": int(time.time() * 1000), "entry_price": avg_fill_price,
                         "contract": self.contract, "strategy": self.strat_name, "side": position_side,
                         "status": "open", "pnl": 0, "quantity": trade_size, "entry_id": order_status.order_id})
      self.trades.append(new_trade)

  def _check_tp_sl(self, trade: Trade):

    """
      Check Take Profit and Stop Loss
      Close the Position when Take profit and Stop losss Reached
    """

    tp_triggered = False
    sl_triggered = False

    price = self.candles[-1].close

    if trade.side == "long":
      if self.stop_loss is not None:
        if price <= trade.entry_price * ( 1 - self.stop_loss / 100 ):
          sl_triggered = True
        
      if self.take_profit is not None:
        if price >= trade.entry_price * ( 1 + self.take_profit / 100 ):
          tp_triggered = True

    elif trade.side == "short":
      if self.stop_loss is not None:
        if price >= trade.entry_price * ( 1 + self.stop_loss / 100 ):
          sl_triggered = True
        
      if self.take_profit is not None:
        if price <= trade.entry_price * ( 1 - self.take_profit / 100 ):
          tp_triggered = True


    if tp_triggered  or sl_triggered:

      self._add_log(f"{'Stop Loss' if sl_triggered else 'Take Profit'} for {self.contract.symbol} {self.tf}")

      order_side = "SELL" if trade.side == "long" else "BUY"
      order_status = self.client.place_order(self.contract, "MARKET", trade.quantity, order_side)

      if order_status is not None:
          self._add_log(f"Exit order on {self.contract.symbol} {self.tf} placed successfully")
          trade.status = "closed"
          self.ongoing_position = False


class TechnicalStrategy(Strategy):

  def __init__(self, client, contract: Contract, exchange: str, timeframe: str,
               balance_pct: float, take_profit: float, stop_loss: float,
               other_params: Dict):
    super().__init__(client, contract, exchange, timeframe, balance_pct, take_profit, stop_loss, "Technical")

    self._ema_fast = other_params['ema_fast']
    self._ema_slow = other_params['ema_slow']
    self._ema_signal = other_params['ema_signal']

    self._rsi_length = other_params['rsi_length']

    self.root= tk.Tk()

  def _candle_list(self):
    """
      Candle Lists of Hisorical Candle:
           - Timeframe, Open, High, Low, Close, Volume
    """
  
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


  def _bollinger_band(self):
    """
      Bollinger Bands (BB)
          - Moving Average (MA)
          - Standard Deviation (STD)
          - Upper Band, Lower Band
    """
    bb_period = 20
    bb_multiplier = 2

    try:
      df = self._candle_list()
      bbdf = df.copy()

      ma = bbdf['Close'].rolling(bb_period).mean()
      std = bbdf['Close'].rolling(bb_period).std()

      bbdf['Upper_band'] = ma + bb_multiplier * std
      bbdf['Lower_band'] = ma - bb_multiplier * std

      bbdf = bbdf.dropna()

      figure = plt.Figure(figsize=(8,8), dpi=200)
      ax = figure.add_subplot(111)
      chart_type = FigureCanvasTkAgg(figure, self.root)
      chart_type.get_tk_widget().pack()
      df = bbdf[['Upper_band','Lower_band']].groupby(bbdf['DateTime']).sum()
      df.plot(kind='line', legend=True, ax=ax, fontsize=10)
      ax.set_title('Bollinger_Bands')

      return bbdf
      # print("bbdf: \n", bbdf)

      # plt.figure(figsize=(2,3), dpi=200)
      # plt.plot(bbdf['timeframe'], bbdf['Upper_band'], label="Upperband")
      # plt.plot(bbdf['timeframe'], bbdf['Lower_band'], label="Lowerband")
      # plt.title("Bollinger Bands", fontdict={'fontsize': 20})
      # plt.xlabel("Timeframe")
      # plt.ylabel("Data")
      # plt.legend()
      # plt.grid()
      # plt.show()

      # bbdf[['Upper_band', 'Lower_band']].plot(figsize=(10,5))
      # plt.grid()
      # plt.show()
      
    except Exception as e:
      print("Error in BB: ", e)
  

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


  def _renko(self):
    """
      Renko Chart with ATR
            - pip install stocktrends
            - from stocktrends import Renko
            - On Historical Data with different timeframe
    """

    n=120
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


    hist_data = self.client.get_historical_candles(self.contract, "1h")
    # print('hist_data:__ ', hist_data)

    tf_list = []
    op_list = []
    hi_list = []
    lo_list = []
    cl_list = []
    vol_list = []

    for candle in hist_data:
      tf_list.append(candle.timestamp)
      op_list.append(candle.open)
      hi_list.append(candle.high)
      lo_list.append(candle.low)
      cl_list.append(candle.close)
      vol_list.append(candle.volume)
    
    # print("tf_list: ", tf_list)

    try:
      df = pd.DataFrame(timeframe_list, columns=['timeframe'])
      df['Open'] = pd.DataFrame(open_list)
      df['High'] = pd.DataFrame(high_list)
      df['Low'] = pd.DataFrame(low_list)
      df['Close'] = pd.DataFrame(close_list)
      df['Volume'] = pd.DataFrame(volume_list)

      # print('dfatrrenko :: \n', df)

      newdf = pd.DataFrame(tf_list, columns=['TF'])
      newdf['Op'] = pd.DataFrame(open_list)
      newdf['Hi'] = pd.DataFrame(high_list)
      newdf['Lo'] = pd.DataFrame(low_list)
      newdf['Cl'] = pd.DataFrame(close_list)
      newdf['Vol'] = pd.DataFrame(volume_list)
      
      # print('newdf :: \n', newdf)

    except Exception as e:
      print("Error DF_R-ATR :", e)

    try:
      atr_df = newdf.copy()
      atr_df["H-L"] = atr_df["Hi"] - atr_df["Lo"]
      atr_df["H-PC"] = abs(atr_df["Hi"] - atr_df["Cl"].shift(1))
      atr_df["L-PC"] = abs(atr_df["Lo"] - atr_df["Cl"].shift(1))
      atr_df["TR"] = atr_df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
      atr_df["ATR"] = atr_df["TR"].ewm(com=n, min_periods=n).mean()
      # print("atr_df :: \n", atr_df)

    except Exception as e:
      print("Error ATR-Renko :", e)

    try:
      ren_df = df.copy()
      # ren_df.reset_index(inplace=True)         #-- Currenty NO
      ren_df.columns = ["date", "open", "high", "low", "close", "volume"]
      
      df2 = Renko(ren_df)
      df2.brick_size = 3 * round(atr_df["ATR"].iloc[-2], 0)
      renko_df = df2.get_ohlc_data()
      # print('renko_df: \n', renko_df)

      return df2.brick_size, renko_df["uptrend"].iloc[-2]

    except Exception as e:
      print("Error in Renko", e)


  def _disp_in(self):
    """
      Disparity Index (DI)
        - lookback period - 14
        - Moving Average (MA)
    """
    lookback = 14

    try:
      df = self._candle_list()
      disp_df = df.copy()

      disp_df['MA'] = disp_df['Close'].rolling(lookback).mean()
      disp_df['DI'] = 100 * ((disp_df['Close'] - disp_df['MA']) / disp_df['MA'])

      disp_df = disp_df.dropna()
      # print('disp_df: \n', disp_df)
      return disp_df['DI'].iloc[-2]

    except Exception as e:
      print("Error in DispIn:", e)


  def _tsi(self):
    """
      True Strenth Index (TSI)
          - lookback period for the long EMA - 25
          - lookback period for the short EMA - 13
          - lookback period for the signal line EMA - (within 7 to 12 )
    """
    long = 25
    short = 7
    signal = 12

    try:
      df = self._candle_list()
      tsi_df = df.copy()

      tsi_df['diff'] = tsi_df['Close'] - tsi_df['Close'].shift(1)
      tsi_df['abs_diff'] = abs(tsi_df['diff'])

      tsi_df['diff_smoothed'] = tsi_df['diff'].ewm(span=long, adjust=False).mean()
      tsi_df['diff_double_smoothed'] = tsi_df['diff_smoothed'].ewm(span=short, adjust=False).mean()
      tsi_df['abs_diff_smoothed'] = tsi_df['abs_diff'].ewm(span=long, adjust=False).mean()
      tsi_df['abs_diff_double_smoothed'] = tsi_df['abs_diff_smoothed'].ewm(span=short, adjust=False).mean()

      tsi_df['tsi'] = 100 * (tsi_df['diff_double_smoothed'] / tsi_df['abs_diff_double_smoothed'])
      tsi_df['tsi_signal'] = tsi_df['tsi'].ewm(span=signal, adjust=False).mean()

      tsi_df = tsi_df.dropna()
      # print("tsi_df: \n", tsi_df)
      
      return tsi_df['tsi'].iloc[-2], tsi_df['tsi_signal'].iloc[-2]

    except Exception as e:
      print("Error in TSI: ", e)


  def _stoch(self):
    """
      Stochastic 
          - n_high 
          - n_low 
          - %k (fast line)
          - %d (slow line)
    """
    k_period = 14
    d_period = 3

    try:
      df = self._candle_list()
      stoch_df = df.copy()

      stoch_df['n_high'] = stoch_df['High'].rolling(k_period).max()
      stoch_df['n_low'] = stoch_df['Low'].rolling(k_period).min()

      stoch_df['%K'] = 100 * ((stoch_df['Close'] - stoch_df['n_low']) / (stoch_df['n_high'] - stoch_df['n_low']))
      stoch_df['%D'] = stoch_df['%K'].rolling(d_period).mean()

      stoch_df = stoch_df.dropna()
      # print("stoch_df: \n", stoch_df)

    except Exception as e:
      print("Error in CCI:", e)
    

  def _cci(self):
    """
      Commodity Channel Index (CCI)
          - Typical Price (TP)
          - Moving Average (SMA)
          - Mean Deviation (MAD)
          - Lambert's constant (0.015)
    """
    constant = 0.015
    lookback = 14

    try:
      df  = self._candle_list()
      cci_df = df.copy()

      cci_df['TP'] = (cci_df['High'] + cci_df['Low'] + cci_df['Close']) / 3
      cci_df['SMA'] = cci_df['TP'].rolling(lookback).mean()
      cci_df['MAD'] = cci_df['TP'].rolling(lookback).apply(lambda x : pd.Series(x).mad())
      cci_df['CCI'] = (cci_df['TP'] - cci_df['SMA']) / (constant * cci_df['MAD'])

      cci_df = cci_df.dropna()
      # print("CCI_df: \n", cci_df)

    except Exception as e:
      print("Error in CCI: ", e)

  def _wir(self):
    """
      Williams %R (WIR)
        - lookback Highest High (High_H)
        - lookback Lowest Low (Low_L)
    """
    lookback=14

    try:
      df = self._candle_list()
      wir_df = df.copy()

      wir_df['High_H'] = wir_df['High'].rolling(lookback).max()
      wir_df['Low_L'] = wir_df['Low'].rolling(lookback).min()
      wir_df['WIR'] = -100 * ((wir_df['High_H'] - wir_df['Close'])  / (wir_df['High_H'] - wir_df['Low_L']))

      wir_df = wir_df.dropna()
      # print('wir_df: \n', wir_df)
      
    except Exception as e:
      print("Error in WIR: ", e)


  def _ichimoku(self):
    """
      Ichimoku Cloud (IC)
        - Tenkan-Sen/Conversion Line      -  Period-20/9
        - Kijun-Sen/Base Line             -  Period-60/26
        - Senkou Sen A/Leading Span A     -  Period-30/26
        - Senkou Sen B/Leading Span B     -  Period-120/52
        - Chikou/Lagging Span             -  Period-30/26
    """
    cl_period = 20
    bl_period = 60
    lead_b_period = 120
    lag_period = 30

    try:
      df = self._candle_list()
      ichi_df = df.copy()

      high_20 = ichi_df['High'].rolling(cl_period).max()
      low_20 = ichi_df['Low'].rolling(cl_period).min()

      ichi_df['Conversion_Line'] = (high_20 + low_20) / 2

      high_60 = ichi_df['High'].rolling(bl_period).max()
      low_60 = ichi_df['Low'].rolling(bl_period).min()

      ichi_df['Base_Line'] = (high_60 + low_60) / 2

      ichi_df['Lead_span_A'] = ((ichi_df['Conversion_Line'] + ichi_df['Base_Line']) / 2).shift(lag_period)

      high_120 = ichi_df['High'].rolling(lead_b_period).max()
      low_120 = ichi_df['Low'].rolling(lead_b_period).min()

      ichi_df['Lead_span_B'] = ((high_120 + low_120) / 2).shift(lead_b_period)

      ichi_df['Lagging_span'] = ichi_df['Close'].shift(-lag_period)

      ichi_df = ichi_df.dropna()
      # print('ichi_df: \n', ichi_df)

    except Exception as e:
      print("Error in Ichimoku: ", e)

  def _psar(self):
    """
      Parabolic SAR (PSAR)
    """
    # import TA-Lib
    import talib
    
    acceleration = 0.02
    maximum = 0.2

    try:
      df = self._candle_list() 
      psar = df.copy()
      
      high = psar['High']
      low = psar['Low']

      psar['SAR'] = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)

      psar = psar.dropna()
      # print('psar: \n', psar)
      
      # psar[['Close', 'SAR']].plot(figsize=(10,5))
      # plt.grid()
      # plt.show()

    except Exception as e:
      print("Error in PSAR: ", e)
  

  def _rsi(self):
    """
      Relative Strength Index (RSI) 
        - delta, up, down
        - avg_gain, avg_loss, rs
    """

    try:
      df = self._candle_list()
      rsi_df = df.copy()

      delta = rsi_df['Close'].diff().dropna()
      up, down = delta.copy(), delta.copy()

      up[up < 0] = 0
      down[down > 0] = 0

      avg_gain = up.ewm(com=(self._rsi_length - 1), min_periods=self._rsi_length).mean()
      avg_loss = down.abs().ewm(com=(self._rsi_length - 1), min_periods=self._rsi_length).mean()

      rs = avg_gain/avg_loss

      rsi_df['RSI'] = (100 - ( 100 / ( 1 + rs ))).round(2)
      rsi_df = rsi_df.dropna()
      # print('rsi_df: \n', rsi_df)

      try:
        figure = plt.Figure(figsize=(8,9), dpi=100)
        ax = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, self.root)
        chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df = rsi_df[['RSI']].groupby(rsi_df['DateTime']).sum()
        df.plot(kind='line', legend=True, ax=ax, fontsize=10)
        ax.set_title('RSI')
      except Exception as e:
        print("Error RSI Graph: ", e)

      return rsi_df['RSI'].iloc[-2]

    except Exception as e:
      print("Error in RSI:", e)
      

  def _macd(self) -> Tuple[float, float]:
    """
      Moving Average Convergence Divergence (MACD) && Exponential Moving Average (EMA) 
        Calculation steps:
       1) Fast EMA Calculation
       2) Slow EMA Calculation
       3) Fast EMA - Slow EMA
       4) Fast EMA on the result of 3)
    """

    try:

      df = self._candle_list()
      macd_df = df.copy()

      ema_fast = macd_df['Close'].ewm(span=self._ema_fast).mean() # ewm() -  Exponential Weighted functions
      ema_slow = macd_df['Close'].ewm(span=self._ema_slow).mean()

      macd_df['MACD_Line'] =  ema_fast - ema_slow      # macd of last finished candle
      macd_df['MACD_Signal'] = macd_df['MACD_Line'].ewm(span=self._ema_signal).mean()

      macd_df = macd_df.dropna()
      # print('macd_df: \n', macd_df)

      try:
        figure = plt.Figure(figsize=(8,9), dpi=100)
        ax = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, self.root)
        chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df = macd_df[['MACD_Line','MACD_Signal']].groupby(macd_df['DateTime']).sum()
        df.plot(kind='line', legend=True, ax=ax, fontsize=10)
        ax.set_title('MACD')
      except Exception as e:
        print("Error MACD Graph: ", e)

      return macd_df['MACD_Line'].iloc[-2], macd_df['MACD_Signal'].iloc[-2]
      
    except Exception as e:
      print("Error in MACD:", e)


  def _check_signal(self):
    """
      Technical strategy check signal:
       For long signal-> 1 || short signal -> -1 || no signal -> 0        
    """

    rsi = self._rsi()
    # self. _disp_in()
    # self._tsi()
    macd_line, macd_signal = self._macd()
    # self._renko()
    # self._adxatr()
    # self._adx()
    # bbdf = self._bollinger_band()
    # self._adxatrcom()
    # self._stoch()
    # self._cci()
    # self._wir()
    # self._ichimoku()
    # self._psar()


    if rsi < 30 and macd_line > macd_signal:
      return 1
    elif rsi > 70 and macd_line < macd_signal:
      return -1 
    else: 
      return 0

  def check_trade(self, tick_type: str):
    
    if tick_type == "new_signal" and not self.ongoing_position:
      signal_result = self._check_signal()

      if signal_result in [1, -1]:
        self._open_position(signal_result)


class BreakoutStrategy(Strategy):

  def __init__(self, client, contract: Contract, exchange: str, timeframe: str,
               balance_pct: float, take_profit: float, stop_loss: float,
               other_params: Dict):
    super().__init__(client, contract, exchange, timeframe, balance_pct, take_profit, stop_loss, "Breakout")

    self._min_volume = other_params['min_volume']


  def _check_signal(self) -> int:
    """
      Breakout strategy check signal:
       For long signal-> 1 || short signal -> -1 || no signal -> 0        
    """

    if self.candles[1].close > self.candles[-2].high and self.candles[-1].volume > self._min_volume:
      return 1

    elif self.candles[1].close < self.candles[-2].low and self.candles[-1].volume > self._min_volume:
      return -1

    else:
      return 0

  def check_trade(self, tick_type: str):

    if not self.ongoing_position:
      signal_result = self._check_signal()

      if signal_result in [1, -1]:
        self._open_position(signal_result)