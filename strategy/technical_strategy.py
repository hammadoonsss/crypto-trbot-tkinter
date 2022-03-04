# Python Packages
import numpy as np
import pandas as pd
import tkinter as tk

# Matplotlib Packages
from matplotlib import style
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# TA-Lib Package
import talib

# stocktrends Package
from stocktrends import Renko

from strategy.strategies import Strategy

from typing import *

from models import *


class TechnicalStrategy(Strategy):

    def __init__(self, client, contract: Contract, exchange: str, timeframe: str,
                 balance_pct: float, take_profit: float, stop_loss: float,
                 other_params: Dict):
        super().__init__(client, contract, exchange, timeframe,
                         balance_pct, take_profit, stop_loss, "Technical")

        self._ema_fast = other_params['ema_fast']
        self._ema_slow = other_params['ema_slow']
        self._ema_signal = other_params['ema_signal']

        self._rsi_length = other_params['rsi_length']

        self.root = tk.Tk()
        style.use('seaborn-darkgrid')

    def _candle_dict(self):
        """
        Candle Dictionary of Historical Candle
        -- Timeframe, Open, High, Low, Close, Volume, DateTime
        """
        try:
            main_dict = {
                'timeframe': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [], 'DateTime': []
            }

            for candle in self.candles:
                main_dict['timeframe'].append(candle.timestamp)
                main_dict['Open'].append(candle.open)
                main_dict['High'].append(candle.high)
                main_dict['Low'].append(candle.low)
                main_dict['Close'].append(candle.close)
                main_dict['Volume'].append(candle.volume)
                date_time = datetime.datetime.fromtimestamp(
                    candle.timestamp / 1000).strftime("%H:%M")
                main_dict['DateTime'].append(date_time)

            candle_df = pd.DataFrame.from_dict(main_dict)
            # print('Candle Dict df: \n', candle_df)
            candle_df = candle_df.dropna()
            return candle_df

        except Exception as e:
            print("Error in Candle Dictionary: ", e)

    def _bollinger_band(self, bb_period=20, bb_multiplier=2):
        """
        Bollinger Bands (BB)
        - Moving Average (MA), Standard Deviation (STD)
        - Upper Band, Lower Band
        """
        try:
            df = self._candle_dict()
            bbdf = df.copy()

            ma = bbdf['Close'].rolling(bb_period).mean()
            std = bbdf['Close'].rolling(bb_period).std()

            bbdf['Upper_band'] = ma + bb_multiplier * std
            bbdf['Lower_band'] = ma - bb_multiplier * std

            bbdf = bbdf.dropna()
            # print("bbdf: \n", bbdf)

            # Bollinger Band Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                cl = bbdf[['Close']].groupby(bbdf['DateTime']).sum()
                ub = bbdf[['Upper_band']].groupby(bbdf['DateTime']).sum()
                lb = bbdf[['Lower_band']].groupby(bbdf['DateTime']).sum()

                cl.plot(kind='line', linestyle='-', linewidth=0.5,
                        ax=ax, color='#322e2f', fontsize=5)
                ub.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#d72631', fontsize=5)
                lb.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#5c3c92', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Bollinger Band--')

            except Exception as e:
                print("Error BB Graph: ", e)

            return bbdf

        except Exception as e:
            print("Error in BB: ", e)

    def _atr(self, lookback=14):
        """
        Average True Range (ATR)
        - True Range (TR)
        """
        try:

            df = self._candle_dict()
            atr_df = df.copy()

            atr_df["H-L"] = atr_df["High"] - atr_df["Low"]
            atr_df["H-PC"] = abs(atr_df["High"] - atr_df["Close"].shift(1))
            atr_df["L-PC"] = abs(atr_df["Low"] - atr_df["Close"].shift(1))

            atr_df["TR"] = atr_df[["H-L", "H-PC", "L-PC"]
                                  ].max(axis=1, skipna=False)
            atr_df["ATR"] = atr_df["TR"].ewm(
                com=lookback, min_periods=lookback).mean()

            # atr_df = atr_df.dropna()  # No dropna() func. - All values requrired
            print('atr_df:___\n', atr_df)

            # Average True Range Graph Plot
            # try:
            #     figure = plt.Figure(figsize=(46, 67), dpi=200)
            #     ax = figure.add_subplot(111)
            #     chart_type = FigureCanvasTkAgg(figure, self.root)
            #     chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

            #     at = atr_df[['ATR']].groupby(atr_df['DateTime']).sum()

            #     at.plot(kind='line', linestyle='-.', linewidth=0.5,
            #             ax=ax, color='#b20238', fontsize=5)

            #     ax.legend(loc='lower right', fontsize=5)
            #     ax.set_title('--Average True Range--')

            # except Exception as e:
            #     print("Error ATR Graph: ", e)

            return atr_df["ATR"]

        except Exception as e:
            print("Error in ATR: ", e)

    def _adx(self, lookback=14):
        """
        Average Directional Index (ADX)
        """
        try:
            df = self._candle_dict()
            adx_df = df.copy()

            atr = self._atr(lookback)

            upmove = adx_df["High"] - adx_df["High"].shift(1)
            downmove = adx_df["Low"].shift(1) - adx_df["Low"]

            plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
            minus_dm = np.where((downmove > upmove) &
                                (downmove > 0), downmove, 0.0)

            adx_df['plus_di'] = 100 * (plus_dm/atr).ewm(alpha=1/lookback, \
                                                        min_periods=lookback).mean()
            adx_df['minus_di'] = 100 * (minus_dm/atr).ewm(alpha=1/lookback, \
                                                        min_periods=lookback).mean()

            pmm = adx_df['plus_di'] - adx_df['minus_di']
            ppm = adx_df['plus_di'] + adx_df['minus_di']

            adx_df["ADX"] = 100 * abs(pmm / ppm).ewm(
                alpha=1/lookback, min_periods=lookback).mean()

            adx_df = adx_df.dropna()
            # print('adx_df:____ \n', adx_df)

            # Average Directional Index Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = adx_df[['Close']].groupby(adx_df['DateTime']).sum()
                at = atr.groupby(adx_df['DateTime']).sum()
                ad = adx_df[['ADX']].groupby(adx_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                at.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#d72631', fontsize=5)
                ad.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#5c3c92', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Average Directional Index--')

            except Exception as e:
                print("Error ADX Graph: ", e)

            return adx_df

        except Exception as e:
            print("Error in ADX:", e)

    def _renko(self):
        """
        Renko Chart with ATR
        - pip install stocktrends
        - from stocktrends import Renko
        - On Historical Data with different timeframe
        """

        n = 120
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

        try:
            df = pd.DataFrame(timeframe_list, columns=['timeframe'])
            df['Open'] = pd.DataFrame(open_list)
            df['High'] = pd.DataFrame(high_list)
            df['Low'] = pd.DataFrame(low_list)
            df['Close'] = pd.DataFrame(close_list)
            df['Volume'] = pd.DataFrame(volume_list)

            print('dfatrrenko :: \n', df)

            newdf = pd.DataFrame(tf_list, columns=['TF'])
            newdf['Op'] = pd.DataFrame(op_list)
            newdf['Hi'] = pd.DataFrame(hi_list)
            newdf['Lo'] = pd.DataFrame(lo_list)
            newdf['Cl'] = pd.DataFrame(cl_list)
            newdf['Vol'] = pd.DataFrame(vol_list)

            print('newdf :: \n', newdf)

        except Exception as e:
            print("Error DF_R-ATR :", e)

        try:
            atr_df = newdf.copy()
            print('atr_df:____ ', atr_df)
            atr_df["H-L"] = atr_df["Hi"] - atr_df["Lo"]
            atr_df["H-PC"] = abs(atr_df["Hi"] - atr_df["Cl"].shift(1))
            atr_df["L-PC"] = abs(atr_df["Lo"] - atr_df["Cl"].shift(1))
            atr_df["TR"] = atr_df[["H-L", "H-PC", "L-PC"]
                                  ].max(axis=1, skipna=False)
            atr_df["ATR"] = atr_df["TR"].ewm(com=n, min_periods=n).mean()
            atr_df = atr_df.dropna()
            print("atr_df ::--------- \n", atr_df)

        except Exception as e:
            print("Error ATR-Renko :", e)

        try:
            # ----some error occur in renko---- 
            ren_df = df.copy()
            print('ren_df1:________\n', ren_df)
            # ren_df.reset_index(inplace=True)         #-- Currenty NO
            ren_df.columns = ["date", "open", "high", "low", "close", "volume"]
            print('ren_df2: -----\n', ren_df)
            print('atr_df["ATR"].iloc[-2]: ', atr_df["ATR"].iloc[-2])

            df2 = Renko(ren_df)
            print('df2: ____\n', df2)
            df2.brick_size = 3 * round(atr_df["ATR"].iloc[-2], 0)
            print('df2.brick_size: ____\n', df2.brick_size)
            renko_df = df2.get_ohlc_data()
            print('renko_df: \n', renko_df)

            # return df2.brick_size, renko_df["uptrend"].iloc[-2]

        except Exception as e:
            print("Error in Renko:", e)

    def _disp_in(self, lookback=14):
        """
        Disparity Index (DI)
        - Moving Average (MA), lookback period - 14
        """
        try:
            df = self._candle_dict()
            disp_df = df.copy()

            disp_df['MA'] = disp_df['Close'].rolling(lookback).mean()
            disp_df['DI'] = 100 * \
                ((disp_df['Close'] - disp_df['MA']) / disp_df['MA'])

            disp_df = disp_df.dropna()
            # print('disp_df: --------\n', disp_df)

            #  Disparity Index Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = disp_df[['Close']].groupby(disp_df['DateTime']).sum()
                di = disp_df[['DI']].groupby(disp_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                di.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#1d2887', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Disparity Index--')

            except Exception as e:
                print("Error DispIn Graph: ", e)

            return disp_df['DI'].iloc[-2]

        except Exception as e:
            print("Error in DispIn:", e)

    def _tsi(self, long=25, short=7, signal=12):
        """
        True Strenth Index (TSI)
        - lookback periods for: the long EMA - 25; short EMA - 13; signal line EMA - (within 7 to 12 );
        """
        try:
            df = self._candle_dict()
            tsi_df = df.copy()

            diff = tsi_df['Close'] - tsi_df['Close'].shift(1)
            abs_diff = abs(diff)

            diff_smoothed = diff.ewm(span=long, adjust=False).mean()
            diff_double_smoothed = diff_smoothed.ewm(
                span=short, adjust=False).mean()
            abs_diff_smoothed = abs_diff.ewm(span=long, adjust=False).mean()
            abs_diff_double_smoothed = abs_diff_smoothed.ewm(
                span=short, adjust=False).mean()

            tsi_df['tsi'] = 100 * \
                (diff_double_smoothed / abs_diff_double_smoothed)
            tsi_df['tsi_signal'] = tsi_df['tsi'].ewm(
                span=signal, adjust=False).mean()

            tsi_df = tsi_df.dropna()
            print("tsi_df:----- \n", tsi_df)

            #  True Strenth Index Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = tsi_df[['Close']].groupby(tsi_df['DateTime']).sum()
                ts = tsi_df[['tsi']].groupby(tsi_df['DateTime']).sum()
                ti = tsi_df[['tsi_signal']].groupby(tsi_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                ts.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#7f593a', fontsize=5)
                ti.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#3a7f43', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--True Strenth Index--')

            except Exception as e:
                print("Error TSI Graph: ", e)

            return tsi_df['tsi'].iloc[-2], tsi_df['tsi_signal'].iloc[-2]

        except Exception as e:
            print("Error in TSI: ", e)

    def _stoch(self, k_period=14, d_period=3):
        """
        Stochastic
        - %K (fast line), %D (slow line)
        """
        try:
            df = self._candle_dict()
            stoch_df = df.copy()

            stoch_df['n_high'] = stoch_df['High'].rolling(k_period).max()
            stoch_df['n_low'] = stoch_df['Low'].rolling(k_period).min()

            stoch_df['%K'] = 100 * ((stoch_df['Close'] - stoch_df['n_low']) /
                                    (stoch_df['n_high'] - stoch_df['n_low']))
            stoch_df['%D'] = stoch_df['%K'].rolling(d_period).mean()

            stoch_df = stoch_df.dropna()
            print("stoch_df:___ \n", stoch_df)

            #  Stochastic Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = stoch_df[['Close']].groupby(stoch_df['DateTime']).sum()
                kp = stoch_df[['%K']].groupby(stoch_df['DateTime']).sum()
                dp = stoch_df[['%D']].groupby(stoch_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                kp.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#1d3c45', fontsize=5)
                dp.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#d2601a', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Stochastic--')

            except Exception as e:
                print("Error Stoch Graph: ", e)

            return stoch_df

        except Exception as e:
            print("Error in Stoch:", e)

    def _cci(self, constant=0.015, lookback=14):
        """
        Commodity Channel Index (CCI)
        - Typical Price (TP), Moving Average (SMA), Mean Deviation (MAD)
        - Lambert's constant (0.015)
        """
        try:
            df = self._candle_dict()
            cci_df = df.copy()

            cci_df['TP'] = (cci_df['High'] + cci_df['Low'] +
                            cci_df['Close']) / 3
            cci_df['SMA'] = cci_df['TP'].rolling(lookback).mean()
            cci_df['MAD'] = cci_df['TP'].rolling(
                lookback).apply(lambda x: pd.Series(x).mad())
            cci_df['CCI'] = (cci_df['TP'] - cci_df['SMA']) / \
                (constant * cci_df['MAD'])

            cci_df = cci_df.dropna()
            print("CCI_df: ----------\n", cci_df)

            #  Commodity Channel Index Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = cci_df[['Close']].groupby(cci_df['DateTime']).sum()
                ma = cci_df[['MAD']].groupby(cci_df['DateTime']).sum()
                ci = cci_df[['CCI']].groupby(cci_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                ma.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#9d1717', fontsize=5)
                ci.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#1e3d59', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('-- Commodity Channel Index--')

            except Exception as e:
                print("Error CCI Graph: ", e)

            return cci_df

        except Exception as e:
            print("Error in CCI: ", e)

    def _wir(self, lookback=14):
        """
        Williams %R (WIR)
        - lookback Highest High (High_H), lookback Lowest Low (Low_L)
        """
        try:
            df = self._candle_dict()
            wir_df = df.copy()

            wir_df['High_H'] = wir_df['High'].rolling(lookback).max()
            wir_df['Low_L'] = wir_df['Low'].rolling(lookback).min()
            wir_df['WIR'] = -100 * ((wir_df['High_H'] - wir_df['Close']) /
                                    (wir_df['High_H'] - wir_df['Low_L']))

            wir_df = wir_df.dropna()
            print('wir_df: ------___\n', wir_df)

            # Williams %R Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                # cl = wir_df[['Close']].groupby(wir_df['DateTime']).sum()
                wi = wir_df[['WIR']].groupby(wir_df['DateTime']).sum()

                # cl.plot(kind='line', linestyle='-', linewidth=0.5,
                #         ax=ax, color='#322e2f', fontsize=5)
                wi.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#1b6535', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Williams %R--')

            except Exception as e:
                print("Error WIR Graph: ", e)

            return wir_df

        except Exception as e:
            print("Error in WIR: ", e)

    def _ichimoku(self, cl_period=20, bl_period=60, lead_b_period=120, lag_period=30):
        """
        Ichimoku Cloud (IC)
        - Tenkan-Sen/Conversion Line   -  Period-20/9
        - Kijun-Sen/Base Line          -  Period-60/26
        - Senkou Sen A/Leading Span A  -  Period-30/26
        - Senkou Sen B/Leading Span B  -  Period-120/52
        - Chikou/Lagging Span          -  Period-30/26
        """
        try:
            df = self._candle_dict()
            ichi_df = df.copy()

            high_20 = ichi_df['High'].rolling(cl_period).max()
            low_20 = ichi_df['Low'].rolling(cl_period).min()

            ichi_df['Conversion_Line'] = (high_20 + low_20) / 2

            high_60 = ichi_df['High'].rolling(bl_period).max()
            low_60 = ichi_df['Low'].rolling(bl_period).min()

            ichi_df['Base_Line'] = (high_60 + low_60) / 2

            ichi_df['Lead_span_A'] = (
                (ichi_df['Conversion_Line'] + ichi_df['Base_Line']) / 2).shift(lag_period)

            high_120 = ichi_df['High'].rolling(lead_b_period).max()
            low_120 = ichi_df['Low'].rolling(lead_b_period).min()

            ichi_df['Lead_span_B'] = (
                (high_120 + low_120) / 2).shift(lead_b_period)

            ichi_df['Lagging_span'] = ichi_df['Close'].shift(-lag_period)

            ichi_df = ichi_df.dropna()
            print('ichi_df:_______--- \n', ichi_df)

            # Ichimoku Cloud Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                cl = ichi_df[['Close']].groupby(ichi_df['DateTime']).sum()
                co = ichi_df[['Conversion_Line']].groupby(ichi_df['DateTime']).sum()
                ba = ichi_df[['Base_Line']].groupby(ichi_df['DateTime']).sum()
                la = ichi_df[['Lead_span_A']].groupby(ichi_df['DateTime']).sum()
                lb = ichi_df[['Lead_span_B']].groupby(ichi_df['DateTime']).sum()
                ls = ichi_df[['Lagging_span']].groupby(ichi_df['DateTime']).sum()

                cl.plot(kind='line', linestyle='-', linewidth=0.5,
                        ax=ax, color='#322e2f', fontsize=5)
                co.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#711515', fontsize=5)
                ba.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#2a7115', fontsize=5)
                la.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#156f71', fontsize=5)
                lb.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#201571', fontsize=5)
                ls.plot(kind='line', linestyle=':', linewidth=0.5,
                        ax=ax, color='#71154c', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Ichimoku Cloud--')

            except Exception as e:
                print("Error Ichimoku Graph: ", e)

            return ichi_df

        except Exception as e:
            print("Error in Ichimoku: ", e)

    def _psar(self, acceleration=0.02, maximum=0.2):
        """
        Parabolic SAR (PSAR)
        """

        try:
            df = self._candle_dict()
            psar = df.copy()

            high = psar['High']
            low = psar['Low']

            psar['SAR'] = talib.SAR(
                high, low, acceleration=acceleration, maximum=maximum)

            psar = psar.dropna()
            # print('psar: ______---\n', psar)

            # Parabolic SAR Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                cl = psar[['Close']].groupby(psar['DateTime']).sum()
                sa = psar[['SAR']].groupby(psar['DateTime']).sum()

                cl.plot(kind='line', linestyle='-', linewidth=0.5,
                        ax=ax, color='#322e2f', fontsize=5)
                sa.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#15686d', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Parabolic SAR--')

            except Exception as e:
                print("Error WIR Graph: ", e)

            return psar['SAR'].iloc[-2]

        except Exception as e:
            print("Error in PSAR: ", e)

    def _kc(self, multiplier=2, kc_period=20, atr_period=10):
        """
        Keltner Channel
        - close_ema, kc_middle, kc_upper, kc_lower
        """
        try:
            df = self._candle_dict()
            kc_df = df.copy()
            atr = self._atr(atr_period)

            close_ema = kc_df['Close'].ewm(kc_period).mean()
            multi_atr = multiplier * atr

            kc_df['kc_middle'] = close_ema
            kc_df['kc_upper'] = close_ema + multi_atr
            kc_df['kc_lower'] = close_ema - multi_atr

            kc_df = kc_df.dropna()
            print('kc_df: \n', kc_df)

            # Keltner Channel Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                cl = kc_df[['Close']].groupby(kc_df['DateTime']).sum()
                uf = kc_df[['kc_upper']].groupby(kc_df['DateTime']).sum()
                lf = kc_df[['kc_lower']].groupby(kc_df['DateTime']).sum()
                mf = kc_df[['kc_middle']].groupby(kc_df['DateTime']).sum()

                cl.plot(kind='line', linestyle='-', linewidth=0.5,
                        ax=ax, color='#322e2f', fontsize=5)
                uf.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#990000', fontsize=5)
                lf.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#0b5394', fontsize=5)
                mf.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#274e13', fontsize=5)

                # ax.xticks(kc_df['DateTime'], rotation=90)
                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Keltner Channel--')

            except Exception as e:
                print("Error KC Graph: ", e)

        except Exception as e:
            print("Error in KC: ", e)

    def _env(self, period=50, constant=0.05):
        """
        Envelopes
        - Simple Moving Average(sma) - Period-20/50
        """
        try:
            df = self._candle_dict()
            env_df = df.copy()

            close_sma = env_df['Close'].rolling(window=period).mean()

            env_df['env_middle'] = close_sma
            env_df['env_upper'] = close_sma + (close_sma * constant)
            env_df['env_lower'] = close_sma - (close_sma * constant)

            env_df = env_df.dropna()
            print('env_df: \n', env_df)

            # Envelopes Graph Plot
            try:
                figure = plt.Figure(figsize=(46, 67), dpi=200)
                ax = figure.add_subplot(111)
                chart_type = FigureCanvasTkAgg(figure, self.root)
                chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

                cl = env_df[['Close']].groupby(env_df['DateTime']).sum()
                uf = env_df[['env_upper']].groupby(env_df['DateTime']).sum()
                lf = env_df[['env_lower']].groupby(env_df['DateTime']).sum()
                mf = env_df[['env_middle']].groupby(env_df['DateTime']).sum()

                cl.plot(kind='line', linestyle='-', linewidth=0.5,
                        ax=ax, color='#322e2f', fontsize=5)
                uf.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#e2d810', fontsize=5)
                lf.plot(kind='line', linestyle='-.', linewidth=0.5,
                        ax=ax, color='#d9138a', fontsize=5)
                mf.plot(kind='line', linestyle='--', linewidth=0.5,
                        ax=ax, color='#12a4d9', fontsize=5)

                ax.legend(loc='lower right', fontsize=5)
                ax.set_title('--Envelopes--')

            except Exception as e:
                print("Error ENV Graph: ", e)

        except Exception as e:
            print("Error in ENV: ", e)

    def _rsi(self):
        """
        Relative Strength Index (RSI)
        - delta, up, down
        - avg_gain, avg_loss, rs
        """

        try:
            df = self._candle_dict()
            rsi_df = df.copy()

            delta = rsi_df['Close'].diff().dropna()
            up, down = delta.copy(), delta.copy()

            up[up < 0] = 0
            down[down > 0] = 0

            avg_gain = up.ewm(com=(self._rsi_length - 1),
                              min_periods=self._rsi_length).mean()
            avg_loss = down.abs().ewm(com=(self._rsi_length - 1),
                                      min_periods=self._rsi_length).mean()

            rs = avg_gain/avg_loss

            rsi_df['RSI'] = (100 - (100 / (1 + rs))).round(2)
            rsi_df = rsi_df.dropna()
            # print('rsi_df: _______---\n', rsi_df)

            # try:
            #   figure = plt.Figure(figsize=(8,9), dpi=100)
            #   ax = figure.add_subplot(111)
            #   chart_type = FigureCanvasTkAgg(figure, self.root)
            #   chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
            #   df = rsi_df[['RSI']].groupby(rsi_df['DateTime']).sum()
            #   df.plot(kind='line', legend=True, ax=ax, fontsize=10)
            #   ax.set_title('RSI')
            # except Exception as e:
            #   print("Error RSI Graph: ", e)

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
            df = self._candle_dict()
            macd_df = df.copy()

            # ewm() -  Exponential Weighted functions
            ema_fast = macd_df['Close'].ewm(span=self._ema_fast).mean()
            ema_slow = macd_df['Close'].ewm(span=self._ema_slow).mean()

            # macd of last finished candle
            macd_df['MACD_Line'] = ema_fast - ema_slow
            macd_df['MACD_Signal'] = macd_df['MACD_Line'].ewm(
                span=self._ema_signal).mean()

            macd_df = macd_df.dropna()
            # print('macd_df: ---------_____\n', macd_df)

            # try:
            #   figure = plt.Figure(figsize=(8,9), dpi=100)
            #   ax = figure.add_subplot(111)
            #   chart_type = FigureCanvasTkAgg(figure, self.root)
            #   chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
            #   df = macd_df[['MACD_Line','MACD_Signal']].groupby(macd_df['DateTime']).sum()
            #   df.plot(kind='line', legend=True, ax=ax, fontsize=10)
            #   ax.set_title('MACD')
            # except Exception as e:
            #   print("Error MACD Graph: ", e)

            return macd_df['MACD_Line'].iloc[-2], macd_df['MACD_Signal'].iloc[-2]

        except Exception as e:
            print("Error in MACD:", e)

    def _check_signal(self):
        """
          Technical strategy check signal:
          For long signal-> 1 || short signal -> -1 || no signal -> 0
        """

        rsi = self._rsi()
        macd_line, macd_signal = self._macd()
        # self._bollinger_band()
        # self._atr()
        # self._adx()
        # self. _disp_in()
        # self._tsi()
        self._renko()
        # self._stoch()
        # self._cci()
        # self._wir()
        # self._ichimoku()
        # self._psar()
        # self._kc()
        # self._env()

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
