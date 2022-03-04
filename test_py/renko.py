# =============================================================================
# Import OHLCV data and transform it to Renko
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

"""

# Import necesary libraries
import yfinance as yf
from stocktrends import Renko

# Download historical data for required stocks
tickers = ["AMZN","GOOG","MSFT"]
ohlcv_data = {}
hour_data = {}
renko_data = {}

# looping over tickers and storing OHLCV dataframe in dictionary
for ticker in tickers:
    temp = yf.download(ticker,period='1mo',interval='5m')
    temp.dropna(how="any",inplace=True)
    ohlcv_data[ticker] = temp
    temp = yf.download(ticker,period='1y',interval='1h')
    temp.dropna(how="any",inplace=True)
    hour_data[ticker] = temp
    
def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    return df["ATR"]

def renko_DF(DF, hourly_df):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df.drop("Close",axis=1,inplace=True)
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = 3*round(ATR(hourly_df,120).iloc[-1],0)
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df

for ticker in ohlcv_data:
    renko_data[ticker] = renko_DF(ohlcv_data[ticker],hour_data[ticker])
    
"""

# Link to stocktrend github page

# https://github.com/ChillarAnand/stocktrends

"""
def _renko(self):
        
        #   Renko Chart with ATR
        #         - pip install stocktrends
        #         - from stocktrends import Renko
        #         - On Historical Data with different timeframe


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

            return df2.brick_size, renko_df["uptrend"].iloc[-2]

        except Exception as e:
            print("Error in Renko:", e)
"""