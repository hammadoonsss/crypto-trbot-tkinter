"""
  Ichimoku Cloud
"""

"""
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Read BTC/USD data
df = yf.download('BTC-USD', '2019-01-01', '2021-01-01')

# [*********************100%***********************]  1 of 1 completed

# Define length of Tenkan Sen or Conversion Line
cl_period = 20 

# Define length of Kijun Sen or Base Line
bl_period = 60  

# Define length of Senkou Sen B or Leading Span B
lead_span_b_period = 120  

# Define length of Chikou Span or Lagging Span
lag_span_period = 30  

# Calculate conversion line
high_20 = df['High'].rolling(cl_period).max()
low_20 = df['Low'].rolling(cl_period).min()
df['conversion_line'] = (high_20 + low_20) / 2

# Calculate based line
high_60 = df['High'].rolling(bl_period).max()
low_60 = df['Low'].rolling(bl_period).min()
df['base_line'] = (high_60 + low_60) / 2

# Calculate leading span A
df['lead_span_A'] = ((df.conversion_line + df.base_line) / 2).shift(lag_span_period)

# Calculate leading span B
high_120 = df['High'].rolling(120).max()
low_120 = df['High'].rolling(120).min()
df['lead_span_B'] = ((high_120 + low_120) / 2).shift(lead_span_b_period)

# Calculate lagging span
df['lagging_span'] = df['Close'].shift(-lag_span_period)

# Drop NA values from Dataframe
df.dropna(inplace=True) 

# Add figure and axis objects
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 9))

# Plot Close with index on x-axis with a line thickness of 4
ax.plot(df.index, df['Close'], linewidth=4)

# Plot Leading Span A with index on the shared x-axis
ax.plot(df.index, df['lead_span_A'])

# Plot Leading Span B with index on the sahred x-axis
ax.plot(df.index, df['lead_span_B'])

# Use the fill_between of ax object to specify where to fill
ax.fill_between(df.index, df['lead_span_A'], df['lead_span_B'],
                where=df['lead_span_A'] >= df['lead_span_B'], color='lightgreen')

ax.fill_between(df.index, df['lead_span_A'], df['lead_span_B'],
                where=df['lead_span_A'] < df['lead_span_B'], color='lightcoral')

plt.legend(loc=0)
plt.grid()
plt.show()
"""


"""
# StackOverflow - Ichimoku Formula

# Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
period9_high = pd.rolling_max(high_prices, window=9)
period9_low = pd.rolling_min(low_prices, window=9)
tenkan_sen = (period9_high + period9_low) / 2

# Kijun-sen (Base Line): (26-period high + 26-period low)/2))
period26_high = pd.rolling_max(high_prices, window=26)
period26_low = pd.rolling_min(low_prices, window=26)
kijun_sen = (period26_high + period26_low) / 2

# Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

# Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
period52_high = pd.rolling_max(high_prices, window=52)
period52_low = pd.rolling_min(low_prices, window=52)
senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

# The most current closing price plotted 22 time periods behind (optional)
chikou_span = close_prices.shift(-22) # 22 according to investopedia
"""


"""
# Working Ichimoku Cloud Function

  def _ichimoku2(self):
    
      # Ichimoku Cloud (IC)
      #   - Tenkan-Sen/Conversion Line      -  Period-20/9
      #   - Kijun-Sen/Base Line             -  Period-60/26
      #   - Senkou Sen A/Leading Span A 
      #   - Senkou Sen B/Leading Span B     -  Period-120/52
      #   - Chikou/Lagging Span             -  Period-30/26
  
    cl_period = 9
    bl_period = 26
    lead_b_period = 52
    lag_period = 26

    try:
      df = self._candle_list()
      ichi_df2 = df.copy()

      high_9 = ichi_df2['High'].rolling(cl_period).max()
      low_9 = ichi_df2['Low'].rolling(cl_period).min()

      ichi_df2['Conversion_Line'] = (high_9 + low_9) / 2

      high_26 = ichi_df2['High'].rolling(bl_period).max()
      low_26 = ichi_df2['Low'].rolling(bl_period).min()

      ichi_df2['Base_Line'] = (high_26 + low_26) / 2

      ichi_df2['Lead_span_A'] = ((ichi_df2['Conversion_Line'] + ichi_df2['Base_Line']) / 2).shift(lag_period)

      high_56 = ichi_df2['High'].rolling(lead_b_period).max()
      low_56 = ichi_df2['Low'].rolling(lead_b_period).min()

      ichi_df2['Lead_span_B'] = ((high_56 + low_56) / 2).shift(lead_b_period)

      ichi_df2['Lagging_span'] = ichi_df2['Close'].shift(-lag_period)

      ichi_df2 = ichi_df2.dropna()
      print('ichi_df2: ', ichi_df2)

    except Exception as e:
      print("Error in Ichimoku: ", e)

  self._ichimoku2()

"""