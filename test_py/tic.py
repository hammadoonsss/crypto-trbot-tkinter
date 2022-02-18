"""
   True Inflation Compensator (USD)
    - timeframe
    - year_ago
    - close
    - FRED:M2
    - FRED:pceplife 
"""

"""
def tic(self):

  i = 14
  sma_length = 20
  year_ago = 0
  timeframe = self.timeframe
  close = self.close

  if timeframe.isdaily:
    year_ago = 365
  elif timeframe.isweekly:
    year_ago = 52

  infl = ((security("FRED:M2", timeframe.period, close)+ security("FRED:pceplife", timeframe.period, close)) / 2)
  draw_infl = close*(1/(infl/infl[i*year_ago]))

"""