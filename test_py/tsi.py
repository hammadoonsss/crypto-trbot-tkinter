"""
   # True Strength Index (TSI)


def get_tsi(close, long, short, signal):
    diff = close - close.shift(1)
    abs_diff = abs(diff)
    
    diff_smoothed = diff.ewm(span = long, adjust = False).mean()
    diff_double_smoothed = diff_smoothed.ewm(span = short, adjust = False).mean()
    abs_diff_smoothed = abs_diff.ewm(span = long, adjust = False).mean()
    abs_diff_double_smoothed = abs_diff_smoothed.ewm(span = short, adjust = False).mean()
    
    tsi = (diff_double_smoothed / abs_diff_double_smoothed) * 100
    signal = tsi.ewm(span = signal, adjust = False).mean()
    tsi = tsi[tsi.index >= '2020-01-01'].dropna()
    signal = signal[signal.index >= '2020-01-01'].dropna()
    
    return tsi, signal

"""