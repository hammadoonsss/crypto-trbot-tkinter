from strategy.strategies import Strategy

from typing import *

from models import *


class BreakoutStrategy(Strategy):

    def __init__(self, client, contract: Contract, exchange: str, timeframe: str,
                 balance_pct: float, take_profit: float, stop_loss: float,
                 other_params: Dict):
        super().__init__(client, contract, exchange, timeframe,
                         balance_pct, take_profit, stop_loss, "Breakout")

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
