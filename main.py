import tkinter as tk
import logging

import os
from dotenv import load_dotenv

load_dotenv()

from connectors.binance_futures import BinanceFuturesClient
from connectors.bitmex import BitmexClient

from interface.root_component import Root

logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':

  
  binance = BinanceFuturesClient(os.getenv('BINANCE_PUBLIC_KEY'),
                                 os.getenv('BINANCE_SECRET_KEY'),
                                 os.getenv('BINANCE_TESTNET')) 

  bitmex = BitmexClient(os.getenv('BITMEX_PUBLIC_KEY'),
                        os.getenv('BITMEX_SECRET_KEY'), 
                        os.getenv('BITMEX_TESTNET'))

  root = Root(binance, bitmex)
  
  root.mainloop()
