from abc import ABC, abstractmethod
from typing import List
from framework.portfolio import Portfolio
from framework.order import Order
from framework.stock_market_data import StockMarketData


class ITrader(ABC):
    """
    Trader interface (abstract base class), that forces traders to have a trade method
    """
    __color: str
    __name: str

    def __init__(self, color: str = 'black', name: str = 'traders interface'):
        assert color is not None
        assert name is not None
        self.__color = color
        self.__name = name

    def get_color(self):
        return self.__color

    def get_name(self):
        return self.__name

    @abstractmethod
    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio: The current Portfolio of this traders
          stock_market_data: The stock market data for evaluation

        Returns:
          A list of orders, may be empty but never `None`
        """
        pass
