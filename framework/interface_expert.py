from abc import ABC, abstractmethod
from framework.stock_data import StockData
from framework.vote import Vote


class IExpert(ABC):
    """
    Expert interface (abstract base class), that forces experts to have a vote method
    """

    @abstractmethod
    def vote(self, data: StockData) -> Vote:
        """
        The expert votes on the stock of a company, given a company's historical stock data.
    
        Args:
          data: Historical stock data of a company

        Returns:
          A vote, which is either buy, hold, or sell
        """
        pass
