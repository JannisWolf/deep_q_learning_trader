from enum import Enum
from framework.company import Company


class OrderType(Enum):
    """
    Represents possible stock market order types
    """
    BUY = 1
    SELL = 2


class Order:
    """
    Represents an action to be taken on a portfolio
    """
    type: OrderType
    company: Company
    amount: float

    def __init__(self, type: OrderType, company: Company, amount: float):
        """
        Constructor
    
        Args:
          type: The order type
          company: The company whose stocks this order is about
          amount: The amount of stocks this order is about
        """
        self.type = type
        self.company = company
        self.amount = amount
