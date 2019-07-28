from enum import Enum


class Vote(Enum):
    """
    Represents the vote of an expert on a certain stock
    """
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
