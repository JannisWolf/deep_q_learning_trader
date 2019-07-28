from enum import Enum


class Period(Enum):
    """
    Represents a certain time period, from which we use financial (stock) data
    """
    TRAINING = "1962-2011"
    TESTING = "2012-2015"
    EVALUATION = "2016-2017"
