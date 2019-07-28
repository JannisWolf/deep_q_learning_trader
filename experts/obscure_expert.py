from typing import Dict
from datetime import date as Date
from framework.vote import Vote
from framework.interface_expert import IExpert
from framework.company import Company
from framework.stock_data import StockData
import pickle
import os
from directories import EXPERTS_DIR

OBSCURE_EXPERT_DATA = 'obscure_expert_data/obscure.p'


class ObscureExpert(IExpert):
    """
    This expert gives a vote.
    """
    __company: Company
    __answers: Dict[Company, Dict[Date, Vote]]

    def __init__(self, company: Company):
        """
        Constructor:
            Load all answers.

        Args:
            company: The company whose stock values we should predict.
        """
        assert company is not None
        self.__company = company
        self.__answers = pickle.load(open(os.path.join(EXPERTS_DIR, OBSCURE_EXPERT_DATA), "rb"))

    def vote(self, stock_data: StockData) -> Vote:
        """
        Vote based on the stock's historic prices.
        :param stock_data: StockData object capturing the past stock prices
        :return:
        """
        assert stock_data is not None

        try:
            (current_date, _) = stock_data.get_last()
            return self.__answers[self.__company][current_date]
        except (ValueError, IndexError):
            assert False
