from typing import List

from framework.company import Company
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.logger import logger
from framework.portfolio import Portfolio
from framework.stock_data import StockData
from framework.stock_market_data import StockMarketData
from framework.order import Order, OrderType
from framework.vote import Vote


class TrustingTrader(ITrader):
    """
    The trusting traders always follows the advice of the experts.
    If both experts vote on buying stocks, then trusting traders prefers buying stock A rather than buying stock B.
    """

    def __init__(self, expert_a: IExpert, expert_b: IExpert, color: str = 'black', name: str = 'tt_trader'):
        """
        Constructor
        """
        super().__init__(color, name)
        assert expert_a is not None
        assert expert_b is not None
        self.__expert_a = expert_a
        self.__expert_b = expert_b

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        order_list = []

        company_list = stock_market_data.get_companies()

        for company in company_list:
            if company == Company.A:
                stock_data_a = stock_market_data[Company.A]
                vote_a = self.__expert_a.vote(stock_data_a)
                self.__follow_expert_vote(Company.A, stock_data_a, vote_a, portfolio, order_list)
            elif company == Company.B:
                stock_data_b = stock_market_data[Company.B]
                vote_b = self.__expert_b.vote(stock_data_b)
                self.__follow_expert_vote(Company.B, stock_data_b, vote_b, portfolio, order_list)
            else:
                assert False
        return order_list

    def __follow_expert_vote(self, company: Company, stock_data: StockData, vote: Vote, portfolio: Portfolio,
                             order_list: List[Order]):
        assert company is not None
        assert stock_data is not None
        assert vote is not None
        assert portfolio is not None
        assert order_list is not None

        if vote == Vote.BUY:
            # buy as many stocks as possible
            stock_price = stock_data.get_last()[-1]
            amount_to_buy = int(portfolio.cash // stock_price)
            logger.debug(f"{self.get_name()}: Got vote to buy {company}: {amount_to_buy} shares a {stock_price}")
            if amount_to_buy > 0:
                order_list.append(Order(OrderType.BUY, company, amount_to_buy))
        elif vote == Vote.SELL:
            # sell as many stocks as possible
            amount_to_sell = portfolio.get_stock(company)
            logger.debug(f"{self.get_name()}: Got vote to sell {company}: {amount_to_sell} shares available")
            if amount_to_sell > 0:
                order_list.append(Order(OrderType.SELL, company, amount_to_sell))
        else:
            # do nothing
            assert vote == Vote.HOLD
            logger.debug(f"{self.get_name()}: Got vote to hold {company}")
