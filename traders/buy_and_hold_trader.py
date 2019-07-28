from typing import List

from framework.order import Order, OrderType
from framework.portfolio import Portfolio
from framework.interface_trader import ITrader
from framework.stock_market_data import StockMarketData


class BuyAndHoldTrader(ITrader):
    """
    BuyAndHoldTrader buys 50% stock A and 50% stock B and holds them over time
    """

    def __init__(self, color: str = 'black', name: str = 'bah_trader'):
        """
        Constructor
        """
        super().__init__(color, name)
        self.__bought_stocks = False

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None

        if self.__bought_stocks:
            return []
        else:
            self.__bought_stocks = True

            # Calculate how many cash to spend per company
            company_list = stock_market_data.get_companies()

            # Invest (100 // `len(companies)`)% of cash into each stock
            order_list = []
            for company in company_list:
                available_cash_per_stock = portfolio.cash / len(company_list)
                most_recent_price = stock_market_data.get_most_recent_price(company)
                amount_to_buy = available_cash_per_stock // most_recent_price
                order_list.append(Order(OrderType.BUY, company, amount_to_buy))
            return order_list
