from typing import Dict, List
from datetime import date as Date
from framework.stock_market_data import StockMarketData
from framework.company import Company
from framework.order import OrderType, Order
from framework.logger import logger


class Portfolio:
    """
    Represents portfolio of a client
    """
    cash: float
    stocks: Dict[Company, float]

    def __init__(self, cash: float = 0, stocks: Dict[Company, float] = None):
        """
        Constructor

        Args:
          cash: The portfolio's initial cash level
          stocks: The portfolio's initial list of stocks
        """
        self.cash = cash
        if stocks is None:
            stocks = {}
        self.stocks = stocks

    def get_value(self, stock_market_data: StockMarketData, date: Date = None) -> float:
        """
        Return the value of this portfolio: It is the contained ash plus the value of all contained stocks.
        If no date is given, the most recent trade day from stock market data is used.
        :param stock_market_data: Information about all stock prices
        :param date: The day we want the portfolio value for
        :return: The portfolio value
        """
        assert stock_market_data is not None

        result = self.cash
        for company in self.stocks.keys():
            if date is None:
                price = stock_market_data.get_most_recent_price(company)
            else:
                stock_data = stock_market_data.__getitem__(company)
                price = stock_data.get_price(date)
            result += self.stocks[company] * price
        return result

    def get_stock(self, company: Company) -> float:
        """
        Return the amount of stocks we hold from the given company.
        If the portfolio doesn't hold any stocks of this company, then 0 ist returned
        :param company: The company for which to return the share count
        :return: The amount of shares of the given company
        """
        try:
            return self.stocks[company]
        except KeyError:
            return 0.0

    def update_with_order_list(self, stock_market_data: StockMarketData, orders: List[Order]):
        """
        Update the portfolio by executing all given stock orders simultaneously.
        Executing simultaneously means:
            1) The order in which the stock orders are executed does not matter.
            2) Cash from selling stocks today is only available for buying stocks tomorrow.
        If a stock order couldn't be executed (e.g., not enough cash/stocks available), then that order is skipped.
        :param stock_market_data: Information about all stock prices
        :param orders: The list of all stock orders
        :return:
        """
        assert stock_market_data is not None
        assert orders is not None

        if len(orders) == 0:
            logger.debug("The order list is empty. No portfolio update this time")
            return

        available_cash = self.cash
        current_date = stock_market_data.get_most_recent_trade_day()
        logger.debug(f"Updating portfolio {self}: Available cash on {current_date} is {available_cash}")

        for order in orders:
            # get the infos about the order and the existing stock
            company = order.company
            current_price = stock_market_data.get_most_recent_price(company)
            amount = order.amount
            trade_volume = amount * current_price
            existing_amount = self.get_stock(company)

            if order.type is OrderType.BUY:
                logger.debug(f"Buying {amount} stocks of '{company}' at {current_price} (total {trade_volume})")
                if trade_volume <= available_cash:
                    self.stocks[company] = existing_amount + amount
                    self.cash -= trade_volume
                    available_cash -= trade_volume
                else:
                    logger.debug(f"Not enough cash ({available_cash}) for transaction with volume of {trade_volume}")
            elif order.type is OrderType.SELL:
                logger.debug(f"Selling {amount} stocks of '{company}' at {current_price} (total {trade_volume})")
                if existing_amount >= amount:
                    self.stocks[company] = existing_amount - amount
                    self.cash += trade_volume
                else:
                    logger.debug(f"Not enough stocks ({existing_amount}) for selling {amount} of them")
            else:
                assert False
            logger.debug(f"Resulting available cash after trade: {self.cash}")
