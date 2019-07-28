import copy
from typing import Dict, List
from matplotlib import pyplot
from datetime import date as Date

from experts.obscure_expert import ObscureExpert
from framework.interface_trader import ITrader
from framework.period import Period
from framework.portfolio import Portfolio
from framework.company import Company
from framework.stock_market_data import StockMarketData
from framework.logger import logger
from traders import deep_q_learning_trader
from traders.trusting_trader import TrustingTrader
from traders.buy_and_hold_trader import BuyAndHoldTrader


class StockExchange:
    """
    This class models the stock exchange where all traders to their trades.
    To prevent cheating, the stock exchange is the golden source of truth for traders portfolios.
    """
    __cash: float
    __trader_portfolios: Dict[ITrader, Dict[Date, Portfolio]]
    __complete_stock_market_data: StockMarketData

    def __init__(self, initial_portfolio_cash: float = 1000.0):
        """
        Constructor
        :param initial_portfolio_cash: The initial cash per portfolio
        """
        self.__cash = initial_portfolio_cash
        self.__trader_portfolios = None
        self.__complete_stock_market_data = None

    def run(self, data: StockMarketData, traders: List[ITrader], offset: int = 0) -> Dict[ITrader, Dict[Date, Portfolio]]:
        """
        Runs the stock exchange over the given stock market data for the given traders.
        :param data: The complete stock market data
        :param traders: A list of all traders
        :param offset: The number of trading days which a will be skipped before (!) trading starts
        :return: The main data structure, which stores one portfolio per trade day, for each traders
        """
        assert data is not None
        assert traders is not None

        # initialize the main data structure: Dictionary over traders, that stores each traders's portfolio per day
        # data structure type is Dict[ITrader, Dict[Date, Portfolio]]
        trade_dates = data.get_trade_days()
        assert trade_dates # must not be empty
        assert 0 <= offset < len(trade_dates) # offset must be feasible
        self.__complete_stock_market_data = data
        self.__trader_portfolios = {trader: {trade_dates[offset]: Portfolio(self.__cash)} for trader in traders}

        # iterate over all trade days minus 1, because we don't trade on the last day
        for tick in range(offset, len(trade_dates) - 1):
            logger.debug(f"Stock Exchange: Current tick '{tick}' means today is '{trade_dates[tick]}'")

            # build stock market data until today
            current_stock_market_data = data.deepcopy_first_n_items(tick + 1)

            # iterate over all traders
            for trader in traders:
                # get the traders's order list by giving him a copy (to prevent cheating) of today's portfolio
                todays_portfolio = self.__trader_portfolios[trader][trade_dates[tick]]
                current_order_list = trader.trade(copy.deepcopy(todays_portfolio), current_stock_market_data)

                # execute order list and save the result as tomorrow's portfolio
                tomorrows_portfolio = copy.deepcopy(todays_portfolio)
                tomorrows_portfolio.update_with_order_list(current_stock_market_data, current_order_list)
                self.__trader_portfolios[trader][trade_dates[tick + 1]] = tomorrows_portfolio

        return self.__trader_portfolios

    def get_final_portfolio_value(self, trader: ITrader) -> float:
        """
        Return the final portfolio value for one traders after (!) the stock exchange ran at least once.
        :param trader: The traders whose final portfolio value will be returned
        :return: The traders's final portfolio value
        """
        assert trader is not None
        assert self.__trader_portfolios is not None
        assert self.__complete_stock_market_data is not None
        final_day = self.__complete_stock_market_data.get_most_recent_trade_day()
        final_portfolio = self.__trader_portfolios[trader][final_day]
        return final_portfolio.get_value(self.__complete_stock_market_data)

    def visualize_last_run(self) -> None:
        """
        Visualize all portfolio values of all traders after (!) the stock exchange ran at least once.
        :return: None
        """
        assert self.__trader_portfolios is not None
        assert self.__complete_stock_market_data is not None
        pyplot.figure()
        trader_names = []
        for trader in self.__trader_portfolios:
            portfolios = self.__trader_portfolios[trader]
            keys = portfolios.keys()
            values = [pf.get_value(self.__complete_stock_market_data, date) for date, pf in portfolios.items()]
            pyplot.plot(keys, values, label=trader.get_name(), color=trader.get_color())
            trader_names.append(trader.get_name())
        pyplot.legend(trader_names)
        pyplot.show()


# This main method evaluates all traders over the testing period and visualize the results.
if __name__ == "__main__":
    # Load stock market data for testing period
    stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # create new stock exchange with initial portfolio cash for each traders
    stock_exchange = StockExchange(2000.0)

    # create the traders
    bah_trader = BuyAndHoldTrader()
    tt_trader_obscure = TrustingTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), 'green', 'tt obscure')
    dql_trader = deep_q_learning_trader.DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False, 'red')

    # run the stock exchange over the testing period, with 100 skipped trading days
    stock_exchange.run(stock_market_data, [bah_trader, dql_trader, tt_trader_obscure])

    # visualize the results
    stock_exchange.visualize_last_run()
