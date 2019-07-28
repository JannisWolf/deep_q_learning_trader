from unittest import TestCase

from datetime import date as Date

import numpy as np

from stock_exchange import StockExchange
from framework.company import Company
from framework.period import Period
from framework.stock_data import StockData
from framework.stock_market_data import StockMarketData
from traders.buy_and_hold_trader import BuyAndHoldTrader


class TestStockExchange(TestCase):
    def test_create_stock_exchange(self):
        stock_exchange = StockExchange()
        self.assertIsNotNone(stock_exchange)

    def test_run_no_stock_market_data(self):
        stock_exchange = StockExchange()
        trader = BuyAndHoldTrader()
        self.assertRaises(AssertionError, stock_exchange.run, None, [trader])

    def test_run_no_trader(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(0)
        self.assertRaises(AssertionError, stock_exchange.run, stock_market_data, None)
        self.assertRaises(AssertionError, stock_exchange.run, stock_market_data, [None])

    def test_run_zero_days(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(0)
        trader = BuyAndHoldTrader()
        self.assertRaises(AssertionError, stock_exchange.run, stock_market_data, [trader])

    def test_run_one_day(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(1)
        trader = BuyAndHoldTrader()
        result = stock_exchange.run(stock_market_data, [trader])

        # test final day
        final_day = stock_market_data.get_most_recent_trade_day()
        self.assertEqual(final_day, Date(2012, 1, 3))

        # test final portfolio
        final_portfolio = result[trader][final_day]
        self.assertIsNotNone(final_portfolio)
        self.assertEqual(final_portfolio.cash, 1000.0)
        self.assertEqual(final_portfolio.get_stock(Company.A), 0)
        self.assertEqual(final_portfolio.get_stock(Company.B), 0)
        self.assertEqual(final_portfolio.get_value(stock_market_data, Date(2012, 1, 3)), 1000)

    def test_run_two_days(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(2)
        trader = BuyAndHoldTrader()
        result = stock_exchange.run(stock_market_data, [trader])

        # test final day
        final_day = stock_market_data.get_most_recent_trade_day()
        self.assertEqual(final_day, Date(2012, 1, 4))

        # test final portfolio
        final_portfolio = result[trader][final_day]
        self.assertIsNotNone(final_portfolio)
        self.assertEqual(final_portfolio.cash, 24.807061999999974)
        self.assertEqual(final_portfolio.get_stock(Company.A), 14)
        self.assertEqual(final_portfolio.get_stock(Company.B), 3)
        self.assertEqual(final_portfolio.get_value(stock_market_data, Date(2012, 1, 3)), 1000)
        self.assertEqual(final_portfolio.get_value(stock_market_data, Date(2012, 1, 4)), 1005.0684910000001)

    def test_run_two_days_incorrect_offset(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(2)
        trader = BuyAndHoldTrader()

        # too small and too big
        self.assertRaises(AssertionError, stock_exchange.run, stock_market_data, [trader], -1)
        self.assertRaises(AssertionError, stock_exchange.run, stock_market_data, [trader], 2)

    def test_run_two_days_correct_offset(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(2)
        trader = BuyAndHoldTrader()
        result = stock_exchange.run(stock_market_data, [trader], 1)

        # test final day
        final_day = stock_market_data.get_most_recent_trade_day()
        self.assertEqual(final_day, Date(2012, 1, 4))

        # test final portfolio
        final_portfolio = result[trader][final_day]
        self.assertIsNotNone(final_portfolio)
        self.assertEqual(final_portfolio.cash, 1000.0)
        self.assertEqual(final_portfolio.get_stock(Company.A), 0)
        self.assertEqual(final_portfolio.get_stock(Company.B), 0)
        self.assertEqual(final_portfolio.get_value(stock_market_data), 1000.0)

    def test_run_two_days_two_traders(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(2)
        trader1 = BuyAndHoldTrader()
        trader2 = BuyAndHoldTrader()
        result = stock_exchange.run(stock_market_data, [trader1, trader2])

        # test final day
        final_day = stock_market_data.get_most_recent_trade_day()
        self.assertEqual(final_day, Date(2012, 1, 4))

        # test final portfolio1
        final_portfolio1 = result[trader1][final_day]
        self.assertIsNotNone(final_portfolio1)
        self.assertEqual(final_portfolio1.cash, 24.807061999999974)
        self.assertEqual(final_portfolio1.get_stock(Company.A), 14)
        self.assertEqual(final_portfolio1.get_stock(Company.B), 3)
        self.assertEqual(final_portfolio1.get_value(stock_market_data), 1005.0684910000001)

        # test final portfolio2
        final_portfolio2 = result[trader2][final_day]
        self.assertIsNotNone(final_portfolio2)
        self.assertEqual(final_portfolio2.cash, 24.807061999999974)
        self.assertEqual(final_portfolio2.get_stock(Company.A), 14)
        self.assertEqual(final_portfolio2.get_stock(Company.B), 3)
        self.assertEqual(final_portfolio2.get_value(stock_market_data), 1005.0684910000001)

    def test_get_final_portfolio_value(self):
        stock_exchange = StockExchange()
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(2)
        trader1 = BuyAndHoldTrader()
        trader2 = BuyAndHoldTrader()
        stock_exchange.run(stock_market_data, [trader1, trader2])

        # test final portfolio value
        self.assertEqual(stock_exchange.get_final_portfolio_value(trader1), 1005.0684910000001)
        self.assertEqual(stock_exchange.get_final_portfolio_value(trader2), 1005.0684910000001)
