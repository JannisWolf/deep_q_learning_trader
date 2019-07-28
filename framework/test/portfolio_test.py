from unittest import TestCase
from datetime import date as Date
from framework.company import Company
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.order import Order, OrderType


class TestPortfolio(TestCase):
    def test_create_portfolio(self):
        # empty portfolio
        portfolio = Portfolio()
        self.assertIsNotNone(portfolio)
        self.assertEqual(portfolio.cash, 0)
        self.assertEqual(portfolio.stocks, {})

        # portfolio with cash
        portfolio = Portfolio(1000.0)
        self.assertIsNotNone(portfolio)
        self.assertEqual(portfolio.cash, 1000.0)
        self.assertEqual(portfolio.stocks, {})

        # portfolio with cash and stocks
        portfolio = Portfolio(1000.0, {Company.A: 10, Company.B: 50})
        self.assertIsNotNone(portfolio)
        self.assertEqual(portfolio.cash, 1000.0)
        self.assertEqual(len(portfolio.stocks.keys()), 2)
        self.assertEqual(portfolio.stocks[Company.A], 10)
        self.assertEqual(portfolio.stocks[Company.B], 50)

    def test_update_no_sufficient_cash_reserve(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        portfolio = Portfolio(0, {Company.A: 200})
        order_list = [Order(OrderType.BUY, Company.A, 100)]

        # Trade volume is too high for current cash reserve. Nothing should happen
        portfolio.update_with_order_list(stock_market_data, order_list)
        self.assertEqual(portfolio.cash, 0)
        self.assertEqual(portfolio.stocks[Company.A], 200)

    def test_update_sufficient_cash_reserve(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        portfolio = Portfolio(20000, {Company.A: 200})
        order_list = [Order(OrderType.BUY, Company.A, 100)]

        # Current cash reserve is sufficient for trade volume. Trade should happen
        portfolio.update_with_order_list(stock_market_data, order_list)
        self.assertEqual(portfolio.cash, 9724.0105)
        self.assertEqual(portfolio.stocks[Company.A], 300)

    def test_update_action_order_does_not_matter(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

        # Create two equal designed portfolios
        portfolio1 = Portfolio(0, {Company.A: 100})
        portfolio2 = Portfolio(0, {Company.A: 100})

        # Create two order lists with the same entries, however in different order
        order_list_1 = [Order(OrderType.BUY, Company.A, 50), Order(OrderType.SELL, Company.A, 100)]
        order_list_2 = [Order(OrderType.SELL, Company.A, 100), Order(OrderType.BUY, Company.A, 50)]

        # Execute the trade action lists on the two portfolios: Sell 100 stocks, skip buying because no cash available
        portfolio1.update_with_order_list(stock_market_data, order_list_1)
        portfolio2.update_with_order_list(stock_market_data, order_list_2)

        # The portfolios should still be equal after applying the actions
        self.assertEqual(portfolio1.cash, 10275.9895)
        self.assertEqual(portfolio1.cash, portfolio2.cash)
        self.assertEqual(portfolio1.stocks[Company.A], 0)
        self.assertEqual(portfolio1.stocks, portfolio2.stocks)

    def test_update_do_not_drop_below_cash_0(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        portfolio = Portfolio(110)

        # Create a order list whose individual actions are within the limit but in sum are over the limit
        # Most recent stock price of stock A is 102.759895
        order_list = [Order(OrderType.BUY, Company.A, 1), Order(OrderType.BUY, Company.A, 1)]
        portfolio.update_with_order_list(stock_market_data, order_list)
        self.assertEqual(portfolio.cash, 7.240105)
        self.assertEqual(portfolio.stocks[Company.A], 1)

    def test_get_value_without_date(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        portfolio = Portfolio()
        self.assertEqual(portfolio.get_value(stock_market_data), 0)

        portfolio = Portfolio(100.0)
        self.assertEqual(portfolio.get_value(stock_market_data), 100.0)
        portfolio = Portfolio(100.0, {Company.A: 10})
        self.assertEqual(portfolio.get_value(stock_market_data), 1127.59895)
        portfolio = Portfolio(100.0, {Company.A: 10, Company.B: 10})
        self.assertEqual(portfolio.get_value(stock_market_data), 2416.5398400000004)

    def test_get_value_with_date(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        date = Date(2012, 1, 3)

        portfolio = Portfolio(100.0)
        self.assertEqual(portfolio.get_value(stock_market_data, date), 100.0)
        portfolio = Portfolio(100.0, {Company.A: 10})
        self.assertEqual(portfolio.get_value(stock_market_data, date), 455.54107999999997)
        portfolio = Portfolio(100.0, {Company.A: 10, Company.B: 10})
        self.assertEqual(portfolio.get_value(stock_market_data, date), 2046.9924999999998)
