from unittest import TestCase

from framework.order import OrderType
from framework.portfolio import Portfolio
from framework.company import Company
from framework.period import Period
from framework.stock_market_data import StockMarketData
from traders.buy_and_hold_trader import BuyAndHoldTrader


class TestBuyAndHoldTrader(TestCase):
    def test_create_bah_trader(self):
        trader = BuyAndHoldTrader('test_color', 'test_name')
        self.assertIsNotNone(trader)
        self.assertEqual(trader.get_color(), 'test_color')
        self.assertEqual(trader.get_name(), 'test_name')

    def test_trader_no_stock(self):
        trader = BuyAndHoldTrader('test_color', 'test_name')

        portfolio = Portfolio(1000)
        stock_market_data = StockMarketData([], [Period.TESTING])
        order_list = trader.trade(portfolio, stock_market_data)
        self.assertIsNotNone(order_list)
        self.assertEqual(len(order_list), 0)

    def test_trade_one_stock(self):
        trader = BuyAndHoldTrader('test_color', 'test_name')

        portfolio = Portfolio(1000)
        stock_market_data = StockMarketData([Company.A], [Period.TESTING])
        order_list = trader.trade(portfolio, stock_market_data)
        self.assertIsNotNone(order_list)
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].type, OrderType.BUY)
        self.assertEqual(order_list[0].company, Company.A)
        self.assertEqual(order_list[0].amount, 9)

    def test_trade_two_stocks(self):
        trader = BuyAndHoldTrader('test_color', 'test_name')

        portfolio = Portfolio(1000)
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])
        order_list = trader.trade(portfolio, stock_market_data)
        self.assertIsNotNone(order_list)
        self.assertEqual(len(order_list), 2)
        self.assertEqual(order_list[0].type, OrderType.BUY)
        self.assertEqual(order_list[0].company, Company.A)
        self.assertEqual(order_list[0].amount, 4)
        self.assertEqual(order_list[1].type, OrderType.BUY)
        self.assertEqual(order_list[1].company, Company.B)
        self.assertEqual(order_list[1].amount, 3)
