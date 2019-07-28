from unittest import TestCase
from experts.perfect_expert import PerfectExpert
from framework.order import OrderType
from framework.portfolio import Portfolio
from framework.company import Company
from framework.period import Period
from framework.stock_market_data import StockMarketData
from traders.trusting_trader import TrustingTrader


class TestTrustingTrader(TestCase):
    def test_create_tt_trader(self):
        expert_a = PerfectExpert(Company.A)
        expert_b = PerfectExpert(Company.B)
        trader = TrustingTrader(expert_a, expert_b, 'test_color', 'test_name')
        self.assertIsNotNone(trader)
        self.assertEqual(trader.get_color(), 'test_color')
        self.assertEqual(trader.get_name(), 'test_name')

    def test_trade_vote_up_stock_a(self):
        expert_a = PerfectExpert(Company.A)
        expert_b = PerfectExpert(Company.B)
        trader = TrustingTrader(expert_a, expert_b, 'test_color', 'test_name')

        portfolio = Portfolio(1000.0, {Company.A: 10, Company.B: 10})
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(1)
        order_list = trader.trade(portfolio, stock_market_data)
        self.assertIsNotNone(order_list)
        self.assertEqual(len(order_list), 2)
        self.assertEqual(order_list[0].type, OrderType.BUY)
        self.assertEqual(order_list[0].company, Company.A)
        self.assertEqual(order_list[0].amount, 28.0)
        self.assertEqual(order_list[1].type, OrderType.SELL)
        self.assertEqual(order_list[1].company, Company.B)
        self.assertEqual(order_list[1].amount, 10.0)

    def test_trade_vote_down_stock_a(self):
        expert_a = PerfectExpert(Company.A)
        expert_b = PerfectExpert(Company.B)
        trader = TrustingTrader(expert_a, expert_b, 'test_color', 'test_name')

        portfolio = Portfolio(1000.0, {Company.A: 10, Company.B: 10})
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING]).deepcopy_first_n_items(4)
        order_list = trader.trade(portfolio, stock_market_data)
        self.assertIsNotNone(order_list)
        self.assertEqual(len(order_list), 2)
        self.assertEqual(order_list[0].type, OrderType.SELL)
        self.assertEqual(order_list[0].company, Company.A)
        self.assertEqual(order_list[0].amount, 10.0)
        self.assertEqual(order_list[1].type, OrderType.SELL)
        self.assertEqual(order_list[1].company, Company.B)
        self.assertEqual(order_list[1].amount, 10.0)
