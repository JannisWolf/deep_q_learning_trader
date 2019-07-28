from unittest import TestCase
from datetime import date as Date
from framework.period import Period
from framework.company import Company
from framework.stock_market_data import StockMarketData


class TestStockMarketData(TestCase):

    def test_stock_market_data_one_company_one_period(self):
        stock_market_data = StockMarketData([Company.A], [Period.TRAINING])

        self.assertIsNotNone(stock_market_data)
        self.assertEqual(stock_market_data.get_number_of_companies(), 1)
        self.assertEqual(stock_market_data.get_row_count(), 12588)
        self.assertEqual(stock_market_data.get_most_recent_trade_day(), Date(2011, 12, 30))
        self.assertEqual(stock_market_data.get_most_recent_price(Company.A), 34.802376)
        self.assertIsNone(stock_market_data.get_most_recent_price(Company.B))

    def test_stock_market_data_two_companies_one_period(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])

        self.assertIsNotNone(stock_market_data)
        self.assertEqual(stock_market_data.get_number_of_companies(), 2)
        self.assertEqual(stock_market_data.get_row_count(), 12588)
        self.assertEqual(stock_market_data.get_most_recent_trade_day(), Date(2011, 12, 30))
        self.assertEqual(stock_market_data.get_most_recent_price(Company.A), 34.802376)
        self.assertEqual(stock_market_data.get_most_recent_price(Company.B), 157.07785)

    def test_stock_market_data_one_company_two_periods(self):
        stock_market_data = StockMarketData([Company.A], [Period.TRAINING, Period.TESTING])

        self.assertIsNotNone(stock_market_data)
        self.assertEqual(stock_market_data.get_number_of_companies(), 1)
        self.assertEqual(stock_market_data.get_row_count(), 13594)
        self.assertEqual(stock_market_data.get_most_recent_trade_day(), Date(2015, 12, 31))
        self.assertEqual(stock_market_data.get_most_recent_price(Company.A), 102.759895)
        self.assertIsNone(stock_market_data.get_most_recent_price(Company.B))

    def test_stock_market_data_two_companies_two_periods(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING, Period.TESTING])

        self.assertIsNotNone(stock_market_data)
        self.assertEqual(stock_market_data.get_number_of_companies(), 2)
        self.assertEqual(stock_market_data.get_row_count(), 13594)
        self.assertEqual(stock_market_data.get_most_recent_trade_day(), Date(2015, 12, 31))
        self.assertEqual(stock_market_data.get_most_recent_price(Company.A), 102.759895)
        self.assertEqual(stock_market_data.get_most_recent_price(Company.B), 128.894089)

    def test_deepcopy_first_n_items(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING, Period.TESTING])

        # test copying zero items
        copy = stock_market_data.deepcopy_first_n_items(0)
        self.assertIsNotNone(copy)
        self.assertNotEqual(stock_market_data, copy)
        self.assertEqual(copy.get_number_of_companies(), 2)
        self.assertEqual(copy.get_row_count(), 0)

        # test copying one item
        copy = stock_market_data.deepcopy_first_n_items(1)
        self.assertIsNotNone(copy)
        self.assertNotEqual(stock_market_data, copy)
        self.assertEqual(copy.get_number_of_companies(), 2)
        self.assertEqual(copy.get_most_recent_trade_day(), Date(1962, 1, 2))
        self.assertEqual(copy.get_most_recent_price(Company.A), 0.059620)
        self.assertEqual(copy.get_most_recent_price(Company.B), 2.192523)

    def test_get_most_recent_trade_day(self):
        """
        Tests: StockMarketData#get_most_recent_trade_day

        Read the stock market data and check if the last available date is determined correctly
        """
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING, Period.TESTING])
        self.assertEqual(stock_market_data.get_most_recent_trade_day(), stock_market_data[Company.A].get_last()[0])

    def test_get_most_recent_price(self):
        """
        Tests: StockMarketData#get_most_recent_price

        Read the stock market data and check if the last available stock price is determined correctly
        """
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING, Period.TESTING])
        self.assertEqual(stock_market_data.get_most_recent_price(Company.A), stock_market_data[Company.A].get_last()[1])

    def test_get_row_count(self):
        stock_market_data = StockMarketData([Company.A, Company.B], [Period.TRAINING, Period.TESTING])
        self.assertEqual(stock_market_data.get_row_count(), stock_market_data[Company.A].get_row_count())
