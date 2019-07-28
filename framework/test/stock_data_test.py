from unittest import TestCase
from datetime import date as Date
from framework.stock_data import StockData


def get_test_data():
    return StockData([(Date(2017, 1, 1), 150.0), (Date(2017, 1, 2), 200.0)])


class TestStockData(TestCase):
    def test_deepcopy_first_n_items(self):
        stock_data = get_test_data()

        # copy first 0 items
        copy = stock_data.deepcopy_first_n_items(0)
        self.assertIsNotNone(copy)
        self.assertEqual(copy.get_row_count(), 0)

        # copy first 1 items
        copy = stock_data.deepcopy_first_n_items(1)
        self.assertIsNotNone(copy)
        self.assertEqual(copy.get_row_count(), 1)
        self.assertEqual(copy.get(0), (Date(2017, 1, 1), 150.0))

    def test_get_dates(self):
        stock_data = get_test_data()
        self.assertEqual(stock_data.get_dates(), [Date(2017, 1, 1), Date(2017, 1, 2)])

    def test_get_values(self):
        stock_data = get_test_data()
        self.assertEqual(stock_data.get_values(), [150, 200])
