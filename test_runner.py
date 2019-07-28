import unittest

from framework.test.portfolio_test import TestPortfolio
from framework.test.stock_data_test import TestStockData
from framework.test.stock_exchange_test import TestStockExchange
from framework.test.stock_market_data_test import TestStockMarketData
from traders.test.bah_trader_test import TestBuyAndHoldTrader
from traders.test.dql_trader_test import TestDeepQLearningTrader
#from traders.test.tt_trader_test import TestTrustingTrader

suite = unittest.TestSuite()
suite.addTests(unittest.makeSuite(TestPortfolio))
suite.addTests(unittest.makeSuite(TestStockData))
suite.addTests(unittest.makeSuite(TestStockExchange))
suite.addTests(unittest.makeSuite(TestStockMarketData))
suite.addTests(unittest.makeSuite(TestBuyAndHoldTrader))
suite.addTests(unittest.makeSuite(TestDeepQLearningTrader))
#suite.addTests(unittest.makeSuite(TestTrustingTrader))
unittest.TextTestRunner(verbosity=2).run(suite)
