from typing import Dict, List, Tuple
from framework.period import Period
from framework.logger import logger
from framework.company import Company
from framework.stock_data import StockData
from datetime import date as Date, datetime
import os
from directories import DATASETS_DIR
import numpy


class StockMarketData:
    """
    Represents current and historical stick market data of all companies
    """
    __market_data: Dict[Company, StockData]

    def __init__(self, companies: List[Company] = None, periods: List[Period] = None):
        """
        TODO refactor comment
        Reads the "cross product" from `stocks` and `periods` from CSV files and creates a `StockMarketData` object from
        this. For each defined stock in `stocks` the corresponding value from `Company` is used as logical name. If
        there are `periods` provided those are each read.

        Args:
            stocks: The company names for which to read the stock data. *Important:* These values need to be stated in `CompanyEnum`
            periods: The periods to read. If not empty each period is appended to the filename like this: `[stock_name]_[period].csv`

        Returns:
            The created `StockMarketData` object

        Examples:
            * Preface: Provided stock names are supposed to be part to `CompanyEnum`. They are stated plaintext-ish here to show the point:
            * `(['stock_a', 'stock_b'], ['1962-2011', '2012-2017'])` reads:
                * 'stock_a_1962-2011.csv'
                * 'stock_a_2012-2015.csv'
                * 'stock_b_1962-2011.csv'
                * 'stock_b_2012-2015.csv'
              into a dict with keys `CompanyEnum.COMPANY_A` and `CompanyEnum.COMPANY_B` respectively
            * `(['stock_a'], ['1962-2011', '2012-2017'])` reads:
                * 'stock_a_1962-2011.csv'
                * 'stock_a_2012-2015.csv'
              into a dict with a key `CompanyEnum.COMPANY_A`
            * `(['stock_a', 'stock_b'], ['1962-2011'])` reads:
                * 'stock_a_1962-2011.csv'
                * 'stock_b_1962-2011.csv'
              into a dict with keys `CompanyEnum.COMPANY_A` and `CompanyEnum.COMPANY_B` respectively
            * `(['stock_a', 'stock_b'], [])` reads:
                * 'stock_a.csv'
                * 'stock_b.csv'
              into a dict with keys `CompanyEnum.COMPANY_A` and `CompanyEnum.COMPANY_B` respectively
        """
        data = dict()
        if companies is not None and periods is not None:
            # Read *all* available data
            for company in companies:
                filename = company.value
                if len(periods) is 0:
                    data[company] = StockData(self.__read_stock_market_data([[company, filename]])[company])
                else:
                    period_data = list()
                    for period in periods:
                        period_data.append(self.__read_stock_market_data([[company, ('%s_%s' % (filename, period.value))]]))

                    data[company] = StockData(
                        [item for period_dict in period_data if period_dict is not None for item in period_dict[company]])

        self.__market_data = data

    # todo refactor, das kann ja keiner lesen
    def __read_stock_market_data(self, names_and_filenames: list) -> Dict[Company, List[Tuple[Date, float]]]:
        """
        Reads CSV files from "../`DATASETS_DIR`/`name`.csv" and creates a `StockMarketData` object from this

        Args:
            names_and_filenames: Tuples of filenames and logical names used as dict keys

        Returns:
            A dict. Structure: { CompanyEnum: List[Tuple[dt.datetime.date, float]] }
        """
        # The csv's column keys
        DATE, OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME = range(7)
        data = {}
        #logger.info(f"read names_and_filenames: {names_and_filenames}")

        for company, filename in names_and_filenames:
            filepath = os.path.join(DATASETS_DIR, filename + '.csv')

            if not os.path.exists(filepath):
                continue

            na_portfolio = numpy.loadtxt(filepath, dtype='|S15,f8,f8,f8,f8,f8,i8',
                                         delimiter=',', comments="#", skiprows=1)
            dates = list()
            for day in na_portfolio:
                date = datetime.strptime(day[DATE].decode('UTF-8'), '%Y-%m-%d').date()
                dates.append((date, day[ADJ_CLOSE]))

            data[company] = dates

        return data if len(data) > 0 else None

    def deepcopy_first_n_items(self, n: int):
        """
        Returns a deep copy of this stock market data, trimmed to the first n items.
        :param n:
        :return: StockMarketData object
        """
        assert n >= 0
        dictionary = {}
        for company in self.__market_data:
            company_stock_data = self.__market_data[company].deepcopy_first_n_items(n)
            dictionary[company] = company_stock_data
        deepcopy = StockMarketData()
        deepcopy.__market_data = dictionary
        return deepcopy

    def get_most_recent_trade_day(self):
        """
        Determines the latest trade day of this stock market data

        Returns:
            A `datetime.date` object with the latest trade day
        """
        return next(iter(self.__market_data.values())).get_last()[0]

    def get_trade_days(self):
        """
        Returns list of all contained trade days.
        :return:
        """
        stock_data = next(iter(self.__market_data.values()))
        return stock_data.get_dates()

    def get_most_recent_price(self, company_enum: Company) -> float:
        """
        Determines the latest stock price of the given `company_enum`.
        Returns None if no stock price for the given company was found.

        Args:
            company_enum: The company to determine the stock price of

        Returns:
            The latest `company_enum`'s stock price or None.
        """
        company_data = self.__market_data.get(company_enum)
        if company_data is not None:
            return company_data.get_last()[1]
        else:
            return None

    def get_row_count(self) -> int:
        """
        Determines how many data rows are available for the first company in the underlying stock market data

        Returns:
            The row count
        """
        return next(iter(self.__market_data.values())).get_row_count()

    def __getitem__(self, company: Company) -> StockData:
        """
        Delivers stock data for the given company, or `None` if no data can be found

        Args:
            company: The company to return the data for

        Returns:
            `StockData` object for the given company
        """
        return self.__market_data.get(company)

    def get_number_of_companies(self) -> int:
        """
        Returns number of companies stored in this market data.

        Returns:
            Number of companies as integer.
        """
        return len(self.__market_data)

    def get_companies(self):
        """
        Returns a list of companies stored in this market data

        Returns:
            The list of companies
        """
        return list(self.__market_data.keys())

    def check_data_length(self) -> bool:
        """
        Checks if all underlying stock data lists have the same count. Does this by extracting every
        row count, inserting those numbers into a set and checking if this set has the length of 1

        Returns:
            `True` if all value rows have the same length, `False` if not
        """
        return len(set([stock_data.get_row_count() for stock_data in self.__market_data.values()])) == 1
