from typing import List, Tuple
from datetime import date as Date


class StockData:
    """
    Objects of this class comprise a list of tuples which in turn consist of a mapping between dates (type
    `datetime.date`) and stock prices (type `float`)
    """
    __stock_data: List[Tuple[Date, float]]

    def __init__(self, stock_data: List[Tuple[Date, float]]):
        """
        Constructor

        Args:
            stock_data: A list of tuples with dates and the corresponding stock price.
             Structure: `List[Tuple[datetime.date, float]]`
        """
        self.__stock_data = stock_data

    def deepcopy_first_n_items(self, n: int):
        return StockData(self.__stock_data[:n])

    def get_price(self, current_date: Date) -> float:
        assert current_date is not None
        for (date, price) in self.__stock_data:
            if current_date == date:
                return price
        return 0.0

    def get(self, index: int):
        """
        Returns the `index`th item in the list of stock data

        Args:
            index: The index to get

        Returns:
            A tuple consisting of a date and the corresponding stock price
        """
        return self.__stock_data[index]

    def get_first(self):
        """
        Returns the first item in the list of stock data

        Returns:
            A tuple consisting of a date and the corresponding stock price
        """
        return self.__stock_data[0]

    def get_last(self):
        """
        Returns the last item in the list of stock data

        Returns:
            A tuple consisting of a date and the corresponding stock price
        """
        return self.__stock_data[-1]

    def get_from_offset(self, offset: int):
        """
        Calls `[offset:]` on the list of underlying stock data

        Args:
            offset: The offset to take

        Returns:
            A sub-list
        """
        return self.__stock_data[offset:]

    def get_row_count(self):
        """
        Determines how many data rows are available in the underlying stock market data

        Returns:
            The row count
        """
        return len(self.__stock_data)

    def index(self, item: Tuple[Date, float]):
        """
        Calls `#index` on the underlying list of tuples

        Args:
            item: The item to look up the index for

        Returns:
            The index of the given `item`
        """
        return self.__stock_data.index(item)

    def get_dates(self) -> List[Date]:
        """
        Returns all dates

        Returns:
            All dates as a list of dates
        """
        return [data[0] for data in self.__stock_data]

    def get_values(self) -> List[float]:
        """
        Returns all values

        Returns:
            All values as a list of floats
        """
        return [data[1] for data in self.__stock_data]
