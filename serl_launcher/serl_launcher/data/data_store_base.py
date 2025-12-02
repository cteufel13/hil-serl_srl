"""
IMPORTED FROM AGENT LACE
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Any, Union
from collections import deque
from threading import Lock

from abc import abstractmethod
from threading import Lock


class DataStoreBase:
    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def latest_data_id() -> Any:
        """Return the id of the latest data"""
        pass

    @abstractmethod
    def get_latest_data(self, from_id: Any) -> List[Any]:
        """
        provide the all data from the given id
            :return a list of data
        """
        pass

    @abstractmethod
    def __len__(self):
        """Length of the valid data in the store"""
        pass

    @abstractmethod
    def insert(self, data: Any):
        pass

    def batch_insert(self, batch_data: List[Any]):
        """
        Insert a batch of data by using `insert()`
          :param batch_data: a list of data

        NOTE: override this function to create a more efficient
        implementation for custom data store, e.g. slicing in np
        """
        for data in batch_data:
            self.insert(data)
