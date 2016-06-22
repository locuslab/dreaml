from dreaml.dataframe.dataframe import DataFrame
import dreaml
import numpy as np
import json

class TestDataFrameInternal:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def test_tuple_to_query(self):
        pass