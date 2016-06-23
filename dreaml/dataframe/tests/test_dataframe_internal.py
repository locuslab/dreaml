from dreaml.dataframe.dataframe import DataFrame
import dreaml
import numpy as np
import json

class TestDataFrameInternal:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def test_tuple_to_query(self):
        df = DataFrame()
        # Test conversion of hashable elements to their actual queries
        string = "randomstring"
        slice_hash, slice_actual = (slice,(2,4,1)), slice(2,4,1)
        list_hash, list_actual = (list,(1,2,3,4,5,6)), [1,2,3,4,5,6]

        assert df._tuple_element_to_query(string) == string
        assert df._tuple_element_to_query(slice_hash) == slice_actual
        assert df._tuple_element_to_query(list_hash) == list_actual

        assert df._query_to_tuple_element(string) == string
        assert df._query_to_tuple_element(slice_actual) == slice_hash
        assert df._query_to_tuple_element(list_actual) == list_hash

    def test_setitem(self):
        df = DataFrame()
        rows = "row/"
        cols = "col/"
        M = np.arange(6).reshape(2,3)
        df.__setitem__((slice(None,None,None),slice(None,None,None)), M)
        assert (df.get_matrix()==M).all()

        df = DataFrame()
        df.__setitem__((slice(None,None,None),slice(None,None,None)), M, 
                       rows=["a","b"],cols=["c","d","e"])
        assert (df.get_matrix()==M).all()

        df = DataFrame()
        df["x/","y/"].__setitem__((slice(None,None,None),slice(None,None,None)), M, 
                       rows=["a","b"],cols=["c","d","e"])
        print df.rows(),df.cols()
        assert (df.get_matrix() == M).all()
        assert (df["x/","y/"].get_matrix() == M).all()