from dreaml.dataframe.dataframe import DataFrame
import dreaml
import numpy as np
import json

class TestDataFrame:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def test_single_set(self):
        df = DataFrame()
        M = np.arange(4).reshape(2,2)
        df["row/1","col/1"] = 0
        df["row/1","col/2"] = 1
        df["row/2","col/1"] = 2
        assert not df["row/2","col/2"].empty()
        df["row/2","col/2"].set_matrix(3)
        assert(df["row/","col/"].shape==(2,2))
        assert(df["row/","col/"].get_matrix() == M).all()

        df["a/","a/"].set_matrix(3)
        assert(df["a/","a/"].shape==(1,1))
        assert(df["a/","a/"].get_matrix()==3).all()

    def test_simple_matrix_set(self):
        df = DataFrame()
        M = np.arange(4).reshape(2,2)
        df["row/","col/"] = M
        assert(df["row/","col/"].get_matrix() ==M).all()

        df = DataFrame()
        df["row/1","col/1"] = 0
        df["row/1","col/2"] = 1
        df["row/2","col/1"] = 2
        df["row/2","col/2"] = 3
        assert(df["row/","col/"].get_matrix() ==M).all()

        df["row/","col/"] = 2*M
        assert(df["row/","col/"].get_matrix() ==2*M).all()

    def test_dataframe_empty(self):
        df = DataFrame()
        df["row/1","col/1"] = 0
        assert not df["row/1","col/1"].empty()
        try: 
            df["row/1","col/bad"]
        except KeyError:
            pass
        try: 
            df["row/bad","col/1"]
        except KeyError:
            pass
        try:
            df["row/bad","col/bad"]
        except KeyError:
            pass
        assert not df["row/","col/"].empty()
        assert df["bad/","col/"].empty()
        assert df["row/","bad/"].empty()
        assert df["bad/","bad/"].empty()

    def test_from_matrix(self):
        matrix1 = np.ones((2,3))
        df1 = DataFrame.from_matrix(matrix1, row_labels = ["a","b"])
        df2 = DataFrame.from_matrix(matrix1, col_labels = ["a","b","c"])
        df3 = DataFrame.from_matrix(matrix1,
                                    col_labels = ["a","b", "c"],
                                    row_labels = ["a","b"])

        assert(df1.shape==matrix1.shape)
        assert(df2.shape==matrix1.shape)
        assert(df3.shape==matrix1.shape)
        assert(df1.get_matrix().shape==matrix1.shape)
        assert(df2.get_matrix().shape==matrix1.shape)
        assert(df3.get_matrix().shape==matrix1.shape)
        assert(df1.get_matrix()==matrix1).all()
        assert(df2.get_matrix()==matrix1).all()
        assert(df3.get_matrix()==matrix1).all()

    def test_dataframe_set(self):
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        df = DataFrame()
        M_df = DataFrame.from_matrix(matrix1)
        df[row1,col1].set_dataframe(M_df)
        df[row1,col1].set_matrix(matrix1)
        assert (df[row1,col1].get_matrix()==matrix1).all()

    def test_multi_dataframe_set(self):
        df1 = DataFrame()
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        matrix2 = np.arange(6).reshape(2,3)+2
        matrix3 = np.arange(6).reshape(2,3)+4
        matrix4 = np.arange(6).reshape(2,3)+6
        df1[row1,col1].set_matrix(matrix1)
        df1[row1,col2].set_matrix(matrix2)
        df1[row2,col1].set_matrix(matrix3)
        df1[row2,col2].set_matrix(matrix4)

        df2 = DataFrame()
        df2["parentrow/","parentcol/"] = df1
        assert (df2.get_matrix() == df1.get_matrix()).all()

    def test_matrix_set(self):
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        df = DataFrame()
        M_df = DataFrame.from_matrix(matrix1)
        df[row1,col1].set_matrix(matrix1)
        assert (df[row1,col1].get_matrix()==matrix1).all()

    def test_pwd(self):
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        matrix2 = np.arange(6).reshape(2,3)+4
        df = DataFrame()
        df[row1,col1].set_matrix(matrix1)
        df[row2,col2].set_matrix(matrix2)
        assert(df.pwd() == ("",""))
        assert(df[row1,col2].pwd() == (row1,col2))
        assert(df["row/","col/"].pwd() == ("row/","col/"))
        assert(df["row/","col/"]["path1/",:].pwd() == ("row/path1/","col/"))
        assert(df["row/","col/"][0,1:2]["path1/",:].pwd() ==
            ("row/path1/","col/"))
        assert(df["row/path1/0","col/path2/0"].pwd() ==
            ("",""))
        assert(df["row/","col/"]["path2/0","path1/0"].pwd() ==
            ("row/","col/"))

    def test_dataframe_basic(self):
        print "kdfd"
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        df = DataFrame()

        # After inserting one block, other entries should not work
        df[row1,col1] = DataFrame.from_matrix(matrix1)
        assert (df[row1,col1].get_matrix() == matrix1).all()
        try:
            # missing column
            df[row1,col2].get_matrix()
            raise
        except KeyError:
            pass
        try:
            # missing row
            df[row2,col1].get_matrix()
            raise
        except KeyError:
            pass
        try:
            # missing both
            df[row1,col2].get_matrix()
            raise
        except KeyError:
            pass

        # Try overwriting the written matrix: 
        matrix2 = np.arange(6).reshape(2,3)+4
        df[row1,col1] = DataFrame.from_matrix(matrix2)
        assert (df[row1,col1].get_matrix() == matrix2).all()

        # try indexing with slices
        assert(df[row1,col1][0,:].get_matrix()==matrix2[0,:]).all()

        # try subtracting 1 from a row of dataframe partition
        matrix3 = matrix2
        matrix3[0,:] -= 1

        tmp = df[row1,col1][0,:].get_matrix()
        tmp -= 1
        df[row1,col1][0,:] = DataFrame.from_matrix(tmp)
        assert (df[row1,col1].get_matrix()==matrix3).all()

        # Test setting a single element
        df[1,1] = 5
        assert(df[1,1].get_matrix()[0,0] == 5)
        assert(df[1,1]._is_cached())

        # Test use of set_matrix on a single query
        matrix3 = np.arange(6).reshape(2,3)
        df[row1,col1].set_matrix(matrix3+5)
        assert not df[1,1]._is_cached()
        assert(df[row1,col1].get_matrix()==(matrix3+5)).all()
        assert df[row1,col1]._is_cached()

        # Test set_matrix on a nested query
        matrix4 = np.array([[42]])
        df[row1,col1][0,1].set_matrix(matrix4)
        assert (df[row1,col1][0,1].get_matrix()==matrix4).all()

    def test_dataframe_del(self):
        rows = "row/"
        cols = "col/"
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        matrix1 = np.arange(6).reshape(2,3)
        matrix2 = np.arange(6).reshape(2,3)+10
        matrix3 = np.arange(6).reshape(2,3)+20
        matrix4 = np.arange(6).reshape(2,3)+30
        matrix_row1 = np.hstack([matrix1,matrix2])
        matrix_row2 = np.hstack([matrix3,matrix4])
        matrix_all = np.vstack([matrix_row1,matrix_row2])
        df = DataFrame()
        M_df = DataFrame.from_matrix(matrix1)
        df[row1,col1].set_matrix(matrix1)
        df[row1,col2].set_matrix(matrix2)
        df[row2,col1].set_matrix(matrix3)
        df[row2,col2].set_matrix(matrix4)
        assert (df[rows,cols].get_matrix()==matrix_all).all()
        del df[row1,col1]
        try: 
            df[row1,col1].get_matrix()
            raise
        except KeyError: 
            pass
        print df._row_index
        assert (df[rows,cols].get_matrix()==matrix4).all()


    def test_dataframe_hierarchy(self):
        # Add a new row entry within row directory
        rows = "rows/"
        row1 = "rows/path1/"
        row2 = "rows/path2/"
        cols = "cols/"
        col1 = "cols/p1/"
        col2 = "cols/p2/"

        matrix1 = np.arange(6).reshape(2,3)
        matrix2 = np.arange(6).reshape(2,3)+8

        df = DataFrame()
        df[row1,col1] = DataFrame.from_matrix(matrix1)
        assert (df[row1,col1].get_matrix() == matrix1).all()
        assert (df[rows,col1].get_matrix() == matrix1).all()
        assert df[rows,col1]._is_cached()

        df_both_rows = df[rows,col1]

        df[row2,col1] = DataFrame.from_matrix(matrix2)
        # Previous query should be evicted at this point
        assert df[row1,col1]._is_df_cached()
        assert df[row2,col1]._is_df_cached()

        assert not df_both_rows._is_df_cached()
        print df._cache
        assert not df_both_rows._is_cached()
        assert (df[row2,col1].get_matrix() == matrix2).all()

        # Check fetching everything
        matrix3 = np.vstack([matrix1,matrix2])
        print df[rows,col1].get_matrix()
        print matrix3
        print df[rows,col1]._row_index
        assert (df[rows,col1].shape == matrix3.shape)
        assert (df[rows,col1].get_matrix() == matrix3).all()



    def test_dataframe_evictions(self):
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col/path1/"
        all_rows = "row/"

        matrix1 = np.arange(8).reshape(2,4)
        matrix2 = np.arange(8).reshape(2,4)+2
        df = DataFrame()
        df[row1,col1] = DataFrame.from_matrix(matrix1)
        df[row2,col1] = DataFrame.from_matrix(matrix2)

        # load into cache
        df[row1,col1].get_matrix()
        assert(df[row1,col1]._is_cached())

        df[row2,col1].get_matrix()
        assert(df[row1,col1]._is_cached())
        assert(df[row2,col1]._is_cached())

        # assert that the combined value is correct
        assert (df[all_rows,col1].get_matrix() ==
                np.vstack([matrix1,matrix2])).all()
        
        #assert cache membership
        assert(df[all_rows,col1]._is_cached())
        assert(not df[row1,col1]._is_cached())
        assert(not df[row2,col1]._is_cached())

        # attempt to modified a cached element
        M = df[all_rows,col1].get_matrix()
        M[0,:] = 0
        df[all_rows,col1] = DataFrame.from_matrix(M)

        # Assert the matrix is still in the cache
        assert df[all_rows,col1]._is_cached()
        # Assert that the new value is saved to cache
        assert (df[all_rows,col1].get_matrix()==M).all()

        # Pull an entry and evict all_rows
        M1 = df[row1,col1].get_matrix()
        M2 = df[row2,col1].get_matrix()

        assert df[row1,col1]._is_cached()
        assert df[row2,col1]._is_cached()
        assert not df[all_rows,col1]._is_cached()
        # assert that the pulled values match what they should
        assert (np.vstack([M1,M2])==M).all()

        assert (df[all_rows,col1].get_matrix()==M).all()
        df[all_rows,col1][0:1,:].get_matrix()
        assert (df[all_rows,col1][0:1,:]._is_cached())

        # evict an entry that will change
        row3 = "row/path3/"
        matrix3 = np.arange(8).reshape(2,4)+4
        df[row3,col1].set_matrix(matrix3)
        assert (df[row3,col1].get_matrix()==matrix3).all()
        # assert (df[all_rows,col1].get_matrix()==np.vstack([M,matrix3])).all()

    def test_dataframe_update_propogation(self):
        row1 = "row/path1/"
        row11 = "row/path1/sub1/"
        row12 = "row/path1/sub2/"
        row2 = "row/path2/"
        cols = "col/"
        df = DataFrame()

        matrix1 = np.arange(8).reshape(2,4)

        df[row11,cols] = DataFrame.from_matrix(matrix1)
        df[row2,cols] = dreaml.transformations.Identity(df[row1,cols])
        # Check a few equalities from directory indexing 
        assert (df[row11,cols].get_matrix()==df[row1,cols].get_matrix()).all()
        assert (df[row1,cols].get_matrix()==matrix1).all()
        assert (df[row2,cols].get_matrix()==matrix1).all()

        # validate some simple graph properties
        h1 = df[row1,cols].hash()
        h2 = df[row2,cols].hash()
        assert df._graph.node[h1]["status"] == df.STATUS_GREEN
        assert df._graph.node[h2]["status"] == df.STATUS_GREEN

        matrix2 = np.arange(8).reshape(2,4) + 5
        df[row12,cols] = DataFrame.from_matrix(matrix2)
        # Check that cache entries have been properly invalidated
        assert not df[row1,cols]._is_cached()

        # Check that set matrices are correct
        assert (df[row11,cols].get_matrix() == matrix1).all()
        assert df[row11,cols]._is_cached()
        assert (df[row12,cols].get_matrix() == matrix2).all()
        assert df[row12,cols]._is_cached()

        # check that total matrix is correct
        matrix3 = np.vstack([matrix1,matrix2])
        assert (df[row1,cols].get_matrix()==matrix3).all()

        # assert that the two matrices are equal
        assert (df[row1,cols].shape==df[row2,cols].shape)
        assert (df[row1,cols].get_matrix()==df[row2,cols].get_matrix()).all()
        
        # Attempt modifying a single entry instead of extending

    def test_cache_rows_then_evict_all(self):
        row = "row/"
        col = "col/"

        nrows = 4
        matrix1 = np.arange(8).reshape(nrows,2)
        df = DataFrame()
        df[row,col] = DataFrame.from_matrix(matrix1)

        for i in range(nrows):
            assert (df[i,:].get_matrix()==matrix1[i,:]).all()

        assert (df[:,:].get_matrix() == matrix1).all()

        for i in range(0,nrows,2):
            assert (df[i:i+2,:].get_matrix()==matrix1[i:i+2,:]).all()

        assert (df[:,:].get_matrix() == matrix1).all()

    def test_df_cache(self):
        rows = "row/"
        row1 = "row/path1/"
        row2 = "row/path2/"
        col1 = "col1/"
        col2 = "col2/"
        matrix1 = np.arange(6).reshape(2,3)
        matrix2 = np.arange(6).reshape(2,3)+2
        df = DataFrame()
        df[row1,col1].set_matrix(matrix1)

        # Put rows,col1 into the dataframe cache
        assert (df[rows,col1].get_matrix() == matrix1).all()

        df[row2,col2].set_matrix(matrix2)

        assert (df[rows,col1].shape==(4,3))
        assert (df[rows,col1].get_matrix()==
                np.vstack([matrix1,np.zeros((2,3))])).all()

    def test_nonexistant_query(self):
        row1 = "row/path1/"
        row2 = "row/path2/"
        row3 = "row/path3/"
        col1 = "col/path1/"
        col2 = "col/path2/"
        col3 = "col/path3/"
        matrix1 = np.arange(6).reshape(2,3)
        df = DataFrame()
        df[row1,col1] = DataFrame.from_matrix(matrix1)
        df[row2,col2]
        df[row3,col3] = dreaml.transformations.Identity(df[row1,col1])

        assert df[row2,col2].empty()
        assert df[row2,col3].empty()
        assert df[row3,col2].empty()

        assert not df[row1,col1].empty()
        assert not df[row3,col3].empty()

    def test_extend(self):
        # Test extending an existing partition
        row = "row/"
        col = "col/"
        df = DataFrame()
        matrix1 = np.arange(8).reshape(4,2)
        df[row,col].set_matrix(matrix1)
        assert(df[row,col].shape==(4,2))
        rows = [str(v) for v in range(6)]
        cols = [str(v) for v in range(8)]
        df[row,col]._extend(rows,cols)
        assert(df[row,col].shape==(6,8))
        M = df[row,col].get_matrix()
        assert (M[0:4,0:2]==matrix1).all()
        assert (M[5:,:]==0).all()
        assert (M[:,2:]==0).all()

        assert (df[row,col]._row_index.keys()==rows)
        assert (df[row,col]._col_index.keys()==cols)

        # Test extending an empty partition
        row2 = "row2/"
        col2 = "col2/"
        df[row2,col2]._extend(rows,cols)
        assert (df[row2,col2].shape == (6,8))
        print df[row2,col2].get_matrix()
        assert (df[row2,col2].get_matrix()==np.zeros((6,8))).all()

    def test_json(self):
        row = "row/"
        col = "col/"
        df = DataFrame()
        matrix1 = np.arange(8).reshape(4,2)
        df[row,col].set_matrix(matrix1)
        d = json.loads(df.structure_to_json())
        assert d["rows"][0]["directory"]
        assert d["rows"][0]["query"] == "row/"
        for i in range(4):
            assert (d["rows"][0]["files"][i] == 
                {"directory":False, 'query':str(i)})

        assert d["cols"][0]["directory"]
        assert d["cols"][0]["query"] == "col/"
        for i in range(2):
            assert (d["cols"][0]["files"][i] == 
                {"directory":False, 'query':str(i)})
        print d["row_index"]
        print d["col_index"]

        assert (d["row_index"] == {
            "row/0":0,
            "row/1":0,
            "row/2":0,
            "row/3":0
        })

        assert(d["col_index"] == {
            "col/0":0,
            "col/1":0,
        })
        assert(d["partitions"] == {
            "(0, 0)" : 0
        })

    def test_matrix_property(self):
        row = "row/"
        col = "col/"
        df = DataFrame()
        matrix = np.arange(8).reshape(4,2)
        df[row,col] = matrix

        assert (df.r_matrix == matrix).all()
        assert (df[row,col].r_matrix == matrix).all()

        assert (df[row,col].rw_matrix == matrix).all()

        df[row,col].rw_matrix = np.ones((4,2))
        assert (df[row,col].rw_matrix == np.ones((4,2))).all()

        try: 
            df[row,col].rw_matrix = 2
        except ValueError:
            pass

        df[row,col].rw_matrix[:] = 2
        assert (df[row,col].rw_matrix == 2*np.ones((4,2))).all()

        df[row,col].rw_matrix = 2*matrix
        assert (df[row,col].rw_matrix == 2*matrix).all()


