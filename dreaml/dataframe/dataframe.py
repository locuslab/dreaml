from index import Index
import itertools
import operator
import numpy as np
import scipy.sparse as sp
from transform import Transform
from collections import OrderedDict
import networkx as nx
from os.path import commonprefix
from time import time
from threading import Lock
import json

class DataFrame(object):
    """ DataFrame class, maintains a sparse block array format. """

    # Blue blocks are the parent of the propogation
    STATUS_BLUE="blue"
    # Green blocks are completely done propogating
    STATUS_GREEN="green"
    # Red blocks are stopped and awiting to propogate
    STATUS_RED="red"

    def __init__(self):
        """ Initializer, create empty dataframe. """
        self._row_index = Index()
        self._col_index = Index()
        self._partitions = {}
        self._cache = {}
        self._cache_lock = Lock()
        self._df_cache = {}
        self._df_cache_lock = Lock()
        self._row_counts = []
        self._col_counts = []

        self._top_df = self
        self._row_query = ()
        self._col_query = ()

        self.hash = lambda: (self._row_query,self._col_query)

        self._graph = nx.DiGraph()
        self._threads = {}
        self._plots = []

    @classmethod
    def from_csv(cls, filename, header=True, index_col=None):
        """ Load from csv file, setting column names from header and row names
        from entries in indx_col."""
        pass

    @classmethod
    def from_pandas(cls, pandas_df):
        """ Load from a pandas dataframe, keeping row and column names (but 
        converting to strings when they are not strings). """
        pass

    @classmethod
    def from_matrix(cls, matrix, row_labels=None, col_labels=None,):
        """ Initialize from matrix (2D numpy array or 2D numpy matrix, or
        any type of scipy sparse matrix).  Keep whatever format the matrix is
        currently in. """
        if not matrix.shape>0:
            raise ValueError
        if row_labels==None:
            row_labels = [str(i) for i in range(matrix.shape[0])]
        if col_labels==None:
            col_labels = [str(i) for i in range(matrix.shape[1])]

        assert((len(row_labels),len(col_labels)) == matrix.shape)

        df = DataFrame()
        row_id = df._add_rows(row_labels)
        col_id = df._add_cols(col_labels)

        df._partitions[row_id,col_id] = matrix
        return df

    def pwd(self):
        """ Return the working directory of both the row and column index """
        # row = ""
        # col = ""
        # for q in self._row_query:
        #     if isinstance(q,str):
        #         row+=q
        # for q in self._col_query:
        #     if isinstance(q,str):
        #         col+=q
        row = DataFrame._concat_strings(self._row_query)
        col = DataFrame._concat_strings(self._col_query)
        return row,col

    @staticmethod
    def _concat_strings(seq):
        s = ""
        for q in seq:
            if isinstance(q,str):
                s += q
        return s

    def shape(self):
        """ Return the shape of the values in the DataFrame """
        self._refresh_index()
        return (len(self._row_index),len(self._col_index))

    def empty(self):
        """ Return whether the DataFrame has non-zero width or height. This
        ignores the initialization status of the actual partitions. """
        self._refresh_index()
        n_rows,n_cols = self.shape()
        # return n_rows==0 or n_cols==0
        if n_rows==0 or n_cols==0:
            return True

    def set_matrix(self,M):
        """ Set the DataFrame's contents to be the matrix M """
        if self.empty():
            self.set_dataframe(DataFrame.from_matrix(M))
        else:
            self._refresh_index()
            rows = self._row_index.keys()
            cols = self._col_index.keys()
            self.set_dataframe(DataFrame.from_matrix(M,
                                                     row_labels=rows,
                                                     col_labels=cols))

    def set_dataframe(self,M_df):
        """ Set the DataFrame's contents to match the given DataFrame M_df """
        self._refresh_index()
        if len(self._row_query)>0:
            df,r,c = self._last_query((self._row_query,self._col_query))
            df[r,c] = M_df
        else:
            self.__setitem__((slice(None,None,None),slice(None,None,None)),M_df)

    def get_matrix(self,readonly=False,type=None):
        """ Dump the dataframe as a matrix by iterating over all the keys. 
        Readonly: assumes the data is static, and does not purge conflicts.
        Multiple overlapping read-only matrices can simultaneously exist in the
        cache. 
        Type: If a type is specified, then the matrix returned is of that type. 
        If a type is not specified, then the matrix returned inherits the type
        of the upper left most partition of the queried matrix. 
        """
        # If entry is cached, return it
        self._refresh_index()

        assert((len(self._row_index),len(self._col_index))
                == self.shape())

        i_j = (self._row_query,self._col_query)
        if i_j == ((),()):
            i_j = (((None,None,None),),((None,None,None),))
        if (i_j) in self._cache:
            A = self._cache_fetch(i_j)
            assert A.shape == self.shape()
            return A

        # If matrix is empty, raise error
        if self.empty():
            raise KeyError

        # # Otherwise purge the cache of related entries and repull from DF
        if not readonly:
            for j_k in self._cache_find_evictions(i_j):
                self._cache_evict(j_k)

        row_vals = self._row_index.values()
        col_vals = self._col_index.values()
        if not (len(row_vals) > 0 and len(col_vals) > 0):
            raise KeyError
        row_id = row_vals[0][0]
        col_id = col_vals[0][0]

        # If the entire dataframe exists in a single partition
        # return a subset of that partition
        # TODO: return an A of the type specified by type
        if all(v[0]==row_id for v in row_vals) and \
           all(v[0]==col_id for v in col_vals):
            if (row_id,col_id) in self._partitions:
                partition = self._partitions[row_id,col_id]
                row_idx = [[v[1]] for v in row_vals]
                col_idx = [v[1] for v in col_vals]
                A = partition[row_idx,col_idx]
            else:
                A = np.zeros((len(row_vals),len(col_vals)))
                self._partitions[row_id,col_id] = A
        else: 
            A = np.zeros((len(row_vals),len(col_vals)))
            i=0
            for row_id,row_idx in row_vals:
                j=0
                for col_id,col_idx in col_vals:
                    if (row_id,col_id) in self._partitions:
                        A[i,j] = self._partitions[row_id,col_id][row_idx,col_idx]
                    else:
                        self._partitions[row_id,col_id] = \
                            np.zeros((self._row_counts[row_id],\
                                     self._col_counts[col_id]))
                    j+=1
                i+=1
        # Finally, store the cached matrix
        self._cache_add(i_j, A)
        return A

    def set_structure(self,rows,cols):
        self._refresh_index()
        if self._row_index.keys()==rows and self._col_index.keys()==cols:
            return
        else:
            self._extend(rows,cols)

    def copy_structure(self,df):
        """ Extend the DataFrame to match the structure given in df """
        self._refresh_index()
        df._refresh_index()
        rows = df._row_index.keys()
        cols = df._col_index.keys()
        if self._row_index.keys()==rows and self._col_index.keys()==cols:
            return
        else:
            self._extend(rows,cols)

    def structure_to_json(self):
        """ Produce a JSON object with all the structural information of the
        dataframe """
        rows = []
        cols = []
        for r in self._row_index:
            self._add_index_to_json_array(rows,r)
        for c in self._col_index:
            self._add_index_to_json_array(cols,c)

        out = {
            "rows": rows,
            "cols": cols,
            "row_index": {k:v[0] for k,v in self._row_index.iteritems()},
            "col_index": {k:v[0] for k,v in self._col_index.iteritems()},
            "partitions": {str(v):u for u,v in enumerate(self._partitions.keys
                ())}
        }
        return json.dumps(out)

    def graph_to_json(self):
        """ Produce a JSON object with all the computational graph information 
        of the dataframe """
        nodes = [DataFrame._query_to_string(n) for n in self._graph.nodes()]
        edges = [(DataFrame._query_to_string(n1),
                  DataFrame._query_to_string(n2))
                  for (n1,n2) in self._graph.edges()]

        implicit = []

        for node in self._graph.nodes():
            implicit += ([(node,e) for e 
                            in self._get_implicit_dependents(node)
                            if node != e])

        implicit = [(DataFrame._query_to_string(n1),
                     DataFrame._query_to_string(n2)) for (n1,n2) in implicit]

        out = {
            "nodes": nodes,
            "edges": edges,
            "implicit": implicit
        }
        return json.dumps(out)

    def graph_to_cytoscape_json(self):
        """ Produce a JSON object with all the computational graph information 
        of the dataframe in the format required for cytoscape"""
        nodes = [{
            "data": {"id": DataFrame._query_to_string(n)}
        } for n in self._graph.nodes()]
        edges = [{
            "data": {
                "source": DataFrame._query_to_string(n1),
                "target": DataFrame._query_to_string(n2),
                "type": "explicit",
                "display": True
                
            }
        } for (n1,n2) in self._graph.edges()]

        implicit = []

        for node in self._graph.nodes():
            for e in self._get_implicit_dependents(node):
                if node != e:
                    if (e,node,True) not in implicit:
                        implicit.append((node,e,True))
                    else:
                        implicit.append((node,e,False))
        
        implicit = [{
            "data":{
                "source": DataFrame._query_to_string(n1),
                "target": DataFrame._query_to_string(n2),
                "type": "implicit",
                "display": display
            }
        } for (n1,n2,display) in implicit]

        out = {
            "nodes": nodes,
            "edges": edges+implicit
        }
        return json.dumps(out)

    def T(self):
        if self.hash() in self._graph.node:
            return self._graph.node[self.hash()]["transform"]
        else:
            return None

    def status(self):
        if self.hash() in self._graph.node:
            return self._graph.node[self.hash()]["status"]
        else:
            return None

    def rows(self):
        return self._row_index.keys()

    def cols(self):
        return self._col_index.keys()

    @staticmethod
    def _add_index_to_json_array(arr,path):
        """ Add a path to an array of hierarchical indices """
        if '/' in path:
            s = path.split('/',1)
            query = s[0]+'/'
            remaining_path = s[1]
            for d in arr:
                if d["query"] == query:
                    DataFrame._add_index_to_json_array(d["files"],remaining_path)
                    return
            l = []
            arr.append({
                "directory": True,
                "query": query,
                "files": l
                })
            DataFrame._add_index_to_json_array(l,remaining_path)
        else:
            arr.append({"directory": False, "query": path})
        
    def _subset(self, i, j):
        """ Return a subset of a DataFrame, just creating new index. """
        subset = DataFrame()
        subset._row_index = self._row_index.subset(i)
        subset._col_index = self._col_index.subset(j)
        subset._row_counts = self._row_counts
        subset._col_counts = self._col_counts

        subset._partitions = self._partitions
        subset._cache = self._cache
        subset._df_cache = self._df_cache
        subset._cache_lock = self._cache_lock
        subset._df_cache_lock = self._df_cache_lock
        
        subset._row_query = self._row_query \
                            + (DataFrame._query_to_tuple_element(i),)
        subset._col_query = self._col_query \
                            + (DataFrame._query_to_tuple_element(j),)
        self.hash = lambda: (DataFrame._row_query,self._col_query)

        subset._top_df = self._top_df

        subset._graph = self._graph
        subset._threads = self._threads
        subset._plots = self._plots
        return subset

    def _skeleton_copy(self):
        """Return an empty index-level skelen copy of the Dataframe with
        cleared caches for the purposes of indexing without caches

        Should only be called on top_df
        """
        df = DataFrame()
        df._row_index = self._row_index
        df._col_index = self._col_index
        df._row_counts = self._row_counts
        df._col_counts = self._col_counts

        df._top_df = df
        df._row_query = self._row_query
        df._col_query = self._col_query
        df.hash = lambda: (self._row_query,self._col_query)
        return df

    def __getitem__(self, i_j_type):
        """ Get a portion of the dataframe, passing row/column indices and an
        optional type parameter.

        The calling convertions for the getitem class are:
            df[row_indexing, col_indexing, type=DataFrame]

        If type == DataFrame, the method will subset the rows and column indices
        based upon the i,j terms, and return a dataframe with the same

        This function currently only tested and coded for dense numpy arrays
        """
        if len(i_j_type) == 2:
            i,j = i_j_type
            typ = DataFrame
        elif len(i_j_type) == 3:
            i,j,typ = i_j_type
        else:
            raise ValueError("Indices must be i,j pairs or i,j,type triplets")

        if typ == DataFrame:
            k_l = (self._row_query + (self._query_to_tuple_element(i),),
                   self._col_query + (self._query_to_tuple_element(j),))
            if k_l in self._df_cache:
                return self._df_cache[k_l]
            else: 
                df_subset = self._subset(i,j)
                self._df_cache_add(k_l,df_subset)
                return df_subset


        # This code doesn't actually run at all. __getitem__ currently just
        # returns the dataframe and ignores type. Type should be moved to
        # get_matrix()

        # Otherwise, actually return the data in the format given by type
        # Should we check for membership of i,j or just let the built in 
        # dict.__getitem__ handle the error for missing keys?
        row_index = self._row_index.subset(i)
        col_index = self._col_index.subset(j)
        row_id = row_index[0][0]
        col_id = col_index[0][0]

        assert(all(row_index[0][0] == v[0] for v in row_index.values()) and \
            all(col_index[0][0] == v[0] for v in col_index.values()))
        row_idx = [[v[1]] for v in row_index.values()]
        col_idx = [v[1] for v in col_index.values()]

        # For now just return the np array
        # TODO: remove this part of the code? 
        return np.array(self._partitions[row_id,col_id] \
                                        [row_idx,col_idx])
    @staticmethod
    def _tuple_element_to_query(i):
        """ Convert a tuple element to an actual query used to index into the 
        DataFrame. 
            str -> str
            int -> int
            (a,b,c) -> slice(a,b,c)
        """
        if isinstance(i,str) or isinstance(i,int):
            return i
        elif isinstance(i,tuple):
            return slice(i[0],i[1],i[2])
        raise ValueError

    @staticmethod
    def _query_to_tuple_element(i):
        """ Convert a query to a tuple used to hash into dictionaries. 
            str -> str
            int -> int
            slice(a,b,c) -> (a,b,c) 
        """
        if isinstance(i,str) or isinstance(i,int):
            return i
        elif isinstance(i,slice):
            return (i.start,i.stop,i.step)
        raise ValueError

    @staticmethod
    def _tuple_element_to_string(i):
        """ Convert a tuple element to a string that visually matches
        the query typed by the user """
        if isinstance(i,str):
            return i
        elif isinstance(i,tuple):
            return str(i[0])+":"+str(i[1])+":"+str(i[2])
        else:
            return str(i)

    @staticmethod
    def _query_to_string(i_j):
        """ Convert an i_j query to a string that matches the query typed by the
        user """
        i,j = i_j
        s = ""
        for k in range(len(i)):
            s += "["
            s += DataFrame._tuple_element_to_string(i[k])
            s += ","
            s += DataFrame._tuple_element_to_string(j[k])
            s += "]"
        return s

    def _reindex(self,i_j,ignore_df_cache=False):
        """ Reindexes into the DataFrame starting from the top level, doing
        nothing if there is no query and performing the last query if there is
        one """
        i,j = i_j
        if len(i)>0:
            df,r,c = self._last_query(i_j,ignore_df_cache)
            return df[r,c]
        else: 
            return self._top_df

    def _last_query(self,i_j,ignore_df_cache=False):
        """ Reindexes into the DataFrame starting from the top level given
        a sequence of row/col queries i_j, returning df,r,c where r,c are the
        last queries and df[r,c] is the resulting DataFrame after the i_j
        queries. There must be at least one index into the DataFrame for this
        call. """
        i,j = i_j
        assert(len(i) == len(j))
        assert(len(i) > 0)
        if ignore_df_cache:
            df = self._top_df._skeleton_copy()
        else: 
            df = self._top_df
        for k in range(len(i)-1):
            row_q = DataFrame._tuple_element_to_query(i[k])
            col_q = DataFrame._tuple_element_to_query(j[k])
            df = df[row_q,col_q]
        return (df, 
                DataFrame._tuple_element_to_query(i[-1]), 
                DataFrame._tuple_element_to_query(j[-1]))

    def _get_full_rows_and_cols(self,i_j,ignore_df_cache=False):
        """Retrieves all rows and columns indices for query i_j. These are
        full rows and columns from the perspective of the top-level DataFrame.
        """
        i,j = i_j
        df_indexed = self._reindex(i_j,ignore_df_cache)

        if df_indexed.empty():
            return ([],[])

        (row_prefix,col_prefix) = df_indexed.pwd()

        rows = [row_prefix+ri for ri in df_indexed._row_index.keys()]
        cols = [col_prefix+ci for ci in df_indexed._col_index.keys()]

        assert(len(rows) > 0 and len(cols) > 0)

        return (rows,cols)


    def _add_cols(self, col_keys):
        """ Add col_keys to _col_index for a new partition, update col_counts,
        and return the partition id.
        """
        col_id = len(self._col_counts)
        self._col_counts.append(len(col_keys))
        col_vals = [(col_id, col_idx) for col_idx in range(len(col_keys))]
        self._col_index[col_keys] = col_vals
        return col_id

    def _add_rows(self, row_keys):
        """ Add row_keys to _row_index for a new partition, update row_counts,
        and return the partition id. 
        """
        row_id = len(self._row_counts)
        self._row_counts.append(len(row_keys))
        row_vals = [(row_id, row_idx) for row_idx in range(len(row_keys))]
        self._row_index[row_keys] = row_vals
        return row_id

    def __setitem__(self, i_j, val):
        """ Set a portion of the dataframe, passing row/column indices to the
        values stored in val. If the input is not a DataFrame, then try to
        convert the input into a DataFrame. 

        If type = Transform, then we evaluate the transform and call
            __setitem__ on the result (within the apply function)
        If type = singleton, then wrap it in a proper dataframe. 
        Otherwise, we case on whether the keys for the columns and rows
        already exist or not. 

        To maintain consistency between sub-DataFrames, all insertions actually
        occur at the top level DataFrame. All child DataFrames refetch their
        row/column indices from the top level DataFrame when necessary via
        refresh_index(). 
        """
        i,j = i_j

        top_df = self._top_df

        # This probably needs to be fixed for setting subsetted dataframes
        node = (self._row_query+(DataFrame._query_to_tuple_element(i),),
                self._col_query+(DataFrame._query_to_tuple_element(j),))

        # If the input is a Transform, evaluate the transform and update the
        # computational graph
        if isinstance(val,Transform):
            # # Right now, run the init and refresh the transform's variables
            # # on every step
            target_df = self._reindex(node)
            # val.apply_init(target_df)
            # self._refresh(val)
            # # Always apply once, so that all entries are initialized
            # val.apply_to(target_df)

            # # Now add to graph, since the entries exist now. 
            # if not val.subroutine:
            self._add_to_graph(node,status=self.STATUS_BLUE,transform=val)

            # # If continuous, spawn the long running process
            # if val.continuous: 
            #     self._threads[node] = val.apply_continuous(target_df)
            thread = val.apply(target_df)
            if thread is not None:
                self._threads[node] = thread

            return

        if isinstance(val,(int, long, float)):
            val = self.from_matrix(np.array([val]).reshape(1,1))

        # Assume our input is one giant matrix
        # assert(len(val._partitions)==1)
        # M = val._partitions[0,0]
        # TODO: slow... should not have to fetch the matriix to set. 
        M = val.get_matrix()

        # First check the cache for a fast set. 
        if node in self._cache:
            self._cache_add(node,M)
            return

        # If query is a directory or if the dataframe has no existing entries, 
        # then label the rows according to the input dataframe

        row_prefix,col_prefix = self.pwd()

        if isinstance(i,str):
            rows = [row_prefix+i+k for k in val._row_index.keys()]
        elif len(self._row_index.subset(i))==0:
            rows = [row_prefix+k for k in val._row_index.keys()]
        # If the rows already exist, and we have a non-string query, then just
        # pull existing row labels
        else:
            rows = [row_prefix+k for k in self._row_index.subset(i).keys()]
        if isinstance(j,str):
            cols = [col_prefix+j+k for k in val._col_index.keys()]
        elif len(self._col_index.subset(j))==0:
            cols = [col_prefix+k for k in val._col_index.keys()]
        else:
            cols = [col_prefix+k for k in self._col_index.subset(j).keys()]
        
        # Require equal sizes for source df and target index
        assert((len(rows),len(cols)) == \
               (len(val._row_index),len(val._col_index)))

        # From here on out, rows and cols are full path names; these go into
        # top_df, not the self dataframe

        # "all in or all out" requirement
        all_rows_exist = all(k in top_df._row_index for k in rows)
        all_cols_exist = all(k in top_df._col_index for k in cols)
        no_rows_exist = all(k not in top_df._row_index for k in rows)
        no_cols_exist = all(k not in top_df._col_index for k in cols)

        assert(all_rows_exist or no_rows_exist)
        assert(all_cols_exist or no_cols_exist)

        # If this is going to change the row/column structure, stop all dependents
        if no_cols_exist: 
            col_id = top_df._add_cols(cols)
            col_ids = [col_id]
        else: 
            col_ids = OrderedDict.fromkeys([v[0] \
                for v in top_df._col_index[cols]]).keys()
        #     col_p_id = self._col_index[cols[0]][0]
        if no_rows_exist:
            row_id = top_df._add_rows(rows)
            row_ids = [row_id]
        else: 
            row_ids = OrderedDict.fromkeys([v[0] \
                for v in top_df._row_index[rows]]).keys()

        # require all partitions to already exist or not exist
        all_pairs = itertools.product(set(row_ids),set(col_ids))
        all_parts_exist = all(pair in self._partitions \
            for pair in all_pairs)
        no_parts_exist = all(pair not in self._partitions \
            for pair in all_pairs)

        if not (all_parts_exist or no_parts_exist):
            raise KeyError

        # If the entries do not exist, then start by
        # stopping all dependencies that do exist
        # all entries should exist at this point
        if no_rows_exist or no_cols_exist:
            for k_l in self._get_implicit_dependents(node):
                if self._graph.node[k_l]["status"] != self.STATUS_BLUE:
                    self._propogate_stop(k_l)
        # From here on out we assume its a DataFrame. First we must evict all
        # conflicts, since the matrix is being changed. 
        for j_k in self._cache_find_evictions(node):
            self._cache_evict(j_k)
        # Manually update the dataframe
        if all_rows_exist and all_cols_exist and all_parts_exist:
            top_df._write_matrix_to(M,rows,cols)
        else:
            # Create a new column index partition
            # Create a new partition block
            cur_row = 0
            for row_id in row_ids:
                cur_col = 0
                for col_id in col_ids:
                    row_count = top_df._row_counts[row_id]
                    col_count = top_df._col_counts[col_id]
                    # If the matrix constitutes the entire partition, just 
                    # set without slicing
                    if (M.shape == (row_count, col_count) \
                        and (cur_row,cur_col) == (0,0)):  
                        self._partitions[row_id,col_id] = M
                    else:
                    # Otherwise, select the parts of the matrix that are
                    # applicable and set them for each partition
                        self._partitions[row_id,col_id] \
                            = M[cur_row:cur_row+top_df._row_counts[row_id], \
                                cur_col:cur_col+top_df._col_counts[col_id]]
                    cur_col += top_df._col_counts[col_id]
                cur_row += top_df._row_counts[row_id]
            
            # need to update df cache
            # It is important this occurs after adding the indices to the 
            # row and column indexes, since the df_cache logic uses the
            # indices to determine overlap. 
            self._df_cache_flush(node)
        # self._cache_add(node,M)
        if no_rows_exist or no_cols_exist:
            for k_l in self._get_implicit_dependents(node):
                if self._graph.node[k_l]["status"] != self.STATUS_BLUE:
                    self._propogate_start(k_l)
        if node in self._graph.node:
            self._graph.node[node]["status"] = self.STATUS_GREEN

    def __delitem__(self,i_j):
        """ Delete the entries at i_j from the dataframe """
        # TODO: delete just rows or just columns
        i,j = i_j
        node = (self._row_query+(DataFrame._query_to_tuple_element(i),),
                self._col_query+(DataFrame._query_to_tuple_element(j),))
        # First stop dependencies:
        for k_l in self._get_implicit_dependents(node):
            self._propogate_stop(k_l)

        # Evict all conflicts
        for j_k in self._cache_find_evictions(node):
            self._cache_evict(j_k)

        df = self[i_j]
        row_prefix,col_prefix = df.pwd()
        full_rows = [row_prefix+v for v in df._row_index.keys() ]
        full_cols = [col_prefix+v for v in df._col_index.keys() ]

        # Also flush the df cache
        # Here, we do this before deletion since the cache relies on the 
        # indices to determine overlap. 
        self._df_cache_flush(node)

        # Then delete the indices
        del self._top_df._row_index[full_rows]
        del self._top_df._col_index[full_cols]

        # Finally restart dependencies
        for k_l in self._get_implicit_dependents(node):
            self._propogate_start(k_l)


    def _write_matrix_to(self,M,rows,cols): 
        """ Directly write a matrix to the specified rows and columns into the
        underlying DataFrame, bypassing all other checks and constructs. 
        If the underlying dataframe is non-initialized, we write it."""
        assert(M.shape == (len(rows),len(cols)))
        row_val = 0
        for (row_id,row_idx) in self._row_index[rows]:
            col_val = 0
            for (col_id,col_idx) in self._col_index[cols]:
                # set it element-wise
                # if (row_id,col_id) not in self._partitions:
                #     self._partitions[row_id,col_id] = \
                #         np.zeros(self._row_counts[row_id],
                #                  self._col_counts[col_id])
                self._partitions[row_id,col_id] \
                                [row_idx,col_idx] \
                                = M[row_val,col_val]
                col_val += 1
            row_val += 1

    ###########################################################################
    # Stack related functions for variable length partitions                  #
    ###########################################################################
    # def _push(self,val,axis=0):


    ###########################################################################
    # Graph related functions                                                 #
    ###########################################################################
    # A few basic rules for graph dependencies: 
    # If A = f(X1,...Xn) then X1 -> A (simple computational dependency)
    # If A is a path and B is a subdirectory, then B -> A
    # The previous can be rephrased as
    # if B is a path and A is a parent directory, then B -> A
    # If A is dirty, then all nodes dependent on A are dirty. 
    # This is equivalent to saying all parent directories and all
    # computational dependencies are dirty, and recursively apply. 

    def stop(self):
        """ Stop the continuous thread associated with the DataFrame """
        i_j = self.hash()
        if i_j in self._threads and \
           self._graph.node[i_j]["status"] != self.STATUS_RED:
            self._graph.node[i_j]["status"] = self.STATUS_RED
            self._threads[i_j].join()
            del self._threads[i_j]
        else: 
            print "Thread not found!"

    def _add_to_graph(self,i_j, status,transform=None):
        """ Add a node to the graph and add all of its explicit edges.
        Explicit edges are dependences from the arguments of the transform to
        the resulting DataFrame. """
        if transform is None:
            # If the input is not a transofrm, simply add it to the graph if it
            # doesn't yet exist. 
            if i_j not in self._graph:
                self._graph.add_node(i_j,status=status,transform=None)
        elif isinstance(transform, Transform):
            # If the input is a transform, store the corresponding transform for
            # later rebuilds. 
            self._graph.add_node(i_j,status=status,transform=transform)
        else:
            raise ValueError

        # If the input is a transform, it has explicit dependencies that need 
        # to be added to the graph
        if transform is not None:
            for x in transform.args:
                if isinstance(x,DataFrame):
                    k_l = (x._row_query,x._col_query)
                    if k_l not in self._graph:
                        self._add_to_graph(k_l,self.STATUS_GREEN,None)
                    if not k_l == i_j:
                        self._graph.add_edge((x._row_query,x._col_query),i_j)
            for k in transform.kwargs:
                v = transform.kwargs[k]
                if isinstance(v,DataFrame):
                    k_l = (v._row_query,v._col_query)
                    if k_l not in self._graph:
                        self._add_to_graph(k_l,self.STATUS_GREEN,None)
                    if not k_l == i_j:
                        self._graph.add_edge((v._row_query,v._col_query),i_j)

    @staticmethod
    def _node_directory_overlap(i_j,k_l):
        """ Return whether the directories are subdirectories of each other. 
        A False return here is definitely False, and we can skip any extra
        work. A true result here is not sufficient to conclude that there is no
        overlap. """
        i,j = i_j
        k,l = k_l
        i_str = DataFrame._concat_strings(i)
        j_str = DataFrame._concat_strings(j)
        k_str = DataFrame._concat_strings(k)
        l_str = DataFrame._concat_strings(l)
        is_subdirectory = lambda a,b: a.startswith(b) or b.startswith(a)
        row_is_subdirectory = is_subdirectory(i_str,k_str)
        col_is_subdirectory = is_subdirectory(j_str,l_str)
        return row_is_subdirectory,col_is_subdirectory
        # if not row_is_subdirectory or not col_is_subdirectory:
        #     return False
        # else:
        #     return True

    def _get_implicit_dependents(self,i_j):
        """ Return all nodes implicitly dependent on i_j """
        i,j = i_j
        (rows,cols) = self._get_full_rows_and_cols(i_j, ignore_df_cache=True)

        dependents = set()
        for (i0_j0) in self._graph.nodes_iter():
            row_dir,col_dir = DataFrame._node_directory_overlap(i_j,i0_j0)
            if row_dir and col_dir:
                (rows0,cols0) = self._get_full_rows_and_cols(i0_j0,
                                                             ignore_df_cache=True)
                if (any(r in rows for r in rows0) \
                    and any(c in cols for c in cols0)):
                    # or self._is_sub_directory(i_j,i0_j0):
                    dependents.add(i0_j0)
        return dependents

    def _get_df_implicit_dependents(self,i_j):
        """ Return all nodes implicitly dependent on i_j in the dataframe
        cache. This differs from get_implicit_dependents: rather than tracking
        overlap within the dataframe, this tracks overlap in either set of
        indicies.

        If j_k shares any rows or columns with i_j, then we add it to the
        returned set. """
        (rows,cols) = self._get_full_rows_and_cols(i_j, ignore_df_cache=True)

        dependents = set()

        self._df_cache_lock.acquire()
        for (i0_j0) in self._df_cache:
            row_dir,col_dir = DataFrame._node_directory_overlap(i_j,i0_j0)
            if row_dir or col_dir:
                (rows0,cols0) = self._get_full_rows_and_cols(i0_j0,
                                                             ignore_df_cache=True)
                if (any(r in rows for r in rows0) \
                    or any(c in cols for c in cols0)):
                    dependents.add(i0_j0)
        self._df_cache_lock.release()
        return dependents

    def _get_explicit_dependents(self,i_j):
        """ Return all nodes explicitly dependent on i_j, as denoted by the 
        computational graph """
        if i_j in self._graph:
            return self._graph.successors(i_j)
        else:
            return []

    def _get_all_dependents(self,i_j):
        """ Return all nodes dependent on i_j, implicit or explicit """
        imp = self._get_implicit_dependents(i_j)
        exp = self._get_explicit_dependents(i_j)
        return list(imp)+exp


    def _extend(self,row_labels,col_labels,typ=np.ndarray):
        """Insert row/col labels in this dataframe that don't yet exist. This
        function requires there to be new labels to be inserted; otherwise it
        should not be called as a no-op. """
        top_df = self._top_df
        row_prefix,col_prefix = self.pwd()

        # Fetch all rows
        full_rows = [row_prefix+v for v in row_labels ]
        full_cols = [col_prefix+v for v in col_labels ]

        # Filter out rows that don't exist in the DataFrame yet. 
        new_full_rows = [v for v in full_rows if v not in top_df._row_index]
        new_full_cols = [v for v in full_cols if v not in top_df._col_index]

        assert len(new_full_rows)>0 or len(new_full_cols)>0

        if len(new_full_rows)>0:
            top_df._add_rows(new_full_rows)
        if len(new_full_cols)>0:
            top_df._add_cols(new_full_cols)
        
        # Get partition ids
        row_ids = [v[0] for v in top_df._row_index[full_rows]]
        col_ids = [v[0] for v in top_df._col_index[full_cols]]

        # for row_id in row_ids:
        #     for col_id in col_ids:
        #         if (row_id,col_id) not in top_df._partitions:
        #             # Currently set to a numpy array of zeros
        #             # TODO: set according to type specified
        #             top_df._partitions[row_id,col_id] \
        #                 = np.zeros((top_df._row_counts[row_id], \
        #                             top_df._col_counts[col_id]))
        if self.hash() in top_df._graph and \
            top_df._graph.node[self.hash()]["transform"] is not None:
            top_df._refresh(top_df._graph.node[self.hash()]["transform"])
        top_df._df_cache_flush(self.hash()) 
        # If we update params_df here, then it won't update any parents of 
        self._refresh_index()

        for j_k in self._cache_find_evictions(self.hash()):
            self._cache_evict(j_k)

    def _df_cache_flush(self,i_j):
        """ Remove all cached dataframe entries that are dependent on i_j """
        if i_j in self._df_cache:
            # del self._df_cache[i_j]
            self._df_cache_del(i_j)
            if i_j in self._cache:
                self._cache_evict(i_j)
        implicit_dependents = self._get_df_implicit_dependents(i_j)
        for k_l in implicit_dependents:
            if k_l in self._df_cache:
                # Delete from both caches, since the target has changed
                # This order matters, since the eviction needs to write
                # back to the dataframe according to the old value of the df
                # cache. 
                if k_l in self._cache:
                    self._cache_evict(k_l)
                self._df_cache_del(k_l)


    def _refresh(self,T): 
        """ Reindex into all the arguments for a given transformation. This
        ensures the sizes are up to date. This may be deprecated with
        refresh_index... """
        args = list(T.args)
        for i,df in enumerate(args):
            if isinstance(df,DataFrame):
                d = self._reindex((df._row_query,df._col_query))
                args[i] = d
                query = (args[i]._row_query,args[i]._col_query)
                if query in self._cache:
                    self._cache_evict(query)
                # I think this is unnecessary: the df has not changed shape here
                # self._df_cache_flush(query)
        T.args = tuple(args)
        for k,df in T.kwargs.iteritems():
            if isinstance(df,DataFrame):
                d = self._reindex((df._row_query,df._col_query))
                T.kwargs[k] = d

    def _refresh_index(self):
        """ Re-index into the dataframe, and update the corresponding row/column
        index and counts. This should be called whenever the DataFrame's size
        has changed. """
        new_df = self._reindex((self._row_query,self._col_query))
        # new_df = d[r,c]
        self._row_index = new_df._row_index
        self._col_index = new_df._col_index
        self._row_counts = new_df._row_counts
        self._col_counts = new_df._col_counts


    def _propogate_stop(self,i_j):
        """ Stop all transformations dependent on node i_j """
        # If node is already stopped, return
        if self._graph.node[i_j]["status"] == self.STATUS_RED:
            return

        implicit_dependents = self._get_implicit_dependents(i_j)
        explicit_dependents = self._graph.successors(i_j)

        self._graph.node[i_j]["status"] = self.STATUS_RED
        if self._graph.node[i_j]["transform"] is not None \
            and i_j in self._threads:
            self._threads[i_j].join()

        for v in implicit_dependents:
            self._propogate_stop(v)
        for v in explicit_dependents:
            self._propogate_stop(v)

    def _propogate_start(self,i_j,ignore=None):
        """ Start all transformations that depend on i_j, if possible. """
        if (self._graph.node[v]["status"] != self.STATUS_RED \
            for v in self._graph.predecessors(i_j)) and \
            self._graph.node[i_j]["status"] == self.STATUS_RED and \
            i_j != ignore:
            implicit_dependents = self._get_implicit_dependents(i_j)
            explicit_dependents = self._graph.successors(i_j)
            # if all parents are green, we can restart if necessary and set
            # self to green
            if self._graph.node[i_j]["transform"] is not None:
                # If its a transform, refresh and start it up
                self._refresh(self._graph.node[i_j]["transform"])
                df,r,c = self._last_query(i_j)
                df[r,c] = self._graph.node[i_j]["transform"]

            # set self to green
            self._graph.node[i_j]["status"] = self.STATUS_GREEN
            # recurse on all children
            for v in implicit_dependents:
                if self._graph.node[v]["status"] == self.STATUS_RED:
                    self._propogate_start(v,ignore)

            for v in explicit_dependents:
                if self._graph.node[v]["status"] == self.STATUS_RED:
                    self._propogate_start(v,ignore)

    ###########################################################################
    # Cache related functions                                                 #
    ###########################################################################
    # cache[i_j][0] is the actual cached entry
    # cache[i_j][1] is the dataframe object for this entry at the time of query
    def _is_cached(self):
        """ Test whether the underlying matrix for the DataFrame is cached """
        i_j = (self._row_query,self._col_query)
        return i_j in self._cache

    def _is_df_cached(self):
        """ Test whether the DataFrame is cached """
        i_j = (self._row_query,self._col_query)
        return i_j in self._df_cache

    def _cache_find_evictions(self,i_j): 
        """ Find all cached entries that depend on node i_j """
        # Same logic as get_implicit_dependents but searching the cache instead
        if len(self._cache)==0:
            return set()

        i,j = i_j

        (rows,cols) = self._get_full_rows_and_cols(i_j, ignore_df_cache=True)
        # Return a list of nodes that have a common intersection with i_j

        self._cache_lock.acquire()
        evictions = set()
        for (i0_j0) in self._cache:
            if DataFrame._node_directory_overlap(i_j,i0_j0):
                (rows0,cols0) = self._get_full_rows_and_cols(i0_j0,ignore_df_cache=True)
                if (any(r in rows for r in rows0) \
                    and any(c in cols for c in cols0)):
                    evictions.add(i0_j0)
        self._cache_lock.release()
        return evictions

    def _cache_evict(self,i_j):
        """ Evict the matrix for node i_j from the cache, and write the
        cached data through to do the underlying DataFrame. """
        self._cache_lock.acquire()
        if (i_j in self._cache):
            M = self._cache_fetch(i_j)
            old_rows = self._cache_rows(i_j)
            old_cols = self._cache_cols(i_j)
            self._cache_del(i_j)

            # Remove from cache before setting in dataframe
            i,j = i_j
            assert(len(i) == len(j))
            df = self._reindex(i_j)        
            df._write_matrix_to(M,old_rows,old_cols)
        self._cache_lock.release()

    def _cache_add(self,i_j,A):
        """ Add the matrix A for node i_j into the cache """
        df = self._reindex(i_j)
        self._cache_lock.acquire()
        self._cache[i_j] = (A,
                            df._row_index.keys(),
                            df._col_index.keys())
        self._cache_lock.release()

    def _cache_del(self,i_j):
        """ Remove node i_j from the cache """
        if i_j in self._cache:
            del self._cache[i_j]

    def _cache_fetch(self,i_j):
        """ Retrieve the matrix for node i_j """
        return self._cache[i_j][0]

    def _cache_rows(self,i_j):
        """ Retrieve the rows for node i_j corresponding to the the cached
        matrix """
        return self._cache[i_j][1]

    def _cache_cols(self,i_j):
        """ Retrieve the cols for node i_j corresponding to the the cached
        matrix """
        return self._cache[i_j][2]

    def _df_cache_del(self,i_j):
        self._df_cache_lock.acquire()
        if i_j in self._df_cache:
            del self._df_cache[i_j]
        self._df_cache_lock.release()

    def _df_cache_add(self,i_j,df):
        self._df_cache_lock.acquire()
        self._df_cache[i_j] = df
        self._df_cache_lock.release()