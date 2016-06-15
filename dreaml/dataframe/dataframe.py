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
from threading import Lock, Thread
import json
from rwlock import ReadWriteLock as RWLock

class DataFrame(object):
    """ The DataFrame class organizes data in a block sparse array format using
    a hierarchical structure. 

    The DataFrame maintains two sets of indices for both the row and the column
    of the DataFrame. Both of these can be indexed via their hierarchical
    structure. 
    """

    # Blue blocks are the parent of the propogation
    STATUS_BLUE="blue"
    # Green blocks are completely done propogating
    STATUS_GREEN="green"
    # Red blocks are stopped and awiting to propogate
    STATUS_RED="red"

    def __init__(self):
        """ Initializes an empty DataFrame. """
        self._row_index = Index()
        self._col_index = Index()
        self._partitions = {}
        self._cache = {}
        self._cache_lock = RWLock()
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
    def _from_csv(cls, filename, header=True, index_col=None):
        """ Load from csv file, setting column names from header and row names
        from entries in indx_col."""
        pass

    @classmethod
    def _from_pandas(cls, pandas_df):
        """ Load from a pandas dataframe, keeping row and column names (but 
        converting to strings when they are not strings). """
        pass

    @classmethod
    def from_matrix(cls, matrix, row_labels=None, col_labels=None,):
        """ Initialize from matrix (2D numpy array or 2D numpy matrix, or
        any type of scipy sparse matrix).  
        
        The DataFrame keeps whatever format the matrix is currently in. 
        If no row or column labels are specified, then the DataFrame defaults to
        numerical labels. 

        Args: 
            matrix: The matrix from which the DataFrame is initialized.
            row_labels: An optional list of labels for the rows of the DataFrame.
            col_labels: An optional list of labels for the columns of the DataFrame.

        Returns:
            A DataFrame containing the input matrix and with row and column
            labels mapping to the rows and columns of the input matrix. If no
            row or column label is specified, we default to numerical labels. 

        Raises:
            ValueError: An error occurred when comparing the shape of the matrix
            with the row and column labels.
        """
        if not matrix.shape>0:
            raise ValueError
        if row_labels==None:
            row_labels = [str(i) for i in range(matrix.shape[0])]
        if col_labels==None:
            col_labels = [str(i) for i in range(matrix.shape[1])]

        if not ((len(row_labels),len(col_labels)) == matrix.shape):
            raise ValueError("Provided row labels and column labels must match\
                the dimensions of the given matrix.")

        df = DataFrame()
        row_id = df._add_rows(row_labels)
        col_id = df._add_cols(col_labels)

        df._partitions[row_id,col_id] = matrix
        return df

    def pwd(self):
        """ Return the working directory of both the row and column index. 

        After a series of hierarchical indexing, pwd will return the current
        working directory given these previous indices. 

        Returns: 
            A two-tuple containing the row-directory and column directory of the
            current DataFrame. 
        """
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

    # def shape(self):
    #     """ Returns the shape of the DataFrame.

    #     Returns: 
    #         A two-tuple containing the number of rows and the number of columns
    #         in the DataFrame. 
    #     """
    #     self._refresh_index()
    #     return (len(self._row_index),len(self._col_index))

    @property
    def shape(self):
        """ Returns the shape of the DataFrame.

        Returns: 
            A two-tuple containing the number of rows and the number of columns
            in the DataFrame. 
        """
        self._refresh_index()
        return (len(self._row_index),len(self._col_index))

    def empty(self):
        """ Return whether the DataFrame has non-zero width or height. 

        This is equivalent to checking if any of the values returned by
        shape are equal to 0. 

        Returns: 
            A boolean indicating whether the DataFrame has 0 size. 

        Note: 
            This ignores the values or presence of of the actual underlying matrices. 
        """
        self._refresh_index()
        n_rows,n_cols = self.shape
        # return n_rows==0 or n_cols==0
        if n_rows==0 or n_cols==0:
            return True

    def set_matrix(self,M,row_labels=None,col_labels=None):
        """ Set the DataFrame's contents to be the matrix M.

        This function uses numerical labels for each dimension in the DataFrame.

        Args: 
            M: The matrix that will be the target of the DataFrame.
            row_labels: (optional) list of labels for the rows of the matrix.
            col_labels: (optional) list of labels for the columns of the matrix.
        """
        if isinstance(M,(int,float,long,complex)):
            M = np.array([[M]])

        if self.empty():
            self.set_dataframe(DataFrame.from_matrix(M,
                                                     row_labels=row_labels,
                                                     col_labels=col_labels))
        else:
            self._refresh_index()
            rows = self._row_index.keys()
            cols = self._col_index.keys()
            self.set_dataframe(DataFrame.from_matrix(M,
                                                     row_labels=rows,
                                                     col_labels=cols))

    def set_dataframe(self,M_df):
        """ Set the DataFrame's contents to match the given DataFrame M_df 

        This function uses the labels present in the given DataFrame as the row
        and column labels. 

        Args: 
            M_df: The DataFrame whose contents will be copied. 
        """
        self._refresh_index()
        if len(self._row_query)>0:
            df,r,c = self._last_query((self._row_query,self._col_query))
            df[r,c] = M_df
        else:
            self.__setitem__((slice(None,None,None),slice(None,None,None)),M_df)

    def get_matrix(self,readonly=False,typ=None):
        """ Return a matrix containing the underlying elements of the DataFrame.

        Args:
            readonly: assumes the data is static, and does not purge conflicts.
                Multiple overlapping read-only matrices can simultaneously exist in the
                cache. 
            type: 
                If a type is specified, then the matrix returned is of that type. 
                If a type is not specified, then the matrix returned inherits the type
                of the upper left most partition of the queried matrix. 
                Currently accepted types are numpy.ndarray and
                scipy.sparse.csr_matrix

        Returns: A matrix whose contents are identical to that of the DataFrame.
        """
        self._refresh_index()
        # This is a useless assert, it is the definition of shape
        # assert((len(self._row_index),len(self._col_index))
        #         == self.shape)

        i_j = (self._row_query,self._col_query)
        if i_j == ((),()):
            i_j = (((None,None,None),),((None,None,None),))

        self._cache_lock.acquire_read()
        try:
            if (i_j) in self._cache:
                # A readonly cache entry can become read-write, but not the other
                # way around (otherwise, read-write entries would not have their
                # changes persist after eviction)
                if self._cache_readonly((i_j)): 
                    self._cache_set_readonly(i_j, readonly)
                A = self._cache_fetch(i_j)
                # if A.shape != self.shape:
                #     print A.shape, self.shape
                # This assert is not always true. If another thread is in the
                # process of adding more rows/columns, then the sizes will mis-match
                # to the cached entry. However, this is OK, since the new
                # rows/columms won't be present in the cache yet. 
                return A
        finally:
            self._cache_lock.release_read()
        # print "cache miss :("+str(i_j)+str(self.shape)

        # If matrix is empty, raise error
        if self.empty():
            raise KeyError

        # # Otherwise purge the cache of related entries and repull from DF
        if not readonly:
            self._safe_cache_find_and_evict(i_j)

        num_rows = len(self._row_index)
        num_cols = len(self._col_index)
        if not (num_rows > 0 and num_cols > 0):
            raise KeyError
        row_id = next(self._row_index.itervalues())
        col_id = next(self._col_index.itervalues())

        if self._is_simple_query():
            A = self._fast_get_matrix()
        else: 
        # This following code is slow, and should only be used when forming
        # matrices from overly complicated queries. Nearly all cases should be
        # handled by the above helper, which assumes a benign sequence of
        # queries. 

        # If the entire dataframe exists in a single partition
        # return a subset of that partition
        # TODO: return an A of the type specified by type
            if all(v[0]==row_id for v in self._row_index.itervalues()) and \
               all(v[0]==col_id for v in self._col_index.itervalues()):
                if (row_id,col_id) in self._partitions:
                    partition = self._partitions[row_id,col_id]
                    row_idx = [[v[1]] for k,v in row_vals]
                    col_idx = [v[1] for k,v in col_vals]
                    A = partition[row_idx,col_idx]
                else:
                    if typ == sp.csr_matrix:
                        A = sp.csr_matrix((num_rows, num_cols))
                    else:
                        A = np.zeros((num_rows, num_cols))
                    self._partitions[row_id,col_id] = A
            else: 
                if typ == sp.csr_matrix:
                    A = sp.csr_matrix((num_rows, num_cols))
                else:
                    A = np.zeros((num_rows, num_cols))
                i=0
                for row_id,row_idx in self._row_index.itervalues():
                    j=0
                    for col_id,col_idx in self._col_index.itervalues():
                        if (row_id,col_id) in self._partitions:
                            A[i,j] = self._partitions[row_id,col_id][row_idx,col_idx]
                        else:
                            if typ == sp.csr_matrix:
                                self._partitions[row_id,col_id] = \
                                    sp.csr_matrix((self._row_counts[row_id],\
                                                   self._col_counts[col_id]))
                            else:
                                self._partitions[row_id,col_id] = \
                                    np.zeros((self._row_counts[row_id],\
                                             self._col_counts[col_id]))
                        j+=1
                    i+=1

            if typ != None:
                if typ == sp.csr_matrix and not sp.issparse(A):
                    A = A.toarray()
                elif typ == np.ndarray and sp.issparse(A):
                    A = sp.csr_matrix(A)

        # Finally, store the cached matrix. Need a lock in case other threads
        # are iterating over the cache
        self._safe_cache_add(i_j, A, readonly=readonly)
        return A

    def _is_simple_query(self):
        """ Return whether the query for this dataframe is simple. Current
        queries designated as simple are single directory queries. 

        Simple queries are contiguous blocks within partitions in their original
        order, allowing for the use of slices as opposed to iteration over the
        elements. 
        """
        if (len(self._row_query)==0 and len(self._col_query)==0):
            return True
        if (len(self._row_query)==1 and
            len(self._col_query)==1 and
            isinstance(self._row_query[0],str) and
            isinstance(self._col_query[0],str)):
            return True

        row_str_slice = (all(isinstance(q,str) for q in self._row_query[:-1])
                         and self._is_encoded_slice(self._row_query[-1]))
        col_str_slice = (all(isinstance(q,str) for q in self._col_query[:-1])
                         and self._is_encoded_slice(self._col_query[-1]))
        if row_str_slice and col_str_slice:
            return True
        return False

    @staticmethod
    def _is_encoded_slice(s):
        return isinstance(s,tuple) and len(s)==3

    def _fast_get_matrix(self):
        """ Return the underlying matrix for the dataframe, assuming the query
        is simple for optimized retrieval. """

        if ((self._row_query[:-1],self._col_query[:-1]) in self._cache
            and self._is_encoded_slice(self._row_query[-1])
            and self._is_encoded_slice(self._col_query[-1])):
            i_j = (self._row_query[:-1],self._col_query[:-1])
            s1 = self._tuple_element_to_query(self._row_query[-1])
            s2 = self._tuple_element_to_query(self._col_query[-1])
            return self._cache_fetch(i_j)[s1,s2]

        i,j = 0,0
        row_it = self._row_index.itervalues()
        col_it = self._col_index.itervalues()
        (rp,ri) = next(row_it)
        (cp,ci) = next(col_it)

        (rp_end,ri_end) = self._row_index[next(reversed(self._row_index))]
        (cp_end,ci_end) = self._col_index[next(reversed(self._col_index))]

        row_list = []
        while(rp != rp_end):
            col_list = []
            while(cp != cp_end):
                p = self._index_partition((rp,cp),
                                          (slice(ri,None,None),
                                           slice(ci,None,None)))
                col_list.append(p)
                col_it = itertools.islice(col_it,p.shape[1]-1,None)
                print cp,ci
                (cp,ci) = next(col_it)
                print cp,ci, p.shape
                assert(ci==0)

            p = self._index_partition((rp,cp),
                                      (slice(ri,None,None),
                                       slice(ci,ci_end+1,None)))
            col_list.append(p)
            col_it = self._col_index.itervalues()
            (cp,ci) = next(col_it)
            if sp.issparse(p):
                row_list.append(sp.hstack(col_list))
            else:
                row_list.append(np.hstack(col_list))

            row_it = itertools.islice(row_it,p.shape[0]-1,None)
            (rp,ri) = next(row_it)
        # last row
        col_list = []
        while(cp != cp_end):
            p = self._index_partition((rp,cp),
                                      (slice(ri,ri_end+1,None),
                                       slice(ci,None,None)))
            col_list.append(p)
            col_it = itertools.islice(col_it, p.shape[1]-1,None)
            (cp,ci) = next(col_it)
            assert(ci==0)
        p = self._index_partition((rp,cp),
                                  (slice(ri,ri_end+1,None),
                                   slice(ci,ci_end+1,None)))
        
        if len(row_list) == 0 and len(col_list) == 0:
            return p

        col_list.append(p)
        if sp.issparse(p):
            row_list.append(sp.hstack(col_list))
            return sp.vstack(row_list)
        else:
            row_list.append(np.hstack(col_list))
            return np.vstack(row_list)

    def _index_partition(self,p_index,m_index):
        if p_index in self._partitions:
            return self._partitions[p_index][m_index]
        else:
            rp,cp = p_index
            return np.zeros((self._row_counts[rp],self._col_counts[cp]))

    def set_structure(self,rows,cols):
        """ Sets the rows and columns labels of the DataFrame to the given lists
        of rows and columns.

        Args: 
            rows: A list of strings that will be set to the rows of this
                DataFrame.
            cols: A list of strings that will be set to the columns of this
                DataFrame. 
        """
        self._refresh_index()
        if self._row_index.keys()==rows and self._col_index.keys()==cols:
            return
        else:
            self._extend(rows,cols)

    def copy_structure(self,df):
        """ Extend the DataFrame to match the structure given in df. 

        Args:
            df: A DataFrame whose hierarchical structure will be copied to the
                current DataFrame
        """
        self._refresh_index()
        df._refresh_index()
        rows = df._row_index.keys()
        cols = df._col_index.keys()
        if self._row_index.keys()==rows and self._col_index.keys()==cols:
            return
        else:
            self._extend(rows,cols)

    def structure_to_json(self):
        """ Produce a JSON object with all the hierarchical structure
        information of the dataframe. 

        Returns:
            A JSON object with properties rows, cols, row_index, col_index, and
            partitions containing the respective information. 
        """
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
        of the dataframe. 

        Returns:
            A JSON object with properties nodes, edges, and implicit containing 
            the respective information (nodes, edges, and implicit edges from
            the dataframe hierarchicalstructure). 
        """
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

    @property
    def T(self):
        """ Return the Transformation that generates this DataFrame. 

        Returns: A transformation T if the target of T is this DataFrame, and
            None otherwise. 
        """
        if self.hash() in self._graph.node:
            return self._graph.node[self.hash()]["transform"]
        else:
            return None

    def status(self):
        """ Return the running status of the DataFrame.

        Returns blue if the DataFrame is currently the root of a propogation,
        red if the DataFrame is waiting to propogate, and green if the DataFrame
        has finished propogating and is otherwise safe to read from. 

        A propogation occurs when underlying structure changes, and causes
        transformations to be re-run. 

        Returns: 
            blue | green | red 
        """
        if self.hash() in self._graph.node:
            return self._graph.node[self.hash()]["status"]
        else:
            return None

    def rows(self):
        """ Return a list of all the rows that index into the DataFrame. """
        return self._row_index.keys()

    def cols(self):
        """ Return a list of all the columns that index into the DataFrame. """
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
    def _tuple_element_to_query(i):
        """ Convert a query to a tuple used to hash into dictionaries. 
            str -> str
            int -> int
            (a,b,c) -> slice(a,b,c) 
        """
        if isinstance(i,str) or isinstance(i,int):
            return i
        elif isinstance(i,tuple) and len(i)==3:
            return slice(*i)
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
            # If the query is a file and not a directory, initialize it if it
            # doesn't already exist
            if (isinstance(i,str) and not i.endswith('/') and 
                i not in top_df._row_index):
                top_df._add_rows([i])
            if (isinstance(j,str) and not j.endswith('/') and 
                j not in top_df._col_index):
                top_df._add_cols([j])

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
        M = val.get_matrix(readonly=self._cache_readonly(node))

        # First check the cache for a fast set. 
        if node in self._cache:
            if self._cache_readonly(node) == True:
                raise UserWarning("Attempting to set a readonly cache block. " \
                                  "The result will not persist. ")
            self._safe_cache_add(node,M,readonly=self._cache_readonly(node))
            return

        # If query is a directory or if the dataframe has no existing entries, 
        # then label the rows according to the input dataframe

        row_prefix,col_prefix = self.pwd()

        if isinstance(i,str):
            if i.endswith('/'):
                rows = [row_prefix+i+k for k in val._row_index.keys()]
            else: 
                rows = [i]
        elif len(self._row_index.subset(i))==0:
            rows = [row_prefix+k for k in val._row_index.keys()]
        # If the rows already exist, and we have a non-string query, then just
        # pull existing row labels
        else:
            rows = [row_prefix+k for k in self._row_index.subset(i).keys()]

        if isinstance(j,str):
            if i.endswith('/'):
                cols = [col_prefix+j+k for k in val._col_index.keys()]
            else: 
                cols = [j]
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

        # Since the cache maintains indices, we need to lock until the indices
        # are updating, the underlying matrix is written, 
        # and cache eviction are done
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
        # since this occurs after adding the rows and columns
        if no_rows_exist or no_cols_exist:
            for k_l in self._get_implicit_dependents(node):
                if self._graph.node[k_l]["status"] != self.STATUS_BLUE:
                    self._propogate_stop(k_l)

        # From here on out we assume its a DataFrame. First we must evict all
        # conflicts, since the matrix is being changed. 
        self._safe_cache_find_and_evict(node)

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
        # self._cache_lock.release()
        # self._cache_add(node,M)
        if no_rows_exist or no_cols_exist:
            for k_l in self._get_implicit_dependents(node):
                if self._graph.node[k_l]["status"] != self.STATUS_BLUE:
                    self._propogate_start(k_l)
        if (node in self._graph.node and
            self._graph.node[node]["status"] == self.STATUS_BLUE):
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
        self._safe_cache_find_and_evict(node)

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
                if (row_id,col_id) not in self._partitions:
                    self._partitions[row_id,col_id] = \
                        np.zeros((self._row_counts[row_id],
                                 self._col_counts[col_id]))
                assert((row_id,col_id) in self._partitions)
                assert((row_idx,col_idx) < self._partitions[row_id,col_id].shape)
                assert((row_val,col_val) < M.shape)
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
        """ Stop the continuous thread that generates this DataFrame. 

        If a thread is running that generates this DataFrame, that thread will
        be stopped as soon as it finishes an iteration. 
        """
        i_j = self.hash()
        if i_j in self._threads and \
           self._graph.node[i_j]["status"] != self.STATUS_RED:
            self._graph.node[i_j]["status"] = self.STATUS_RED
            self._threads[i_j].join()
            del self._threads[i_j]
        else: 
            print "Thread not found!"

    def stop_all(self):
        pass

    def go(self):
        i_j = self.hash()
        if (self.is_transform() and i_j not in self._threads and 
           self._graph.node[i_j]["status"] == self.STATUS_RED):
            # Note: move this to transform.py
            t = Thread(target = self.T._continuous_wrapper, args=(self,))
            self._threads[i_j] = t
            self._graph.node[i_j]["status"] = self.STATUS_GREEN
            t.start()

    def is_transform(self):
        if self.hash() in self._graph.node:
            return self._graph.node[self.hash()]["transform"] is not None
        else: 
            return False

    def status(self):
        if self.is_transform():
            if self.hash() in self._graph.node:
                return self._graph.node[self.hash()]["status"]
            else:
                raise ReferenceError("Transformation does not " \
                                     "exist in the computational graph. ")
        else:
            raise UserWarning("Asked for status of non continuous " \
                              "transform block.")

    def is_running(self):
        return (self.is_transform() 
            and self._graph.node[self.hash()]["status"] != self.STATUS_RED)

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
        if self.hash() in top_df._graph and self.is_transform():
            top_df._refresh(top_df._graph.node[self.hash()]["transform"])
        top_df._df_cache_flush(self.hash()) 
        # If we update params_df here, then it won't update any parents of 
        self._refresh_index()

        self._safe_cache_find_and_evict(self.hash())

    def _df_cache_flush(self,i_j):
        """ Remove all cached dataframe entries that are dependent on i_j """
        if i_j in self._df_cache:
            # del self._df_cache[i_j]
            self._df_cache_del(i_j)
            # self._cache_lock.acquire()
            if i_j in self._cache:
                self._safe_cache_evict(i_j)
            # self._cache_lock.release()
        implicit_dependents = self._get_df_implicit_dependents(i_j)
        for k_l in implicit_dependents:
            if k_l in self._df_cache:
                # Delete from both caches, since the target has changed
                # This order matters, since the eviction needs to write
                # back to the dataframe according to the old value of the df
                # cache. 
                # self._cache_lock.acquire()
                if k_l in self._cache:
                    self._safe_cache_evict(k_l)
                # self._cache_lock.release()
                self._df_cache_del(k_l)

    def _unsafe_df_cache_flush(self,i_j):
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
                # self._cache_lock.acquire()
                if query in self._cache:
                    self._safe_cache_evict(query)
                # self._cache_lock.release()
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
        if (self._graph.node[i_j]["status"] == self.STATUS_RED or
           self._graph.node[i_j]["status"] == self.STATUS_BLUE):
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
            # Only check implicit descendents if this current node was actually
            # rerun. 
            if self._graph.node[i_j]["transform"] is not None:
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

    def _safe_cache_find_and_evict(self,i_j):
        self._cache_lock.acquire_read()
        try:
            evictions = self._cache_find_evictions(i_j)
        finally:
            self._cache_lock.release_read()

        if len(evictions) > 0:
            self._cache_lock.acquire_write()
            try:
                for j_k in evictions:
                    self._cache_evict(j_k)
            finally: 
                self._cache_lock.release_write()

    def _cache_find_evictions(self,i_j): 
        """ Find all cached entries that depend on node i_j """
        # Same logic as get_implicit_dependents but searching the cache instead
        if len(self._cache)==0:
            return set()

        i,j = i_j

        (rows,cols) = self._get_full_rows_and_cols(i_j, ignore_df_cache=True)
        # Return a list of nodes that have a common intersection with i_j

        # self._cache_lock.acquire()
        evictions = set()
        for (i0_j0) in self._cache:
            if DataFrame._node_directory_overlap(i_j,i0_j0):
                (rows0,cols0) = self._get_full_rows_and_cols(i0_j0,ignore_df_cache=True)
                if (any(r in rows for r in rows0) \
                    and any(c in cols for c in cols0)):
                    evictions.add(i0_j0)
        # self._cache_lock.release()
        return evictions

    def _safe_cache_evict(self,i_j):
        self._cache_lock.acquire_write()
        try:
            self._cache_evict(i_j)
        finally:
            self._cache_lock.release_write()


    def _cache_evict(self,i_j):
        """ Evict the matrix for node i_j from the cache, and write the
        cached data through to do the underlying DataFrame. """
        if (i_j in self._cache):
            if(self._cache_readonly(i_j)):
                # If readonly, then just remove entry from cache
                self._cache_del(i_j)
            else:
                # Otherwise we need to rerwite to the underlying df
                M = self._cache_fetch(i_j)
                old_rows = self._cache_rows(i_j)
                old_cols = self._cache_cols(i_j)
                self._cache_del(i_j)

                # Remove from cache before setting in dataframe
                i,j = i_j
                assert(len(i) == len(j))
                df = self._reindex(i_j)        
                df._write_matrix_to(M,old_rows,old_cols)

    def _safe_cache_add(self,i_j,A,readonly=False):
        """ Add the matrix A for node i_j into the cache """
        self._cache_lock.acquire_write()
        self._cache_add(i_j,A,readonly=readonly)
        self._cache_lock.release_write()

    def _cache_add(self,i_j,A,readonly=False):
        """ Add the matrix A for node i_j into the cache """
        df = self._reindex(i_j)
        self._cache[i_j] = (A,
                            df._row_index.keys(),
                            df._col_index.keys(),
                            readonly)

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

    def _cache_readonly(self,i_j):
        if i_j in self._cache:
            return self._cache[i_j][3]
        return False

    def _cache_set_readonly(self, i_j, tf):
        self._cache[i_j] = self._cache[i_j][:3] + (tf,)

    def _df_cache_del(self,i_j):
        """ Delete the entry for i_j in the df cache """
        self._df_cache_lock.acquire()
        if i_j in self._df_cache:
            del self._df_cache[i_j]
        self._df_cache_lock.release()

    def _df_cache_add(self,i_j,df):
        """ Add the entry for i_j in the df cache """
        self._df_cache_lock.acquire()
        self._df_cache[i_j] = df
        self._df_cache_lock.release()