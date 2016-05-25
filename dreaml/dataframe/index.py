from collections import OrderedDict
import fnmatch
import numpy as np


class Index(OrderedDict):
    """ Index is a class that handles hierarchical indices used for the rows and 
    columns of a dataframe.  Internally, this is stored as an extension of 
    the OrderedDict class that allows for insertion inside the ordered list, 
    plus a wide range of indexing options.  Dataframes will ultimately
    use these indices to store a sequence of:
        str -> (partition_id, partition_index) tuples
    but this class just implements the basic logic for storing anything using
    this type of index.

    In addition to the ordered dictionary, Index maintains an explicit list
    of keys in order (for interger-based lookups, built on an as-needed basis).
    In the future this can be replaced with a different list structure that
    can provide index-based lookups more efficiently.
    """

    def __init__(self, *args):
        OrderedDict.__init__(self, *args)
        if not all(isinstance(k,str) for k in self):
            raise ValueError("All keys must be strings")
        self._list = None
        self._subset_cache = {}


    def iter_slice(self, s):
        """ 
        Iterate over a slice, stopping if we would loop around 
        The slice stop value that's passed in is included in the
        result (as opposed to a normal slice where it is excluded)
        We return to normal semantics by setting the stop to the next
        """
        root = self._OrderedDict__root

        # if empty, do nothing
        if root[0][2] == root[2]:
            return list()

        if s.step is None or s.step >= 0:
            curr = self._OrderedDict__map[s.start] if s.start else root[1]
            stop = self._OrderedDict__map[s.stop] if s.stop else None
            step = s.step if s.step else 1
            nx = 1
        else:
            curr = self._OrderedDict__map[s.start] if s.start else root[0]
            stop = self._OrderedDict__map[s.stop] if s.stop else None
            step = -s.step
            nx = 0

        if stop and curr[2] == stop[2]:
            return list()

        ret = list()
        not_end = True
        while not_end:
            ret.append(curr[2])
            for _ in xrange(step):
                curr = curr[nx]
                # if there is a stop, and we've reached it, we're done
                if stop and curr[2] == stop[2]:
                    not_end = False
                    break
                # if we've reached the end, we're also done. however if
                # stop was set to something, it means it was prior to the start
                # and so we should return an empty result
                if curr[2] == root[2]:
                    if stop:
                        ret = list()
                    not_end = False
                    break

        return ret


    def refresh_index_list(self):
        """ This rebuilds the self._list if it is None (which we set whenever)
        the list of items changes. """
        if self._list == None:
            self._list = list(self)

    def _get_keys(self, i):
        """ For an indexing i, return a iterable of keys specified by i.

        Args:
            The indexing i can be one of many forms:
            str: return the one element, O(1)
            int: return the element at index i, O(n)
            list or ndarray of:
                str: return specified elements, O(k)
                int: return elements at specified indices, O(n)
                bool: return elements from boolean mask, O(n)
            slice:
                str: elements from i.start to i.stop-1 (by i.step)
                int: elements at indices i.start to i.stops (by i.step)

        Returns:
            iterable (of different possible types) with keys from match
        """
        if isinstance(i, str):
            if i[-1:] == "/":
                ret = list()
                node = self._OrderedDict__root
                node = node[1]
                while node[2] != self._OrderedDict__root[2]:
                    if node[2].find(i) == 0:
                        ret.append(node[2])
                    node = node[1]
                return ret
            else:
                return [i]
        elif isinstance(i, int):
            self.refresh_index_list()
            return [self._list[i]]
        elif isinstance(i, list) or isinstance(i, np.ndarray):
            if isinstance(i[0], str):
                ret = list()
                for one_key in i:
                    if one_key[-1:] == "/":
                        node = self._OrderedDict__root
                        node = node[1]
                        while node[2] != self._OrderedDict__root[2]:
                            if node[2].find(one_key) == 0:
                                ret.append(node[2])
                            node = node[1]
                    else:
                        ret.append(one_key)
                return ret
                    
            elif isinstance(i[0], bool):
                # bools are also ints, but ints are not bools
                # so check for bool first
                if len(self) != len(i):
                    raise KeyError("Bool list length must match index key list")
                return [self._list[j] for j in xrange(len(self)) if i[j]]
            elif isinstance(i[0], int):
                self.refresh_index_list()
                return [self._list[j] for j in i]

        elif isinstance(i,slice):
            # a slice's members are read-only so we construct a new slice
            slice_start = i.start
            slice_stop = i.stop
            slice_step = i.step if i.step else 1
            
            if isinstance(i.start, int):
                self.refresh_index_list()
                """
                going out of bounds is valid with slices
                since we will do lookups in the self._list we need to select
                valid entries
                """
                if slice_start <= ((-1 * len(self._list)) - 1):
                    # start before beginning and step backward, do nothing
                    if slice_step < 0:
                        return list()
                    slice_start = None
                elif slice_start >= len(self._list):
                    # start off the end of the list and step forward, do nothing
                    if slice_step > 0:
                        return list()
                    slice_start = None
                else:
                    slice_start = self._list[slice_start]
            if isinstance(i.stop, int):
                self.refresh_index_list()
                """
                going out of bounds is valid with slices
                since we will do lookups in the self._list we need to select
                valid entries
                """
                if slice_stop <= ((-1 * len(self._list)) - 1):
                    # stop is before beginning and step forward, do nothing
                    if slice_step > 0:
                        return list()
                    slice_stop = None
                elif slice_stop >= (len(self._list)):
                    # stop is after end but steping backward, do nothing
                    if slice_step < 0:
                        return list()
                    slice_stop = None
                else:
                    """
                    A slice includes the start but excludes the stop
                    We tweak the stop value so that it is included when we
                    do the list lookup
                    """
                    slice_stop = self._list[slice_stop]

            i_new = slice(slice_start, slice_stop, slice_step)
            return self.iter_slice(i_new)
        else:
            raise KeyError(i)



    def subset(self, i):
        """ For an indexing i, return the subset of the dataframe"""
        keys = self._get_keys(i)
        if isinstance(i,str):
            # if False:
            if i in self._subset_cache:
                return self._subset_cache[i]
            else:
                truncated_keys = [k[len(i):] for k in keys]
                ss = Index([(tk, dict.__getitem__(self,k)) for (tk,k) 
                            in zip(truncated_keys,keys)])
                self._subset_cache[i] = ss
                return ss
        else:
            return Index([(k, dict.__getitem__(self,k)) for k in keys])

    def __getitem__(self, i):
        """ Get items (as array of values) for indexing i """

        if isinstance(i, list) or isinstance(i, slice):
            return [dict.__getitem__(self,k) for k in self._get_keys(i)]
        else:
            keys = self._get_keys(i)
            if keys == None:
                return []

            if len(keys) == 1:
                return dict.__getitem__(self, keys[0])
            elif len(keys) > 1:
                return [dict.__getitem__(self,k) for k in keys]
            else:
                return []


    def __delitem__(self, i):
        """ Delete items for indexing i. """
        keys = self._get_keys(i)
        if any(k not in self for k in keys):
            raise KeyError(i)
        for k in keys:
            OrderedDict.__delitem__(self, k)

        self._subset_cache = {}
        self._list = None

    def relabel(self, keys):
        """ Relabel a collection of keys (a dictionary of old -> new. """
        assert(all(k in self and v not in self for k,v in keys.iteritems()))
        for old_key in keys.keys():
            new_key = keys[old_key]
            l = self._OrderedDict__map[old_key]
            del self._OrderedDict__map[old_key]
            self._OrderedDict__map[new_key] = l
            l[2] = new_key
            dict.__setitem__(self, new_key, dict.pop(self, old_key))


    def __setitem__(self, i, vals):
        """ Set items for indexing i. 

        If the indexing i exists in the Index __setitem__ will set the 
        corresponding values for each key. The number of keys must therefore
        match the number of vals unless i is a singleton; in that case
        the key will be set to the entirety of vals. Note that if i is a list
        then vals must also be a list. If i is a singleton, vals can be anything
        If i is a singleton or a list, and if none of the keys exists in the 
        Index, then we add (append) them to the Index.
        """
        # For now, just reset the entire cache
        self._subset_cache = {}
        keys = self._get_keys(i)

        if isinstance(i, list) or len(keys) > 1:
            if not isinstance(vals, list):
                # Internally, all keys expand into a list but if it is just
                # 1 element we allow it to count as a singleton
                raise ValueError("If key is a list, or expands to a list," +
                                 "vals must also be a list")
            if not len(keys) == len(vals):
                raise ValueError("Found ", len(keys), " keys and ", len(vals), 
                                 "vals")

        # if the keys all exist, set their corresponding values
        if all(k in self for k in keys):
            if not isinstance(vals, list):
                vals = [vals]
            for k,v in zip(keys, vals):
                OrderedDict.__setitem__(self, k, v)

        # if none of the keys exist, insert them
        elif not any(k in self for k in keys):
            self.insert(i, vals)

        else:
            raise ValueError("All keys must exist or none can exist")

        self._list = None
        return vals


    def insert(self, keys, values, before=None):
        """ 
        Insert a key/value pair before 'before' in the dict.
        See __setitem__
        """

        if isinstance(keys, list):
            if not isinstance(values, list):
                raise ValueError("If keys is a list, vals must also be a list")
            if not len(keys) == len(values):
                raise ValueError("Found ", len(keys), " keys and ", len(values), 
                                 "vals")
            assert(all(isinstance(k, str) for k in keys))
            assert(not any(k in self for k in keys))
        elif isinstance(keys, str):
            assert(not keys in self)
            # so that zip works correctly below
            keys = [keys]
            values = [values]

        if before is None:
            link_prev = self._OrderedDict__root[0]
            link_next = link_prev[1]
        else:
            link_prev = self._OrderedDict__map[before][0]
            link_next = link_prev[1]

        for k,v in zip(keys, values):
            l = [link_prev, link_next, k]
            self._OrderedDict__map[k] = l
            link_prev[1] = l
            link_next[0] = l
            dict.__setitem__(self, k, v)
            link_prev = l
            






    
