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
        self._last_index = None
        self._last_dir = None

        # a list of the keys in this Index (including the dirs)
        self._list = None

        # a list of all files in the tree (no dirs)
        # we lazy update this for performance reasons
        # We should think closely about this. If the keyspace is huge
        # pulling this into an array is more expensive than walking
        # the tree
        self._full_key_list = None

        # this should always be accurate
        self._full_key_list_count = 0

        self._path = "/"
        self._parent = None
        OrderedDict.__init__(self, *args)
        if not all(isinstance(k,str) for k in self):
            raise ValueError("All keys must be strings")

        self.refresh_index_list()
        self.refresh_full_key_list(Check=False)
        self._full_key_list_count = len(self._full_key_list)

    def iter_slice(self, s, prefix, offset):
        """ 
        Iterate over a slice, stopping if we would loop around 
        The slice stop value that's passed in is included in the
        result (as opposed to a normal slice where it is excluded)
        We return to normal semantics by setting the stop to the next
        """
        root = self._OrderedDict__root
        # if empty, do nothing
        if root[0][2] == root[2]:
            return list(), 0

        start_part = s.start
        start_part2 = None
        if s.start:
            start_part1, start_part2 = self.__get_dir_component(s.start)
            start_part = start_part1 if start_part1 is not None else s.start

        stop_part = s.stop
        stop_part2 = None
        if s.stop:
            stop_part1, stop_part2 = self.__get_dir_component(s.stop)
            stop_part = stop_part1 if stop_part1 is not None else s.stop

        if s.step is None or s.step >= 0:

            curr = self._OrderedDict__map[start_part] if start_part else root[1]
            stop = self._OrderedDict__map[stop_part] if stop_part else None
            step = s.step if s.step else 1
            nx = 1
        else:
            curr = self._OrderedDict__map[start_part] \
                if start_part else root[0]
            stop = self._OrderedDict__map[stop_part] if stop_part else None

            step = -s.step
            nx = 0

        if stop and curr[2] == stop[2] and stop_part2 is None:
            return list(), 0

        ret = list()
        while True:
            if curr[2][-1:] == "/":
                next_index = dict.__getitem__(self, curr[2])

                # if the current Index's name does not match
                # the start string, we want the whole thing
                new_start = None if start_part != curr[2] else start_part2
                # likewise if the Index's name does not match the stop string
                new_stop = None if stop_part != curr[2] else stop_part2
                i_new = slice(new_start, new_stop, s.step)
                child_ret, offset = next_index.iter_slice(i_new, prefix + curr[2], offset)
                ret.extend(child_ret)
            else:
                if offset == 0:
                    ret.append((prefix + curr[2], curr[2], self))
                    offset = step-1
                else:
                    offset -= 1


            curr = curr[nx]
            # if we reach a stop node we're done (not dirs though)
            if stop and curr[2] == stop[2] and curr[2][-1:] != "/":
                break
            # if we've reached the end, we're also done. however if
            # stop was set to something, it means it was prior to the start
            # and so we should return an empty result
            if curr[2] == root[2]:
                if stop and stop[2][-1:] != "/":
                    ret = list()
                break

        return ret, offset


    def refresh_index_list(self):
        """ This rebuilds the self._list if the list of items changes. """
        if self._list == None:
            self._list = list()
            for k in OrderedDict.__iter__(self):
                self._list.append(k)

    def refresh_full_key_list(self, Check=True):
        self._refresh_full_key_list_main("", Check)

    def _refresh_full_key_list_main(self, prefix, Check):
        self.refresh_index_list()

        # ret is returned to the parent and includes the prefix
        ret = list()
        # self_ret is stored here and does not have the prefix
        self_ret = list()

        for k in self._list:
            if k[-1:] == "/":
                next_index = dict.__getitem__(self, k)
                ret.extend(next_index._refresh_full_key_list_main(prefix + k, False))
                self_ret.extend(next_index._refresh_full_key_list_main(k, False))
            else:
                ret.append(prefix + k)
                self_ret.append(k)
        self._full_key_list = self_ret

        if Check is True:
            assert len(ret) == self._full_key_list_count
        return ret

    def _get_keys(self, i, expand=True):
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
            iterable (of different possible types) tuple of
            key, file name from key, and parent index
            the last two may be None and is a cache to save a lookup
        """
        if isinstance(i, str):
            # if ends in / list the contents of the whole dir
            if expand is True and i[-1:] == "/":
                e_index, unused = self.__find_enclosing_index(i)
                if e_index is None:
                    return list()
                ret = list()
                node = e_index._OrderedDict__root
                node = node[1]
                while node[2] != e_index._OrderedDict__root[2]:
                    ret.append((i + node[2], node[2], e_index))
                    node = node[1]
                return ret
            else:
                e_index, file_end = self.__find_enclosing_index(i)
                return [(i, file_end, e_index)]
        elif isinstance(i, int):
            key = self.__get_key_from_offset(i)
            e_index, file_end = self.__find_enclosing_index(key)
            return [(key, file_end, e_index)]
        elif isinstance(i, list) or isinstance(i, np.ndarray):
            if isinstance(i[0], str):
                ret = list()
                for one_key in i:
                    # if ends in / list the contents of the whole dir
                    if expand == True and one_key[-1:] == "/":
                        e_index, unused = self.__find_enclosing_index(one_key)
                        node = e_index._OrderedDict__root
                        node = node[1]
                        while node[2] != e_index._OrderedDict__root[2]:
                            ret.append((one_key + node[2], node[2], e_index))
                            node = node[1]
                    else:
                        #traceback.print_stack()
                        e_index, file_end = self.__find_enclosing_index(one_key)
                        ret.append((one_key, file_end, e_index))
                return ret
                    
            elif isinstance(i[0], bool):
                # bools are also ints, but ints are not bools
                # so check for bool first
                if self.__len__() != len(i):
                    raise KeyError("Bool list length must match index key list")
                return self.__bool_key_list(i)

            elif isinstance(i[0], int):
                ret = list()
                for j in i:
                    key = self.__get_key_from_offset(j)
                    e_index, file_end = self.__find_enclosing_index(key)
                    ret.append((key, file_end, e_index))
                return ret

        elif isinstance(i,slice):
            # a slice's members are read-only so we construct a new slice
            slice_start = i.start
            slice_stop = i.stop
            slice_step = i.step if i.step else 1

            count_keys = -1
            if isinstance(i.start, int):
                if count_keys == -1:
                    count_keys = self.__len__()
                """
                going out of bounds is valid with slices
                since we will do lookups in the self._list we need to select
                valid entries
                """
                if slice_start <= ((-1 * count_keys) - 1):
                    # start before beginning and step backward, do nothing
                    if slice_step < 0:
                        return list()
                    slice_start = None
                elif slice_start >= count_keys:
                    # start off the end of the list and step forward, do nothing
                    if slice_step > 0:
                        return list()
                    slice_start = None
                else:
                    slice_start = self.__get_key_from_offset(slice_start)
            if isinstance(i.stop, int):
                if count_keys == -1:
                    count_keys = self.__len__() 
                """
                going out of bounds is valid with slices
                since we will do lookups in the self._list we need to select
                valid entries
                """
                if slice_stop <= ((-1 * count_keys) - 1):
                    # stop is before beginning and step forward, do nothing
                    if slice_step > 0:
                        return list()
                    slice_stop = None
                elif slice_stop >= (count_keys):
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
                    slice_stop = self.__get_key_from_offset(slice_stop)

            i_new = slice(slice_start, slice_stop, slice_step)
            return self.iter_slice(i_new, "", 0)[0]
        else:
            raise KeyError(i)



    def subset(self, i):
        """ For an indexing i, return the subset of the dataframe"""
        keys = self._get_keys(i)
        data = list()
        if isinstance(i,str):
            for (key, key_end, new_index) in keys:
                data.append((key_end, dict.__getitem__(new_index, key_end)))
        else:
            for (key, key_end, new_index) in keys:
                data.append((key, dict.__getitem__(new_index, key_end)))
        return Index(data)

    def __getitem__(self, i):
        """ Get items (as array of values) for indexing i """
        if isinstance(i, list) or isinstance(i, slice):
            l = list()
            for k, key_end, e_index in self._get_keys(i):
                if e_index is None:
                    raise KeyError(k)
                else:
                    l.append(dict.__getitem__(e_index, key_end))
            return l

        else:
            keys = self._get_keys(i)
            if keys == None:
                return []
            if len(keys) == 1:
                k, key_end, e_index = keys[0]
                if e_index is None or not dict.__contains__(e_index, key_end):
                    raise KeyError(keys[0])
                if e_index is not None and keys[0][-1] != "/":
                    return dict.__getitem__(e_index, key_end)
                elif not key_end == None:
                    return key_end

            elif len(keys) > 1:
                l = list()
                for k, key_end, e_index in keys:
                    if e_index is None:
                        raise KeyError(k, " enclosing index was not found")
                    if not dict.__contains__(e_index, key_end):
                        raise KeyError(key_end, "was not found")
                    if e_index is not None and k[-1] != "/":
                        l.append(dict.__getitem__(e_index, key_end))
                    elif key_end:
                        l.append(key_end)
                return l
            else:
                return []


    # returns the number of files in the hierarchy, not the number of elements 
    # this index
    def __len__(self):
        return self._full_key_list_count

    def __iter__(self):
        class iterator(object):
            def __init__(self, obj):
                self.cur_dir = obj
                self.cur_index = 0
                self.cur_dir.refresh_full_key_list()
            def __iter__(self):
                return self
            def next(self):
                try:
                    ret = self.cur_dir._full_key_list[self.cur_index]
                except IndexError:
                    raise StopIteration
                self.cur_index += 1
                return ret

        return iterator(self)

    def __delitem__(self, i):
        """ 
        Delete items for indexing. 
        """
        keys = self._get_keys(i)
        if any(not self.key_exists(k, key_end, e_index) for k, key_end, e_index in keys):
            raise KeyError(i)

        for k, key_end, e_index in keys:
            files_count = 1
            if k[-1] is '/':
                files_count -= 1

            if  e_index is None:
                raise KeyError(k)

            OrderedDict.__delitem__(e_index, key_end)
            e_index._list = None
            e_index._full_key_list = None

            while e_index is not None:
                e_index._full_key_list_count -= files_count
                assert e_index._full_key_list_count >= 0
                e_index = e_index._parent

        self._list = None


    def relabel(self, keys):
        """ Relabel a collection of keys (a dictionary of old -> new. 
        This does not allow moving things between different directories """

        keys_repacked = list()
        for k,v in keys.iteritems():
            old_parent, old_file = self.__get_parent_dir_and_file(k)
            new_parent, new_file = self.__get_parent_dir_and_file(v)
            if old_parent != new_parent:
                raise KeyError("key parent directory mismatch:", \
                                   old_key," and ", new_key)

            old_index, unused =  self.__find_enclosing_index(k)
            assert self.key_exists(k, old_file, old_index)
            assert not self.key_exists(v, new_file, old_index)
            keys_repacked.append((old_file, new_file, old_index))

        for old_file, new_file, old_index in keys_repacked:
            l = old_index._OrderedDict__map[old_file]
            del old_index._OrderedDict__map[old_file]
            old_index._OrderedDict__map[new_file] = l
            l[2] = new_file
            dict.__setitem__(old_index, new_file, dict.pop(old_index, old_file))
            self._enclosing_index_cache_dir = None
            self._enclosing_index_cache_value = None

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
        keys = self._get_keys(i, False)

        if isinstance(i, list) or len(keys) > 1:
            if not isinstance(vals, list):
                # Internally, all keys expand into a list but if it is just
                # 1 element we allow it to count as a singleton
                raise ValueError("If key is a list, or expands to a list," +
                                 "vals must also be a list")
            if not len(keys) == len(vals):
                raise ValueError("Found ", len(keys), " keys and ", len(vals), 
                                 "vals")

        exist_count = 0
        for k, key_end, e_index in keys:
            """
            if key is a dir and the dir is there, True
            if key is a dir + file and that is there, True
            This code is duplicated from key_exists_ It was
            better for performance to remove the function call
            """
            if e_index is None:
                continue

            if key_end is None:
                exist_count += 1
                continue

            try:
                dict.__getitem__(e_index, key_end)
                exist_count += 1
                continue
            except KeyError:
                continue


        # if the keys all exist, set their corresponding values
        if exist_count == len(keys):
            if not isinstance(vals, list):
                vals = [vals]
            for k,v in zip(keys, vals):
                key_name, key_end, e_index = k
                OrderedDict.__setitem__(e_index, key_end, v)
                # since the keys are unchanged we don't need to update the
                # key caches

        # if none of the keys exist, insert them
        elif exist_count == 0:
            self._full_key_list = None
            if not isinstance(i, list) and len(keys) == 1:
                keys = keys[0]
            self.__insert_main(keys, vals)
        else:
            raise ValueError("All keys must exist or none can exist")

        return vals

    def insert(self, keys, values, before=None):
        self._full_key_list = None

        before_file = None
        ## check that  if key is dir, val is Index and vice versa
        if before is not None:
            if not isinstance(before, str):
                raise KeyError("before must be a single key")
            before_index, before_file = self.__find_enclosing_index(before)

        if type(keys) is list:
            packed_keys = list()
            if not isinstance(values, list):
                raise ValueError("If keys is a list, vals must also be a list")
            if not len(keys) == len(values):
                raise ValueError("Found ", len(keys), " keys and ", len(values), 
                                 "vals")
            for k in keys:
                assert isinstance(k,str)
                e_index, file_end = self.__find_enclosing_index(k)
                assert not self.key_exists(k, file_end, e_index)
                
                if before is not None:
                    if id(e_index) != id(before_index):
                        raise KeyError("key parent dir and before parent dir mismatch")
                packed_keys.append((k, file_end, e_index))
        else:
            e_index, file_end = self.__find_enclosing_index(keys)
            assert(not self.key_exists(keys[0], file_end, e_index))
            if before is not None:
                if id(e_index) != id(before_index):
                    raise KeyError("key parent dir and before parent dir mismatch")
            packed_keys = (keys, file_end, e_index)

        self.__insert_main(packed_keys, values, before_file)

    def __insert_main(self, keys, values, before_file=None):
        """ 
        Insert a key/value pair before 'before_file' in the dict.
        See __setitem__
        """
        if isinstance(keys[0], str):
            # so that zip works correctly below
            keys = [keys]
            values = [values]

            link_prev = None
            link_next = None
        if before_file is None:
            link_prev = self._OrderedDict__root[0]
            link_next = link_prev[1]

        for key,v in zip(keys, values):
            files_count = 1
            if type(v) is Index:
                files_count += len(v)

            k, file_end, new_index = key
            if new_index is None:
                new_index, file_end = self.__find_enclosing_index_make_missing(k)

            if file_end[-1] == '/':
                files_count -= 1

            if before_file is None:
                link_prev = new_index._OrderedDict__root[0]
                link_next = link_prev[1]
            else:
                link_prev = new_index._OrderedDict__map[before_file][0]
                link_next = link_prev[1]
            l = [link_prev, link_next, file_end]
            new_index._OrderedDict__map[file_end] = l
            link_prev[1] = l
            link_next[0] = l
            dict.__setitem__(new_index, file_end, v)
 
            new_index._list = None
            new_index._full_key_list = None

            while new_index is not None:
                new_index._full_key_list_count += files_count
                new_index = new_index._parent

    def __get_parent_dir_and_file(self, key):
        """
        Return the full parent dir path for the key, if any
        Return None, key if there is no slash in the key
        """
        if key[-1] == '/':
            key = key[:-1]

        loc = key.rfind('/')
        if loc >= 0:
            return key[:loc+1], key[loc+1:]
        else:
            return None, key

    def __get_parent_dir_and_file_keep_dir(self, key):
        """
        Same as __get_parent_dir_and_file but don't delete
        any trailing slash
        """

        loc = key.rfind('/')
        if loc >= 0:
            return key[:loc+1], key[loc+1:]
        else:
            return None, key


    def __get_dir_component(self, key):
        """
        If the key has a '/' in it, return a tuple of the topmost directory and
        the remaining part. The topmost directory includes a trailing slash

        Returns None, None if the key does not have a /

        If the key is just one dir part, return as the first part of the tuple
        """
        loc = key.find('/')
        if loc >= 0:
            return key[:loc+1], key[loc+1:]
        else:
            return None, None

    def __contains__(self, item):
        e_index, file_end = self.__find_enclosing_index(item)
        return self.key_exists(item, file_end, e_index)

    def key_exists(self, key, new_file, new_index):
        """
         if key is a dir and the dir is there, True
         if key is a dir + file and that is there, True
        """
        if new_index is None:
            return False
            
        if new_file is None:
            return True

        try:
            dict.__getitem__(new_index, new_file)
            return True
        except KeyError:
            return False

    def __find_enclosing_index(self, key):
        """
        Return an Index and a file from the key
        We verify that dirs exist but not files
        There are different flavors of this function. It is called 
        so often that we tune each one for performance
        """

        parts = key.split('/')
        file = parts[-1]
        dirs = parts[0:-1]

        if self._last_dir == dirs:
            return self._last_index, file

        cur_index = self
        for p in dirs:
            try:
                cur_index = dict.__getitem__(cur_index, p+"/")
            except KeyError:
                return None, None

        self._last_dir = dirs
        self._last_index = cur_index

        if file == "":
            return cur_index, dirs[-1]+"/"

        return cur_index, file

    def __find_enclosing_index_make_missing(self, key):
        """
        Return an Index and a file from the key
        We verify that dirs exist but not files. We verify
        the file if it is a dir though
        """

        parts = key.split('/')
        file = parts[-1]
        dirs = parts[0:-1]

        if file == "":
            file = dirs[-1]+'/'
            dirs = dirs[0:-1]

        cur_index = self
        for p in dirs:
            try:
                cur_index = dict.__getitem__(cur_index, p+"/")
            except KeyError:
                p = p+"/"
                link_prev = cur_index._OrderedDict__root[0]
                link_next = link_prev[1]
                l = [link_prev, link_next, p]
                cur_index._OrderedDict__map[p] = l
                link_prev[1] = l
                link_next[0] = l

                next_index = Index()
                next_index._path = p
                dict.__setitem__(cur_index, p, next_index)
                next_index._parent = cur_index
                cur_index._list = None
                cur_index = next_index

        return cur_index, file

    def __bool_key_list(self, i):
        """ Given a bool list i, return a list of keys for each True index

        Args:
            A list i which has the same number of elements as the number
            of files in the Index. This includes files in subdirectories
            but does not include the directories themselves

        Returns:
            a list of keys matching the True indices in the list i
        """

        ret = list()
        for i_key, i_bool in zip(self.__iter__(), i):
            if i_bool is True:
                e_index, file_end = self.__find_enclosing_index(i_key)
                ret.append((i_key, file_end, e_index))
        return ret

    def __get_key_from_offset(self, count):
        if self._full_key_list is None:
            self.refresh_full_key_list()
        return self._full_key_list[count]

    def __str__(self):
        print "calling str"
        self.refresh_index_list()
        l = [(k,dict.__getitem__(self,k)) for k in self._list]
        l_str = ["('"+str(k)+"', '"+str(v)+"')" if isinstance(v,str)
                 else "('"+str(k)+"', "+str(v)+")" for k,v in l]
        return "Index(["+', '.join(l_str)+"])"

