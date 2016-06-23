from collections import OrderedDict
import fnmatch
import numpy as np
import itertools


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
        # a list of the keys in this Index (including the dirs)
        self._list = []

        # a list of all files in the tree (no dirs)
        # we lazy update this for performance reasons
        # We should think closely about this. If the keyspace is huge
        # pulling this into an array is more expensive than walking
        # the tree
        self._full_key_list = None

        # this should always be accurate
        self._nfiles = 0

        self._path = "/"

        OrderedDict.__init__(self, *args)
        if not all(isinstance(k,str) for k in self):
            raise ValueError("All keys must be strings")

        self.refresh_full_key_list()
        self._nfiles = len(self._full_key_list)

        self._subset_cache = {}

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
                    ret.append(prefix + curr[2])
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

    def refresh_full_key_list(self):
        self._refresh_full_key_list_main("")

    def _refresh_full_key_list_main(self, prefix):
        # ret is returned to the parent and includes the prefix
        ret = list()
        # self_ret is stored here and does not have the prefix
        self_ret = list()

        if self._list is None:
            self._full_key_list = self_ret
            return ret

        for k in self._list:
            if k[-1:] == "/":
                next_index = dict.__getitem__(self, k)
                ret.extend(next_index._refresh_full_key_list_main(prefix + k))
                self_ret.extend(next_index._refresh_full_key_list_main(k))
            else:
                ret.append(prefix + k)
                self_ret.append(k)
        self._full_key_list = self_ret
        assert len(ret) == self._nfiles
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
            iterable (of different possible types) with keys from match
        """
        if isinstance(i, str):
            # if ends in / list the contents of the whole dir
            if expand is True and i[-1:] == "/":
                e_index, key_end = self.__find_enclosing_index(i, True)
                if e_index is None:
                    # return list()
                    return
                ret = list()
                node = e_index._OrderedDict__root
                node = node[1]
                while node[2] != e_index._OrderedDict__root[2]:
                    # ret.append(i + node[2])
                    yield i + node[2]
                    node = node[1]
            else:
                # return [i]
                yield i
        elif isinstance(i, int):
            key = self.__get_key_from_offset(i)
            yield key
            # return [key]
        elif isinstance(i, list) or isinstance(i, np.ndarray):
            if isinstance(i[0], str):
                ret = list()
                for one_key in i:
                    # if ends in / list the contents of the whole dir
                    if expand is True and one_key[-1:] == "/":
                        e_index, key_end = self.__find_enclosing_index(one_key, True)
                        node = e_index._OrderedDict__root
                        node = node[1]
                        while node[2] != e_index._OrderedDict__root[2]:
                            yield one_key + node[2]
                            # ret.append(one_key + node[2])
                            node = node[1]
                    else:
                        # ret.append(one_key)
                        yield one_key
                # return ret
                    
            elif isinstance(i[0], bool):
                # bools are also ints, but ints are not bools
                # so check for bool first
                if self.__len__() != len(i):
                    raise KeyError("Bool list length must match index key list")
                #return [self._list[j] for j in xrange(len(self)) if i[j]]
                for k in self.__bool_key_list(i):
                    yield k

            elif isinstance(i[0], int):
                # ret = list()
                for j in i:
                    key = self.__get_key_from_offset(j)
                    # ret.append(key)
                    yield key
                # return ret

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
                        # return list()
                        return
                    slice_start = None
                elif slice_start >= count_keys:
                    # start off the end of the list and step forward, do nothing
                    if slice_step > 0:
                        # return list()
                        return
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
                        # return list()
                        return
                    slice_stop = None
                elif slice_stop >= (count_keys):
                    # stop is after end but steping backward, do nothing
                    if slice_step < 0:
                        # return list()
                        return
                    slice_stop = None
                else:
                    """
                    A slice includes the start but excludes the stop
                    We tweak the stop value so that it is included when we
                    do the list lookup
                    """
                    slice_stop = self.__get_key_from_offset(slice_stop)

            i_new = slice(slice_start, slice_stop, slice_step)
            for k in self.iter_slice(i_new, "", 0)[0]:
                yield k
        else:
            raise KeyError(i)



    def subset(self, i):
        """ For an indexing i, return the subset of the dataframe. Exact key
        subset queries do not work. """
        if isinstance(i,str):
            if i in self._subset_cache:
                return self._subset_cache[i]
            else:
                loc = i.find('/')
                if loc >= 0 and i.endswith('/'):
                    if dict.__contains__(self,i[:loc+1]):
                        if loc+1 >= len(i):
                            return dict.__getitem__(self,i)

                        else:
                            return dict.__getitem__(self,i[:loc+1]).subset(i[loc+1:])
                    else:
                        return Index()
                else:
                    return Index([(i,self[i])])
        elif isinstance(i,slice):
            if (i.start,i.stop,i.step) == (None,None,None):
                return self
            else:
                return Index(itertools.islice(self.iteritems(),i.start,i.stop,i.step))
        else:
            keys = self._get_keys(i)
            data = list()
            for k in keys:
                new_index, key_end = self.__find_enclosing_index(k, False)
                if new_index is None:
                    raise KeyError(k)
                data.append((k, dict.__getitem__(new_index, key_end)))
            return Index(data)

    def __getitem__(self, i):
        """ Get items (as array of values) for indexing i """
        if isinstance(i, list) or isinstance(i, slice):
            l = list()
            for k in self._get_keys(i):
                e_index, key_end = self.__find_enclosing_index(k, False)
                if e_index is None:
                    raise KeyError(k)
                else:
                    l.append(dict.__getitem__(e_index, key_end))
            return l

        else:
            keys = list(self._get_keys(i))
            if keys == None:
                return []
            if len(keys) == 1:
                e_index, key_end = self.__find_enclosing_index(keys[0], False)
                if e_index is None or not dict.__contains__(e_index, key_end):
                    raise KeyError(keys[0])
                if e_index is not None and keys[0][-1] != "/":
                    return dict.__getitem__(e_index, key_end)
                elif not key_end == None:
                    return key_end

            elif len(keys) > 1:
                l = list()
                for k in keys:
                    e_index, key_end = self.__find_enclosing_index(k, False)
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
        return self._nfiles

    def __iter__(self):
        # Start at current index at the first index, which is the top level
        # directory
        dir_stack = list()
        cur_dir = self
        cur_index = 0
        cur_dir_name = ""
        while 1:
            # if there are keys left in this dir operate on the next one
            if cur_dir._list is not None and cur_index < len(cur_dir._list):
                key = cur_dir._list[cur_index]
                cur_index += 1
                if key[-1:] == "/":
                    dir_stack.append((cur_dir, cur_index))
                    cur_index = 0
                    cur_dir = dict.__getitem__(cur_dir, key)
                    cur_dir_name += key
                else:
                    yield cur_dir_name + key
            # if there are no keys left go back up one level or end
            else:
                if len(dir_stack) > 0:
                    (cur_dir, cur_index) = dir_stack.pop()
                    last_part = cur_dir_name.rfind('/', 0, -2)
                    if last_part >= 0:
                        cur_dir_name = cur_dir_name[:last_part+1]
                    else:
                        cur_dir_name = ""
                else:
                    break

    def __reversed__(self):
        # Start at current index at the first index, which is the top level
        # directory
        dir_stack = list()
        cur_dir = self
        cur_index = len(cur_dir._list)-1
        cur_dir_name = ""
        while 1:
            # if there are keys left in this dir operate on the previous one
            if cur_dir._list is not None and cur_index >= 0:
                key = cur_dir._list[cur_index]
                cur_index -= 1
                if key[-1:] == "/":
                    dir_stack.append((cur_dir, cur_index))
                    cur_dir = dict.__getitem__(cur_dir, key)
                    cur_index = len(cur_dir._list)-1
                    cur_dir_name += key
                else:
                    yield cur_dir_name + key
            # if there are no keys left go back up one level or end
            else:
                if len(dir_stack) > 0:
                    (cur_dir, cur_index) = dir_stack.pop()
                    last_part = cur_dir_name.rfind('/', 0, -2)
                    if last_part >= 0:
                        cur_dir_name = cur_dir_name[:last_part+1]
                    else:
                        cur_dir_name = ""
                else:
                    break

    def __delitem__(self, i):
        if any(not self.key_exists(k) for k in self._get_keys(i)):
            raise KeyError(i)
        self.__delete_main(self._get_keys(i))
        self._subset_cache = {}

    def __delete_main(self, keys):
        """ Delete items for indexing. """
        for k in keys:

            if k[-1] is '/':
                files_count -= 1
            kpart1, kpart2 = self.__get_dir_component(k)
            if kpart1 is not None:
                next_index = dict.__getitem__(self, kpart1)
                if next_index is None:
                    raise KeyError(k)
                next_index.__delitem__(kpart2)
            else:
                if k[-1] is '/':
                    temp_index = dict.__getitem__(new_index, file_key)
                    if type(temp_index) is Index:
                        files_count -= len(temp_index)
                OrderedDict.__delitem__(self, k)
                #self._list = None
                if self._list != None:
                    self._list.remove(k)
                self._full_key_list = None
            
            self._nfiles -= 1
        assert self._nfiles >= 0


    def relabel(self, keys):
        """ Relabel a collection of keys (a dictionary of old -> new. 
        This does not allow moving things between different directories.
        This code seems useless. The purpose of relabeling is to move around
        files across directories. """

        assert(all(self.key_exists(k) and not self.key_exists(v) \
                       for k,v in keys.iteritems()))
        for old_key in keys.keys():
            new_key = keys[old_key]

            old_parent, old_file = self.__get_parent_dir_and_file(old_key)
            new_parent, new_file = self.__get_parent_dir_and_file(new_key)
            if old_parent != new_parent:
                raise KeyError("key parent directory mismatch:", \
                                   old_key," and ", new_key)
            
            old_index, old_key_file = self.__find_enclosing_index(old_key, False)
            l = old_index._OrderedDict__map[old_file]
            del old_index._OrderedDict__map[old_file]
            old_index._OrderedDict__map[new_file] = l
            l[2] = new_file 
            dict.__setitem__(old_index, new_file, dict.pop(old_index, old_file))
        self._list = [k for k in OrderedDict.__iter__(self)]
        self._full_key_list = None

    def __setitem__(self, i, vals):
        """ Set items for indexing i. 

        If the indexing i exists in the Index __setitem__ will set the 
        corresponding values for each key. The number of keys must therefore
        match the number of vals unless i is a singleton; in that case
        the key will be set to the entirety of vals. Note that if i is a list
        then vals must also be a list. If i is a singleton, vals can be anything
        If i is a singleton or a list, and if none of the keys exists in the 
        Index, then we add (append) them to the Index.

        The key_exists check is inefficient because we re-do it at each 
        subsequent level
        """
        self._subset_cache = {}
        len_keys = sum(1 for _ in self._get_keys(i,False))

        if isinstance(i, list) or len_keys > 1:
            if not isinstance(vals, list):
                # Internally, all keys expand into a list but if it is just
                # 1 element we allow it to count as a singleton
                raise ValueError("If key is a list, or expands to a list," +
                                 "vals must also be a list")
            if not len_keys == len(vals):
                raise ValueError("Found ", len_keys, " keys and ", len(vals), 
                                 "vals")

        # if none of the keys exist, insert them
        if not any(self.key_exists(k) for k in self._get_keys(i, False)):
            self._full_key_list = None
            self.__insert_main(i, vals, True)
        # if the keys all exist, set their corresponding values
        elif all(self.key_exists(k) for k in self._get_keys(i, False)):
            if not isinstance(vals, list):
                vals = [vals]
            for k,v in zip(self._get_keys(i, False), vals):
                e_index, key_end = self.__find_enclosing_index(k, False)
                OrderedDict.__setitem__(e_index, key_end, v)
                # since the keys are unchanged we don't need to update the
                #e_index._list = None
                # key caches
                #e_index._full_key_list = None
                #self._full_key_list = None

        else:
            raise ValueError("All keys must exist or none can exist")

        return vals


    def insert(self, keys, values, before=None):
        self._full_key_list = None
        self.__insert_main(keys, values, False, before)

    def __insert_main(self, keys, values, keys_checked, before=None):
        """ 
        Insert a key/value pair before 'before' in the dict.
        See __setitem__
        """
        if before is not None:
            i = self._list.index(before)
            link_prev, link_next = None, None
            if not isinstance(before, str):
                raise KeyError("before must be a single key")
            before_parent, before_file = self.__get_parent_dir_and_file(before)
        else:
            i = len(self._list)
            link_prev = self._OrderedDict__root[0]
            link_next = link_prev[1]

        new_list = self._list[:i]

        if isinstance(keys, list):
            if not keys_checked:
                if not isinstance(values, list):
                    raise ValueError("If keys is a list, vals must also be a list")
                if not len(keys) == len(values):
                    raise ValueError("Found ", len(keys), " keys and ", len(values), 
                                     "vals")
                assert(all(isinstance(k, str) for k in keys))
                assert(not any(self.key_exists(k) for k in keys))

                if before is not None:
                    for k in keys:
                        key_parent, key_file = self.__get_parent_dir_and_file(k)
                        if key_parent != before_parent:
                            raise KeyError("key parent dir and before parent dir mismatch")
            kvs = zip(keys,values)
            files_count = len(keys)
        elif isinstance(keys, str):
            if not keys_checked:
                assert(not self.key_exists(keys))
                if before is not None:
                    key_parent, key_file = self.__get_parent_dir_and_file(keys)
                    if key_parent != before_parent:
                        raise KeyError("key parent dir and before parent dir mismatch")

            # so that zip works correctly below
            kvs = [(keys,values)]
            keys = [keys]
            files_count = 1
        else: 
            kvs = zip(keys,values)
            files_count = len(keys)

        for k,v in kvs:
            kpart1, kpart2 = self.__get_dir_component(k)
            if k[-1] is '/':
                files_count -= 1
            # a dir key and an Index value is allowed
            if kpart2 is not None and len(kpart2) == 0 and type(v) is not Index:
                raise KeyError("Insert has directory path but no file name")

            if type(v) is Index:
                files_count += len(v)
                if len(kpart1) < 1 or kpart1[-1] is not '/':
                    raise KeyError("Value is an Index but key is not a dir")

            if kpart1 is not None:
                if dict.__contains__(self, kpart1):
                    # if the dir exists, recurse
                    next_index = dict.__getitem__(self, kpart1)
                    next_index.__insert_main(kpart2, v, True, before)
                else:
                    # if the dir doesn't exist, insert it, then recurse
                    if link_prev is None:
                        link_prev = self._OrderedDict__map[before_file][0]
                        link_next = link_prev[1]
                    l = [link_prev, link_next, kpart1]
                    self._OrderedDict__map[kpart1]=link_prev[1]=link_next[0]=l

                    if len(kpart2) > 0:
                        next_index = Index()
                        next_index._path = kpart1
                        dict.__setitem__(self, kpart1, next_index)
                        next_index.__insert_main(kpart2, v, True, before)
                        # if we create a new directory, add it to the list
                    else:
                        # this case only happens if key is a dir and value is an Index
                        dict.__setitem__(self, kpart1, v)
                    new_list.append(kpart1)
                    link_prev = l

            else:
                # insert the file into the current dir
                if link_prev is None:
                    link_prev = self._OrderedDict__map[before_file][0]
                    link_next = link_prev[1]

                self._OrderedDict__map[k] = link_prev[1] = link_next[0] = [link_prev, link_next, k]
                dict.__setitem__(self, k, v)
                link_prev = link_prev[1]
                new_list.append(k)
 

        # if self._list is not None: 
        #     if before is None : 
        #         # self._list = self._list + keys
        #         print self._list, keys
        #         self._list = None
        #     else:
        #         i = self._list.index(before)
        #         self._list = self._list[:i] + keys + self._list[i:]
        # else:
        new_list += self._list[i:]
        self._list = new_list

        self._full_key_list = None
        self._nfiles += files_count

    def __get_parent_dir_and_file(self, key):
        """
        Return the full parent dir path for the key, if any
        """
        if key[-1] == '/':
            key = key[:-1]

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
        """
        # if len(key) == 0:
        #     return None, key
        # if len(key) == 0:
        #     return None, None
        loc = key.find('/')
        if loc >= 0:
            return key[:loc+1], key[loc+1:]
        else:
            return None, None

    def __contains__(self, item):
        return self.key_exists(item)

    # if key is a dir and the dir is there, True
    # if key is a dir + file and that is there, True
    def key_exists(self, key):
        """
        if self._full_key_list is None:
            self.refresh_full_key_list()

        for k in self._full_key_list:
            if k == key:
                return True
            if key[-1] is '/' and k.startswith(key):
                return True
        return False
        """
        kpart1, kpart2 = self.__get_dir_component(key)
        if kpart1:
            if self._list is not None and kpart1 in self._list:
                next_index = dict.__getitem__(self, kpart1)
                if len(kpart2) == 0:
                    return True
                return next_index.key_exists(kpart2)
            else:
                return False
        else:
            if self._list is not None and key in self._list:
                return True
            else:
                return False


    def __find_enclosing_index(self, key, expand_dir):
        # split key into the first directory and the remainder of the path
        kpart1, kpart2 = self.__get_dir_component(key)
        if kpart1:
            # if kpart1 is a valid dir, recurse on the remainder
            if kpart1 in self:
                # if the last part of the key is a directory but we aren't to 
                # expand it, just return the key string. The caller will know
                # not to try to look up the value
                if not expand_dir and len(kpart2) == 0:
                    return self, kpart1
                next_index = dict.__getitem__(self, kpart1)  
                return next_index.__find_enclosing_index(kpart2, expand_dir)
            else:
                return None, None
        else:
            return self, key

    def __bool_key_list(self, i):
        """ Given a bool list i, return a generator of keys for each True index

        Args:
            An iterable i which has the same number of elements as the number
            of files in the Index. This includes files in subdirectories
            but does not include the directories themselves

        Returns:
            a generator of keys matching the True indices in the list i
        """
        if self._full_key_list is None:
            self.refresh_full_key_list()

        for i_key, i_bool in zip(self._full_key_list, i):
            if i_bool is True:
                yield i_key

    def __get_key_from_offset(self, offset):
        if self._full_key_list is None:
            self.refresh_full_key_list()
        return self._full_key_list[offset]

    def __str__(self):
        l = [(k,dict.__getitem__(self,k)) for k in self._list]
        l_str = ["('"+str(k)+"', '"+str(v)+"')" if isinstance(v,str)
                 else "('"+str(k)+"', "+str(v)+")" for k,v in l]
        return "Index(["+', '.join(l_str)+"])"

