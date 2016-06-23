from dreaml.dataframe.index import Index

class TestIndex:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def test_index_individual(self):
        new_index = Index()
        new_index["key0"]  = ["value0"]
        new_index["key1"]  = "value1"
        new_index.insert("key2", ["value2"])
        new_index.insert("key3", "value3")

        try:
            new_index[dict()]
            raise
        except KeyError:
            pass
        assert len(new_index.keys()) == 4
        assert new_index["key0"] == ["value0"]
        assert new_index["key1"] == "value1"
        assert new_index["key2"] == ["value2"]
        assert new_index["key3"] == "value3"
        assert new_index[0] == ["value0"]
        assert new_index[1] == "value1"
        assert new_index[2] == ["value2"]
        assert new_index[3] == "value3"

        try: 
            new_index[None]
        except KeyError:
            pass
        del new_index

    def test_index_bulk(self):
        new_index = Index()
        new_index[["key0", "key1"]] = ["value0", "value1"]
        new_index.insert(["key2", "key3"], ["value2", "value3"])
        assert len(new_index.keys()) == 4
        assert new_index["key0"] == "value0"
        assert new_index["key1"] == "value1"
        assert new_index["key2"] == "value2"
        assert new_index["key3"] == "value3"
        assert new_index[0] == "value0"
        assert new_index[1] == "value1"
        assert new_index[2] == "value2"
        assert new_index[3] == "value3"
        assert new_index[["key" + str(s) for s in range(4)]] \
            == ["value" + str(s) for s in range(4)]
        assert new_index[range(4)] == ["value" + str(s) for s in range(4)]
        assert new_index[0:4] == ["value" + str(s) for s in range(4)]

        try:
            new_index[["test_list"]] = "not_a_list"
            raise
        except ValueError:
            pass
        try:
            new_index.insert(["test_list"], "not_a_list")
            raise
        except ValueError:
            pass
        try:
            new_index[["test_list", "test_list1"]] = ["mismatch_list_size"]
            raise
        except ValueError:
            pass
        try:
            new_index.insert(["test_list", "test_list1"], ["mismatch_list_size"])
            raise
        except ValueError:
            pass
        try:
            new_index[["key0", "nonexistent_key"]] = ["a", "b"]
            raise
        except ValueError:
            pass
        try:
            new_index.insert(["key0", "nonexistent_key"], ["a", "b"])
            raise
        except AssertionError:
            pass

        del new_index
        new_index = Index()
        new_index["key2"]  = "value2"
        new_index.insert("key1", "value1", "key2")
        new_index.insert("key0", "value0", "key1")
    #    import IPython
    #    IPython.embed()
        assert new_index[0:3] == ["value" + str(s) for s in range(3)]
        assert new_index[[True, True, True]] == ["value" + str(s) for s in range(3)]
        assert new_index[[True, True, False]] == ["value" + str(s) for s in range(2)]
        try:
            new_index[[True, True, True, True]]
            raise
        except KeyError:
            pass

    def test_index_directories(self):
        new_index = Index()
        new_index[["path1/path2/key" + str(s) for s in range(self.item_count)]] \
            = ["value" + str(s) for s in range(self.item_count)] 
        #import IPython
        #IPython.embed()

        assert new_index["path1/path2/"] == \
            ["value" + str(s) for s in range(self.item_count)]

        assert new_index[["path1/path2/"]] == \
            ["value" + str(s) for s in range(self.item_count)]
        try:
            new_index["path1/path2"] 
            raise
        except KeyError:
            pass
        try:
            new_index[["path1/path2"]]
            raise
        except KeyError:
            pass

    def test_index_slice(self):
        new_index = Index()
        assert new_index[0:10] == []
        assert new_index[:] == []
        assert new_index[::2] == []

        new_index[["key" + str(s) for s in range(self.item_count)]] = \
            ["value" + str(s) for s in range(self.item_count)]
        assert new_index[0:0] == []
        assert new_index[0:0:1] == []
        assert new_index[0:0:2] == []
        assert new_index[0:0:-1] == []
        assert new_index[0:0:-1] == []
        assert new_index[0:0:-2] == []
        assert new_index[2:2] == []
        assert new_index[3:2] == []
        assert new_index[self.item_count:self.item_count] == []
        assert new_index[self.item_count:self.item_count:2] == []
        assert new_index[self.item_count:self.item_count:-2] == [] 
        assert new_index[self.item_count:0] == []
        assert new_index[self.item_count-1:0] == []
        assert new_index[self.item_count:] == []
        assert new_index[(-1*self.item_count)-1::-1] == []
        assert new_index[(-1*self.item_count)-1::] == \
            ["value" + str(s) for s in range(self.item_count)]
        assert new_index[-1:(-1*self.item_count)-1] == []

        temp_list = ["value" + str(s) for s in range(self.item_count)]
        list.reverse(temp_list)
        assert new_index[-1:(-1*self.item_count)-1:-1] == temp_list
        del temp_list

        assert new_index[::] == ["value" + str(s) for s in range(self.item_count)]
        assert new_index[::2] == ["value" + str(s) for s in range(0, self.item_count, 2)]
        assert new_index[::3] == ["value" + str(s) for s in range(0, self.item_count, 3)]
        assert new_index[:self.item_count:] == \
            ["value" + str(s) for s in range(self.item_count)]

        # there is no string key for past the end of the list, so we can only
        # access self.item_count-1 items in this manner
        assert new_index["key0":"key"+str(self.item_count-1)] == \
            ["value" + str(s) for s in range(self.item_count-1)]
        assert new_index["key2":"key"+str(self.item_count-1)] == \
            ["value" + str(s) for s in range(2, self.item_count-1)]
        assert new_index["key0":] == ["value" + str(s) for s in range(self.item_count)]
        assert new_index[:"key"+str(self.item_count-1)] == \
            ["value" + str(s) for s in range(self.item_count-1)]
        assert new_index["key0":"key"+str(self.item_count-1):2] == \
            ["value" + str(s) for s in range(0,self.item_count-1,2)]

        assert new_index["key0":4] == ["value" + str(s) for s in range(4)]
        assert new_index[0:"key4"] == ["value" + str(s) for s in range(4)]

        assert new_index["key4":"key2"] == []

        assert new_index["key2":-1] == \
            ["value" + str(s) for s in range(2, self.item_count-1)]
        del new_index

        """ modify data """
        new_index = Index()
        new_index[["key" + str(s) for s in range(self.item_count)]] = \
            ["value" + str(s) for s in range(self.item_count)]

        new_index["key0"] = "new_value0"
        new_index["key2"] = "new_value2"
        new_index[["key1", "key3"]] = ["new_value1", "new_value3"]
        assert len(new_index.keys()) == self.item_count
        assert new_index[["key" + str(s) for s in range(self.item_count)][0:4]] == \
            ["new_value" + str(s) for s in range(self.item_count)][0:4]
        if (self.item_count > 4):
            assert new_index[["key" + str(s) for s in range(self.item_count)][4:]] == \
                ["value" + str(s) for s in range(self.item_count)][4:]

        del new_index

    def test_index_delete(self):
        new_index = Index()
        new_index[["key" + str(s) for s in range(self.item_count)]] = \
            ["value" + str(s) for s in range(self.item_count)]

        del new_index["key0"]
        assert len(new_index.keys()) == self.item_count - 1
        try:
            new_index["key0"]
            raise
        except KeyError:
            pass
        del new_index[0]
        assert len(new_index.keys()) == self.item_count - 2
        try:
            new_index["key1"]
            raise
        except KeyError:
            pass
        try:
            del new_index["key1"]
            raise
        except KeyError:
            pass


        assert new_index[range(self.item_count - 2)] == \
            ["value" + str(s) for s in range(self.item_count)][2:]
        del new_index[["key" + str(s) for s in range(self.item_count)[2:]]]
        assert len(new_index.keys()) == 0
        del new_index

    def test_index_relabel(self):
        new_index = Index()
        new_index[["key" + str(s) for s in range(self.item_count)]] = \
            ["value" + str(s) for s in range(self.item_count)]

        new_index.relabel({"key0" : "new_key0", "key1": "new_key1"})
        new_index["new_key0"] = "new_value0"

        new_index.relabel({"new_key0": "key0", "new_key1" : "key1"})
        assert len(new_index.keys()) == self.item_count
        assert new_index["key0"] == "new_value0"
        assert new_index[["key" + str(s) for s in range(self.item_count)][1:]] == \
            ["value" + str(s) for s in range(self.item_count)][1:]
        del new_index

        pass

    def test_index_hierarchy(self):
        new_index = Index()
        all_keys = ["k/e/y/"+str(s) for s in range(self.item_count)]
        new_index[all_keys] = ["value"+str(s) for s in range(self.item_count)]

        # Test membership of hierarchical data
        assert(all(k in new_index for k in all_keys))
        
        # Test sequence of queries
        sub1_index = new_index.subset("k/")
        assert sub1_index.keys() == \
            ["e/y/"+str(s) for s in range(self.item_count)]
        sub2_index = sub1_index.subset("e/")
        assert sub2_index.keys() == \
            ["y/"+str(s) for s in range(self.item_count)]
        sub3_index = sub2_index.subset("y/")
        assert sub3_index.keys() == \
            [str(s) for s in range(self.item_count)]

        # Test multiple queries at once
        sub4_index = new_index.subset("k/e/")
        assert sub4_index.keys() == \
            ["y/"+str(s) for s in range(self.item_count)]
        sub5_index = new_index.subset("k/e/y/")
        assert sub5_index.keys() == \
            [str(s) for s in range(self.item_count)]

        # Test nonexistent queries
        sub6_index = new_index.subset("nonexistent_key/")
        assert sub6_index.keys()==[]
        assert len(sub6_index)==0

        # Test nonexistent keys in existing directories
        # Since this queries an exact key, it should throw an error
        try:
            sub7_index = new_index.subset("k/nonexistent_key")
            raise
        except KeyError:
            pass
        try:
            sub8_index = new_index.subset("k/e/nonexistent_key")
            raise
        except KeyError:
            pass
        try:
            sub9_index = new_index.subset("k/e/y/nonexistent_key")
            raise
        except KeyError:
            pass

        # Test nonexistent directories in existing directories
        # Since directories are "soft" this should never throw an error
        sub10_index = new_index.subset("k/nonexistent_dir/")
        assert sub10_index.keys()==[]
        assert len(sub10_index)==0
        sub11_index = new_index.subset("k/e/nonexistent_dir/")
        assert sub11_index.keys()==[]
        assert len(sub11_index)==0
        sub12_index = new_index.subset("k/e/y/nonexistent_dir/")
        assert sub12_index.keys()==[]
        assert len(sub12_index)==0

    def test_index_iter(self):
        new_index = Index()
        all_keys = ["k/e/y/"+str(s) for s in range(self.item_count)]
        all_values = ["value"+str(s) for s in range(self.item_count)]
        new_index[all_keys] = all_values

        i = 0
        for key in new_index:
            assert(all_keys[i]==key)
            assert(all_values[i]==new_index[key])
            i += 1

        # Test iteritems
        i = 0
        for key,val in new_index.iteritems():
            assert(all_keys[i]==key)
            assert(all_values[i]==val)
            i += 1

        for i in new_index.iterkeys():
            print i

        # Test reverse iteration
        i = len(new_index)-1
        for key in reversed(new_index):
            print all_keys[i], key
            assert all_keys[i]==key
            assert all_values[i]==new_index[key]
            i -= 1
