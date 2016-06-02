
# valid options: index, hindex
WHICH_INDEX = "hindex"

if WHICH_INDEX == "index":
    print "Using flat Index"
elif WHICH_INDEX == "hindex":
    print "Using hierarchical Index"
else:
    raise "Invalid WHICH_INDEX"
