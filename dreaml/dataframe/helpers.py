def is_scalar(x):
    return isinstance(x,(long,float,int))

def len_iter(it):
    return sum(1 for _ in it)