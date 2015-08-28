'''determine total size in bytes of a python object

ref: code.activestate.com/recipes/577504/
'''

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    '''Return approximate memory footprint of object and all of its content

    Knows how to handle: tuple, list, deque, dict, set, frozenset.

    To handle other containers, add handlers:
        handlers = {SomeContainClass: iter,
                    OtherContainerCLass: OtherContainClass.get_elements}
    '''
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: lambda d: chain.from_iterable(d.items()),
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()                   # track object id's that have been processes
    default_size = getsizeof(0)    # estimated sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:          # don't double count
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


if __name__ == '__main__':
    d = dict(a=1, b=2, c=3, d=[4, 5, 6, 7], e='a string of chars')
    print(total_size(d, verbose=True))
    import numpy as np
    # TODO: extend to handle numpy arrays
    x = np.array([10, 20, 30], dtype=np.float64)
    print(total_size(x, verbose=True))
    y = np.empty(1000, dtype=np.float64)
    print(total_size(x, verbose=True))
    # TODO: extend to handle pandas Series and DataFrames
