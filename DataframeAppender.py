'create data frame and append items to it'


import pandas as pd
import pdb


class DataframeAppender(object):
    def __init__(self, name_dtype):
        self.names = []
        self.dtypes = []
        for name, dtype in name_dtype:
            self.names.append(name)
            self.dtypes.append(dtype)
        self.df = self._make(len(self.names) * [None])  # initialize

    def append(self, lst):
        assert(len(lst) == len(self.names))
        new_df = self._make(lst)
        self.df = self.df.append(new_df, ignore_index=True)
        pass

    def result(self):
        return self.df

    def _make(self, lst):
        def maybe(x):
            return [] if x is None else [x]

        def s(x, dtype):
            return pd.Series(maybe(x), dtype=dtype)

        assert(len(lst) == len(self.names))
        d = {}
        for i in xrange(len(self.names)):
            d[self.names[i]] = s(lst[i], self.dtypes[i])
        return pd.DataFrame(d)


if __name__ == '__main__':
    if False:  # avoid pyflakes warning
        pdb.set_trace()
