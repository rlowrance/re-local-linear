'''Python Uniform Containers that mimic K's container types'''

import abc
import collections
import numpy as np
import pandas as pd
import pdb
import unittest


class PUC(object):
    __metaclass__ = abc.ABCMeta

    def __len__(self):
        return self.nparray.size

    def as_numpy_array(self):
        return self.nparray

    def _items(self):
        s = ''
        for i in xrange(self.nparray.size):
            s += self.nparray[i].__str__()
            if i != self.nparray.size - 1:
                s += ', '
        return s

    def _validate_len_key(self, key):
        'continue if len(key) is ok'
        if len(key) == self.nparray.size:
            return
        format = 'len(key) %s != len(data) %s'
        msg = format % (len(key), self.nparray.size)
        raise IndexError(msg)

    def _getitem(self, Cls, key):
        '''return Cls instance or a python scalar or raise IndexError

        ARGS
        key : if plain integer or long integer
              then return a python float

              if Vint64
              then return a Vfloat64 as selected by index values

              if Vint1
              then return a Vfloat64 as selected by 1 values
        '''
        # note: in Q, indexing a missing item result in null, not an error
        if isinstance(key, (int, long)):
            # note: in Python 3, int and long are the same type
            return self.nparray[key]
        if isinstance(key, bool):
            return self.nparray[int(key)]
        if isinstance(key, Vint64):
            return Cls(self.nparray[key.nparray])
        if isinstance(key, Vbool):
            self._validate_len_key(key)
            return Cls(self.nparray[key.nparray])
        # TODO: extend to allow indexing by Vobj
        raise IndexError('type(key) = ' + type(key))

    def _setitem(self, key, value):
        '''mutate self and return None'''
        if isinstance(key, (int, long)):
            self.nparray[key] = value
            return
        if isinstance(key, bool):
            self.nparray[int(key)] = value
        if isinstance(key, Vint64):
            self.nparray[key.nparray] = value
            return
        if isinstance(key, Vbool):
            self._validate_len_key(key)
            self.nparray[key.nparray] = value
            return
        raise IndexError('type(key) = ' + type(key))

    def _join(self, other):
        'extend self by appending each element of other'
        # Q operator
        pass

    def _find(self, other):
        'return indices of each element of other in self'
        # Q operator
        pass

    def _equal(self, other):
        'order is significant'
        pass

    def _identical(self, other):
        'have same adddress'
        pass


class Vfloat64(PUC):
    '64-bit floating point vector'

    def __init__(self, obj):
        self.nparray = np.array(
            obj,
            dtype=np.float64,
            copy=True,
            order='C',
            subok=False,
            ndmin=1)

    def __str__(self):
        return 'Vfloat64(' + super(Vfloat64, self)._items() + ')'

    def __repr__(self):
        return 'Vfloat64([' + super(Vfloat64, self)._items() + '])'

    def __getitem__(self, key):
        return self._getitem(Vfloat64, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def exp(self):
        'return new Vfloat64'
        return Vfloat64(np.exp(self.nparray))

    def add(self, other):
        'return new Vfloat64'
        if isinstance(other, (Vfloat64, Vint64, Vbool)):
            return Vfloat64(np.add(self.nparray, other.nparray))
        if isinstance(other, (int, long, float)):
            # int, long, float
            return Vfloat64(np.add(self.nparray,
                                   np.full(self.nparray.shape, other)))
        raise TypeError('type(other) = ' + type(other))


class Vint64(PUC):
    '64-bit integer vector'

    def __init__(self, obj):
        self.nparray = np.array(
            obj,
            dtype=np.int64,
            copy=True,
            order='C',
            subok=False,
            ndmin=1)

    def __str__(self):
        return 'Vint64(' + super(Vint64, self)._items() + ')'

    def __repr__(self):
        return 'Vint64([' + super(Vint64, self)._items() + '])'

    def __getitem__(self, key):
        return self._getitem(Vint64, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def add(self, other):
        'return new V'
        if isinstance(other, Vfloat64):
            return Vfloat64(np.add(self.nparray, other.nparray))
        if isinstance(other, (Vint64, Vbool)):
            return Vint64(np.add(self.nparray, other.nparray))
        if isinstance(other, float):
            # int, long, float
            return Vfloat64(np.add(self.nparray,
                                   np.full(self.nparray.shape, other)))
        if isinstance(other, (int, long, float)):
            # int, long, float
            if isinstance(other, float):
                return Vfloat64(np.add(self.nparray,
                                       np.full(self.nparray.shape, other)))
            if isinstance(other, (int, bool)):
                return Vint64(np.add(self.nparray,
                                     np.full(self.nparray.shape, other)))
        raise TypeError('type(other) = ' + type(other))


class Vbool(PUC):
    'boolean vector'

    def __init__(self, obj):
        self.nparray = np.array(
            obj,
            dtype=np.bool,  # TODO: make smaller
            copy=True,
            order='C',
            subok=False,
            ndmin=1)

    def __str__(self):
        return 'Vbool(' + super(Vbool, self)._items() + ')'

    def __repr__(self):
        return 'Vbool([' + super(Vbool, self)._items() + '])'

    def __getitem__(self, key):
        return self._getitem(Vbool, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def add(self, other):
        'return new V'
        if isinstance(other, Vfloat64):
            return Vfloat64(np.add(self.nparray, other.nparray))
        if isinstance(other, Vint64):
            return Vint64(np.add(self.nparray, other.nparray))
        if isinstance(other, Vbool):
            # note: numpy treats + for bools as "or", not as "and"
            return Vint64(np.add(self.nparray.astype(long),
                                 other.nparray.astype(long)))
        if isinstance(other, (int, long, float)):
            # int, long, float
            if isinstance(other, (int, bool)):
                return Vint64(np.add(self.nparray,
                                     np.full(self.nparray.shape, other)))
            if isinstance(other, float):
                return Vfloat64(np.add(self.nparray,
                                       np.full(self.nparray.shape, other)))
        raise TypeError('type(other) = ' + type(other))


def Vobj(PUC):
    'vector of arbitrary objects'
    pass


class D(object):
    'dictionary with [] extended to allow for a sequence'

    def __init__(self, key_list, value_list):
        '''initialize'''
        # Note: in Q, the keys do not need to be unique
        self.d = None  # TODO: write me

    def keys(self):
        '''return list of keys'''
        pass

    def cols(self):
        return self.keys()

    def values(self):
        '''return list of values'''

    def equal(self, other):
        '''order of keys is significant; not the identity operator

        If other is a V, compare self.values and other
        '''
        pass

    def identical(self, other):
        'have the same address'
        pass

    def find(self, other):
        'reverse lookup; always succeeds, possibly returning None'
        pass

    def join(self, other):
        'the mapping in other dominates'
        pass

    def as_python_dict(self):
        '''return Python dict'''
        return self.d

    def take(self, keys):
        '''return new D with the keys and self[keys]'''
        pass

    def __del__(self, key):
        '''delete key and value

        in Q, removing a key that does not exist has no effect
        '''
        pass

    def add(self, other):
        '''Perform + on command keys; others merge (as in join)
        '''
        pass

    def __getitem__(self, key):
        '''return Python list of same shape as key or scalar or None

        ARGS
        key : if scalar
              then return d[key]

              if a sequence (including a V)
              then return {v[0], v[1], ... }

        In Q, if keys are not of uniform shape, the lookup fails
        at the first key of a different shape.
        '''
        if isinstance(key, Vfloat64):
            raise IndexError('attempt to index D by Vfloat64')
        if isinstance(key, (Vint64, Vbool)):
            result = []
            for key_value in np.nditer(key.nparray):
                if key_value in self.d:
                    result.append(self.d[key_value])
                else:
                    result.append(None)
            return key_value
        if isinstance(key, collections.Iterable):
            result = []
            for key_value in key:
                if key_value in self.d:
                    result.append(self.d[key_value])
                else:
                    result.append(None)
            return key_value
        if key in self.d:
            return self.d[key]
        else:
            return None

    def __setitem__(sefl, key, value):
        '''mutate and return self'''
        pass


class T(object):
    'a flipped dictionary with string keys and V values, all of same len'

    def __init__(self, obj):
        assert(type(obj) == D)
        self.d = obj

    def as_pandas_dataframe(self):
        '''return Pandas DataFrame with numpy.array columns'''
        items = [(k, self.d[k].as_numpy_array()) for k in self.d.keys()]
        return pd.from_items(items, orient='columns')

    def __getitem__(self, key):
        'maybe also return a python scalar'
        rows, cols = key
        pass  # figure out the combinations

    def __setitem__(self, key, value):
        pass

    def ply(self, vars, fun, extra):
        '''apply fun to groups defined by vars

        Inspired by the plyr package in R

        ARGS
        vars: a list of strings, each string is a column name in self
        fun(var_values, extra):  a function returning a T with
              the same columns for each invocation

        RETURNS
        new T with one column for each var plus one result column from each T
        returned by the calls to fun
        '''
        pass

    # these methods are inspired by Wickham's dplyr package for R

    def filter(self, vars, fun, extra):
        'return new T for which fun(var_values, extra) returns true'
        pass

    def select_rows(self, slice):
        'return new T with rows specified by the python slice'''
        pass

    def select_columns(self, vars):
        'return new T with columns as specified by the vars'
        pass

    def rename(self, old_new_names):
        '''return new T selectively updating column names

        ARGS
        old_new_names = ((old_name,new_name), ...)
        '''
        pass

    def distinct(self):
        '''return new T without duplicated rows'''
        pass

    def mutate(self, vars, fun, extra):
        '''like ply, but mutate self by adding the extra columns'''
        pass

    def summarize(self, fun, extra):
        '''return new T formed by passing each row to fun(var_values, extra)'''
        pass

    def sample_n():
        '''return new T with n randomly-selected rows'''
        pass


class TestVfloat64(unittest.TestCase):
    def assert_equal_Vfloat64(self, a, b):
        self.assertTrue(isinstance(a, Vfloat64))
        self.assertTrue(isinstance(b, Vfloat64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def test_construction_from_list(self):
        x = [10, 23]
        v = Vfloat64(x)
        self.assertTrue(isinstance(v, Vfloat64))
        self.assertTrue(isinstance(v, PUC))
        self.assertTrue(len(v) == 2)
        # TODO: these don't work until __getitem__ is implemented
        self.assertTrue(v[0] == 10.0)
        self.assertTrue(v[1] == 23.0)

    def test_get_item(self):

        v = Vfloat64([10.0, 23.0])

        # index via Vint64
        index1 = Vint64([1, 1, 1])
        v_index1 = v[index1]
        self.assertTrue(isinstance(v_index1, Vfloat64))
        self.assertEqual(len(v_index1), 3)
        self.assertEqual(v_index1[0], 23.0)
        self.assertEqual(v_index1[1], 23.0)
        self.assertEqual(v_index1[2], 23.0)

        # index via Python int
        index2 = 1
        v_index2 = v[index2]
        self.assertTrue(isinstance(v_index2, float))
        self.assertEqual(v_index2, 23.0)

        # index via Vbool
        index3 = Vbool([True, False])
        v_index3 = v[index3]
        self.assertTrue(isinstance(v_index3, Vfloat64))
        self.assertEqual(len(v_index3), 1)
        self.assertEqual(v_index3[0], 10.0)

        # show throw: too few boolean indices
        try:
            index4 = Vbool([True])
            v_index4 = v[index4]
            print v_index4
            self.assertFalse(True)
        except:
            self.assertTrue(True)

    def test_set_item(self):

        # index via Vint64
        v = Vfloat64([10.0, 23.0, 47.0])
        index1 = Vint64([2, 0])
        v[index1] = 99.0
        self.assertTrue(isinstance(v, Vfloat64))
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 99.0)
        self.assertEqual(v[1], 23.0)
        self.assertEqual(v[2], 99.0)

        # index via Python int
        v = Vfloat64([10.0, 23.0, 47.0])
        v[1] = 99.0
        self.assertTrue(isinstance(v, Vfloat64))
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 10.0)
        self.assertEqual(v[1], 99.0)
        self.assertEqual(v[2], 47.0)

        # index via Vbool
        v = Vfloat64([10.0, 23.0, 47.0])
        index3 = Vbool([True, False, True])
        v[index3] = 99.0
        self.assertTrue(isinstance(v, Vfloat64))
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 99.0)
        self.assertEqual(v[1], 23.0)
        self.assertEqual(v[2], 99.0)

    def test_exp(self):
        v = Vfloat64([1, 2, 3])
        e = v.exp()
        self.assertEqual(len(e), 3)
        self.assertAlmostEqual(v[2], 3)
        self.assertAlmostEqual(e[2], 20.08, 1)

    def test_add(self):
        # add(float, float)
        va = Vfloat64([10, 20])
        vb = Vfloat64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # add(float, int)
        vb = Vint64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # add(float, bool)
        vb = Vbool([False, True])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[1], 21)

        # test propagation of scalars: float, int, bool
        r = va.add(1.0)
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))
        r = va.add(2)
        self.assert_equal_Vfloat64(r, Vfloat64([12, 22]))
        r = va.add(True)
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))


class TestVint64(unittest.TestCase):
    def assert_equal_Vfloat64(self, a, b):
        self.assertTrue(isinstance(a, Vfloat64))
        self.assertTrue(isinstance(b, Vfloat64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def assert_equal_Vint64(self, a, b):
        self.assertTrue(isinstance(a, Vint64))
        self.assertTrue(isinstance(b, Vint64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def test_construction_from_list(self):
        x = [10, 23]
        v = Vint64(x)
        self.assertTrue(isinstance(v, Vint64))
        self.assertTrue(isinstance(v, PUC))
        self.assertTrue(len(v) == 2)
        self.assertEqual(v[0], 10)
        self.assertEqual(v[1], 23)

    def test_add(self):
        # add(int, float)
        va = Vint64([10, 20])
        vb = Vfloat64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # add(int, int)
        vb = Vint64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110)
        self.assertEqual(r[1], 220)

        # add(int, bool)
        vb = Vbool([False, True])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[1], 21)

        # test propagation of scalars: float, int, bool
        r = va.add(1.0)
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))
        r = va.add(2)
        self.assert_equal_Vint64(r, Vint64([12, 22]))
        r = va.add(True)
        self.assert_equal_Vint64(r, Vint64([11, 21]))


class TestVbool(unittest.TestCase):
    def assert_equal_Vfloat64(self, a, b):
        self.assertTrue(isinstance(a, Vfloat64))
        self.assertTrue(isinstance(b, Vfloat64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def assert_equal_Vint64(self, a, b):
        self.assertTrue(isinstance(a, Vint64))
        self.assertTrue(isinstance(b, Vint64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def test_construction_from_list(self):
        x = [False, True]
        v = Vbool(x)
        self.assertTrue(isinstance(v, Vbool))
        self.assertTrue(isinstance(v, PUC))
        self.assertTrue(len(v) == 2)
        self.assertEqual(v[0], False)
        self.assertEqual(v[1], True)

    def test_add(self):
        # add(bool, float)
        va = Vbool([False, True])
        vb = Vfloat64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 100)
        self.assertEqual(r[1], 201)

        # add(bool, int)
        vb = Vint64([100, 200])
        r = va.add(vb)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 100)
        self.assertEqual(r[1], 201)

        # add(bool, bool)
        vb = Vbool([False, True])
        r = va.add(vb)  # numpy treats this as or, not +
        print r
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[1], 2)

        # test propagation of scalars: float, int, bool
        r = va.add(1.0)
        self.assert_equal_Vfloat64(r, Vfloat64([1, 2]))
        r = va.add(2)
        self.assert_equal_Vint64(r, Vint64([2, 3]))
        r = va.add(True)
        self.assert_equal_Vint64(r, Vint64([1, 2]))


class TestVobj(unittest.TestCase):
    def test_construction_from_list(self):
        self.assertTrue(False)  # write me


class TestD(unittest.TestCase):
    def test_construction_from_two_lists(self):
        self.assertTrue(False)  # write me


if __name__ == '__main__':
    if False:
        # avoid warnings from pyflakes
        pdb.set_trace()
    unittest.main()
