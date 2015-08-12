'''Python Uniform Containers that mimic K's container types'''

# TODO: write documentation for V, H, T, KT
# goal: assure that the API is easy to describe

import abc
import collections
import numpy as np
import pandas as pd
import pdb
import unittest


class PUC(object):
    __metaclass__ = abc.ABCMeta
    pass


class V(PUC):
    __metaclass__ = abc.ABCMeta

    def __len__(self):
        return self.nparray.size

    def to_numpy_array(self):
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

    def _concatenate(self, other):
        'extend self by appending each element of other'
        # Q operator
        pass

    def _find(self, other):
        'return indices of each element of other in self'
        if isinstance(other, V):
            # np.append appends to copy of first arg
            return np.append(self.nparray, other.nparray)
        # Q operator
        pass

    def _equal(self, other):
        'order is significant'
        pass

    def _identical(self, other):
        'have same adddress'
        pass


class Vfloat64(V):
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

    def __radd__(self, other):
        'other + V'
        return self.__add__(other)

    def __add__(self, other):
        'return new Vfloat64'
        if isinstance(other, (Vfloat64, Vint64, Vbool)):
            return Vfloat64(np.add(self.nparray, other.nparray))
        if isinstance(other, (int, long, float)):
            # int, long, float
            return Vfloat64(np.add(self.nparray,
                                   np.full(self.nparray.shape, other)))
        raise TypeError('type(other) = ' + str(type(other)))

    def concatenate(self, other):
        'append other to self; do not type conversions'
        if isinstance(other, Vfloat64):
            # np.append appends to copy of first arg
            return Vfloat64(np.append(self.nparray, other.nparray))
        if isinstance(other, float):
            return Vfloat64(np.append(self.nparray, np.array([other])))
        raise TypeError('type(other) = ' + str(type(other)))


class Vint64(V):
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

    def __radd__(self, other):
        'other + V'
        return self.__add__(other)

    def __add__(self, other):
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

    def concatenate(self, other):
        'append other to self; do not type conversions'
        if isinstance(other, Vint64):
            # np.append appends to copy of first arg
            return Vint64(np.append(self.nparray, other.nparray))
        if isinstance(other, (int, long, bool)):
            return Vint64(np.append(self.nparray, np.array([other])))
        raise TypeError('type(other) = ' + str(type(other)))


class Vbool(V):  # TODO: Rename Vint8 (if its signed)
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

    def __radd__(self, other):
        'return new V: other + vbool'
        return self.__add__(other)

    def __add__(self, other):
        'return new V: Vbool + other'
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

    def concatenate(self, other):
        'append other to self; do not type conversions'
        if isinstance(other, Vbool):
            # np.append appends to copy of first arg
            return Vbool(np.append(self.nparray, other.nparray))
        if isinstance(other, (bool)):
            return Vbool(np.append(self.nparray, np.array([other])))
        raise TypeError('type(other) = ' + str(type(other)))


class Vobj(V):
    'vector of arbitrary objects'
    def __init__(self, obj):
        self.nparray = np.array(
            obj,
            dtype=np.object,
            copy=True,
            order='C',
            subok=False,
            ndmin=1)

    def __str__(self):
        return 'Vobj(' + super(Vobj, self)._items() + ')'

    def __repr__(self):
        return 'Vobj([' + super(Vobj, self)._items() + '])'

    def __getitem__(self, key):
        return self._getitem(Vobj, key)

    def __setitem__(self, key, value):
        self._setitem(key, value)

    def __radd__(self, other):
        'other + VObj'
        # TODO: fix, does not work for string concatenation
        return self.__add__(other)

    def __add__(self, other):
        'Vobj + other'
        if isinstance(other, V):
            if len(self.nparray) != len(other.nparray):
                msg = 'different lengths: %d, %d' % (len(self.nparray), len(other.nparray))
                raise TypeError(msg)
            result = np.empty(shape=(len(self.nparray)), dtype=object)
            for i in xrange(len(self.nparray)):
                a = self.nparray[i]
                b = other.nparray[i]
                try:
                    r = a + b
                except TypeError, m:
                    msg = 'type(a) %s type(b) %s' % (type(a), type(b))
                    raise TypeError(m + ' ' + msg)
                result[i] = r
            return Vobj(result)
        raise NotImplemented('implement other cases')

    def concatenate(self, other):
        if isinstance(other, V):
            return Vobj(np.append(self.nparray, other.nparray))
        try:
            return Vobj(np.append(self.nparray, other))
        except:
            raise TypeError('other is type %s' % type(other))
        raise TypeError('other is type %s' % type(other))


class D(PUC):  # TODO: rename to H for hash; want name not in base Python
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

    def concatenate(self, other):
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
        '''Perform + on command keys; others merge (as in concatenate)
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

    def test_plus(self):
        # plus(float, float)
        va = Vfloat64([10, 20])
        vb = Vfloat64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # plus(float, int)
        vb = Vint64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # plus(float, bool)
        vb = Vbool([False, True])
        r = va + vb
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[1], 21)

        # test propagation of scalars: float, int, bool
        r = va + 1.0
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))
        r = va + 2
        self.assert_equal_Vfloat64(r, Vfloat64([12, 22]))
        r = va + True
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))

    def test_concatenate(self):

        # other is Vfloat64
        va = Vfloat64([10, 20])
        vb = Vfloat64([100, 200])
        r = va.concatenate(vb)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 4)
        self.assertEqual(r[0], 10.0)
        self.assertEqual(r[3], 200.0)

        # other is float
        r = va.concatenate(23.0)
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 10.0)
        self.assertEqual(r[2], 23.0)

        # other is anything else
        for other in (True, 23, Vint64([100, 200]), Vbool([False, True])):
            try:
                v = va.concatenate(other)
                print other, v
                self.assertTrue(False)  # expected to throw
            except TypeError:
                self.assertTrue(True)


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

    def test_plus(self):
        # plus(int, float)
        va = Vint64([10, 20])
        vb = Vfloat64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110.0)
        self.assertEqual(r[1], 220.0)

        # plus(int, int)
        vb = Vint64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 110)
        self.assertEqual(r[1], 220)

        # plus(int, bool)
        vb = Vbool([False, True])
        r = va + vb
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[1], 21)

        # test propagation of scalars: float, int, bool
        r = va + 1.0
        self.assert_equal_Vfloat64(r, Vfloat64([11, 21]))
        r = va + 2
        self.assert_equal_Vint64(r, Vint64([12, 22]))
        r = va + True
        self.assert_equal_Vint64(r, Vint64([11, 21]))

    def test_concatenate(self):

        # other is Vint64
        va = Vint64([10, 20])
        vb = Vint64([100, 200])
        r = va.concatenate(vb)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 4)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[3], 200)

        # other is long
        r = va.concatenate(23L)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[2], 23)

        # other is bool
        r = va.concatenate(True)
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 10)
        self.assertEqual(r[2], 1)

        # other is anything else
        others = (23.0, Vfloat64([100, 200]), Vbool([False, True]))
        for other in others:
            try:
                v = va.concatenate(other)
                print other, v
                self.assertTrue(False)  # expected to throw
            except TypeError:
                self.assertTrue(True)


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

    def test_plus(self):
        # plus(bool, float)
        va = Vbool([False, True])
        vb = Vfloat64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vfloat64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 100)
        self.assertEqual(r[1], 201)

        # plus(bool, int)
        vb = Vint64([100, 200])
        r = va + vb
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 100)
        self.assertEqual(r[1], 201)

        # plus(bool, bool)
        vb = Vbool([False, True])
        r = va + vb  # numpy treats this as or, not +
        self.assertTrue(isinstance(r, Vint64))
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[1], 2)

        # test propagation of scalars: float, int, bool
        r = va + 1.0
        self.assert_equal_Vfloat64(r, Vfloat64([1, 2]))
        r = va + 2
        self.assert_equal_Vint64(r, Vint64([2, 3]))
        r = va + True
        self.assert_equal_Vint64(r, Vint64([1, 2]))

    def test_concatenate(self):

        # other is Vbool
        va = Vbool([False, True])
        vb = Vbool([True, False])
        r = va.concatenate(vb)
        self.assertTrue(isinstance(r, Vbool))
        self.assertEqual(len(r), 4)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[3], 0)

        # other is bool
        r = va.concatenate(True)
        self.assertTrue(isinstance(r, Vbool))
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[2], 1)

        # other is anything else
        others = (20, 23.0, Vfloat64([100, 200]), Vint64([20]))
        for other in others:
            try:
                v = va.concatenate(other)
                print other, v
                self.assertTrue(False)  # expected to throw
            except TypeError:
                self.assertTrue(True)


class TestVobj(unittest.TestCase):
    def assert_equal_Vfloat64(self, a, b):
        self.assertTrue(isinstance(a, Vfloat64))
        self.assertTrue(isinstance(b, Vfloat64))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    def assert_equal_Vobj(self, a, b):
        self.assertTrue(isinstance(a, Vobj))
        self.assertTrue(isinstance(b, Vobj))
        self.assertEqual(len(a), len(b))
        for i in xrange(len(a)):
            self.assertEqual(a[i], b[i])

    def test_construction_from_list(self):
        f64 = Vfloat64([10, 20])
        x = [True, 23.0, f64, 'abc']
        v = Vobj(x)
        self.assertTrue(isinstance(v, Vobj))
        self.assertTrue(isinstance(v, V))
        self.assertEqual(len(v), 4)
        self.assertEqual(v[0], True)
        self.assertEqual(v[1], 23.0)
        self.assert_equal_Vfloat64(v[2], f64)
        self.assertEqual(v[3], 'abc')

    def test_add(self):
        # Vobj + Vobj
        va = Vobj([10, Vfloat64([100, 200]), 'abc'])
        vb = Vobj([20, 1, 'def'])
        r = va + vb
        self.assertTrue(isinstance(r, Vobj))
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 30.0)
        self.assert_equal_Vfloat64(r[1], Vfloat64([101, 201]))
        self.assertEqual(r[2], 'abcdef')
        r2 = vb + va
        self.assertEqual(r[0], r2[0])
        self.assert_equal_Vfloat64(r[1], r2[1])
        self.assertNotEqual(r[2], r2[2])

    def test_concatenate(self):
        va = Vobj(['a', 10])
        vb = Vobj([True, 23.0])
        r = va.concatenate(vb)
        self.assertEqual(len(r), 4)
        self.assertEqual(r[0], 'a')
        self.assertEqual(r[3], 23.0)
        r = va.concatenate('abc')
        self.assertEqual(len(r), 3)
        self.assertEqual(r[0], 'a')
        self.assertEqual(r[2], 'abc')


class TestD(unittest.TestCase):
    def test_construction_from_two_lists(self):
        self.assertTrue(False)  # write me


if __name__ == '__main__':
    if False:
        # avoid warnings from pyflakes
        pdb.set_trace()
    unittest.main()
