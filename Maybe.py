NOVALUE = object()


class Maybe(object):
    '''Similar to Haskell's Maybe type.

    Based on code.activestate.com/recipes/577248-maybe-pattern
    '''

    _has_value = False
    _value = None

    def __init__(self, value):
        if value is not NOVALUE:
            self._has_value = True
            self._value = value

    @property
    def has_value(self):
        return self._has_value

    @property
    def value(self):
        return self._value

    def __repr__(self):
        if self.has_value:
            return 'Maybe(%s)' % self.value
        else:
            return 'NOVALUE'


# Sugar factories
def NoValue():
    return Maybe(NOVALUE)

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):
        def test_has_value(self):
            x = Maybe(123)
            self.assertTrue(x.has_value)
            self.assertTrue(x.value == 123)

            y = Maybe(None)
            self.assertTrue(y.has_value)
            self.assertTrue(y.value is None)

        def test_has_novalue(self):
            x = NoValue()
            self.assertFalse(x.has_value)

    unittest.main()
