def round_to_n(x, n):
    '''
    Round x to n significant digits.
    '''
    from math import log10, floor
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):
        def test_one(self):
            self.assertEqual(round_to_n(1234, 1), 1000)

        def test_two(self):
            self.assertEqual(round_to_n(1234.456, 2), 1200)

        def test_negative(self):
            self.assertEqual(round_to_n(-1234, 4), -1234)

    unittest.main()
