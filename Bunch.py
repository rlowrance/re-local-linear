'''Collect a bunch of named items (like a C struct)

ref: Python Cookbook "Collecting a Bunch of Named Items"

usage
  point = Bunch(datum=y, squared=y*y)
  if point.squared > threshold:
      point.is_ok = True
'''
import pdb
import unittest


class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class Test(unittest.TestCase):
    def test(self):
        struct = Bunch(x=10, y=20)
        struct.total = struct.x + struct.y
        self.assertEqual(struct.x, 10)
        self.assertEqual(struct.total, 30)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
