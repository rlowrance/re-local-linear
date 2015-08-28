'''Example of an abstract class

ref: pymotws.com/2/abc
'''

import abc


class Abstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def method1(self, arg1):
        '''provide concrete implementation for use by derived classes'''
        return -23

    @abc.abstractmethod
    def method2(self, arg1, arg2):
        '''provide no implementation'''
        pass


class Concrete(Abstract):

    def method1(self, arg1, arg2):
        # use abstract class's concrete method
        base_data = super(Concrete, self).method1(arg1)
        return abs(base_data)  # or other transformation of base_data

    def method2(self, arg1, arg2):
        return arg1 * arg2


if __name__ == '__main__':
    concrete = Concrete()
    print concrete.method1(10, 20)
    print concrete.method2(10, 20)
