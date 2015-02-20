# create file (selective on how invoked)
# WORKING/chart-01.data pickled file
# WORKING/chart-01.pdf containing chart depicting median
# prices by month

# import built-ins and libraries
import numpy as np
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import pdb

# import my stuff
from directory import directory
from Logger import Logger

from Record import Record


def print_help():
    print 'python chart-01.py SUFFIX'
    print 'where SUFFIX in {"makefile", "data", "txt"}'


class Control(Record):
    def __init__(self, arguments):
        Record.__init__(self, 'control')

        me = 'chart-01'

        log = directory('log')
        src = directory('src')
        working = directory('working')

        # handle command line arguments
        if len(arguments) != 2:
            print_help()
            raise RuntimeError('missing command line argument')

        self.suffix = arguments[1]

        self.path_in = working + 'transactions-subset2.pickle'

        self.path_out_data = working + me + '.data'
        self.path_out_log = log + me + '.log'
        self.path_out_pdf = working + me + '.pdf'
        self.path_out_makefile = src + me + '.makefile'

        self.testing = False


def create_data(control):
    '''Write data file, pickled, to working directory.

    The data file ks is a dict containing median prices
    key = (year, month)
    value = median price of sales in that year and month
    '''

    def make_median_prices(df):
        '''Return list of 2 dictionaries, median_prices, num_transactions.
        '''
        median_price = {}
        num_transactions = {}
        years = df['sale.year']
        for year in xrange(int(np.amin(years)), int(np.amax(years)) + 1):
            for month in xrange(1, 13 if year != 2009 else 4):
                # select subset of transactions in year and month
                in_year = df['sale.year'] == year
                in_month = df['sale.month'] == month
                in_both = np.logical_and(in_year, in_month)
                selected = df[in_both]
                mp = np.median(selected['SALE.AMOUNT'])
                median_price[(year, month)] = mp
                nt = in_both.sum()
                num_transactions[(year, month)] = nt
                print year, month, mp, nt

        return (median_price, num_transactions)

    df = pd.read_pickle(control.path_in)
    data = make_median_prices(df)
    f = open(control.path_out_data, 'wb')
    pickle.dump(data, f)
    f.close()


def create_pdf(control):
    def make_figure(median_price):
        '''
        Return bar chart Figure
        '''

        # build nparrays used an input to matplotlib
        pdb.set_trace()
        x_list = []
        y_list = []
        for k, v in median_price.iteritems():
            print k, v
            x_list.append(datetime(k[0], k[1], 1))  # first day of month
            y_list.append(v)

        x = np.array(x_list)
        y = np.array(y_list)

        # create the figure
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)  # 1 row, 1 col, first Axes
        ax = plt.subplot(111)
        ax.bar(x, y, width=10)
        ax.xaxis_date()
        ax.set_title('Median Prices by Month')
        ax.set_xlabel('Year and Month')
        ax.set_ylabel('Median Price ($)')
        return figure

    f = open(control.path_out_data, 'rb')
    median_price, num_transactions = pickle.load(f)
    figure = make_figure(median_price)
    figure.savefig(control.path_out_pdf)


def create_makefile(control):
    '''Write makefile to source directory.'''

    def make_lines():
        '''Produce lines for makefile.
        '''
        lines = []
        lines.append('chart-01.data: chart-01.py')
        lines.append('\t$(PYTHON) chart-01.py data')
        lines.append('chart-01.pdf: chart-01.data chart-01.py')
        lines.append('\t$(PYTHON) chart-01.py pdf')
        lines.append('chart-01.makefile: chart-01.py')
        lines.append('\t$(PYTHON) chart-01.py makefile')
        return lines

    lines = make_lines()
    f = open(control.path_out_makefile, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()


def main():

    control = Control(sys.argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control

    if control.suffix == 'makefile':
        create_makefile(control)
    elif control.suffix == 'data':
        create_data(control)
    elif control.suffix == 'pdf':
        create_pdf(control)
    else:
        print_help()
        raise RuntimeError('bad command SUFFIX')

    # clean up
    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'


if __name__ == '__main__':
    main()
