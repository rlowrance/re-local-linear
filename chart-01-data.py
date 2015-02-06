# create file WORKING/chart01-data.pickle containing tuple with
# dictionary objects median_price, num_transactions, both with keys
# (year, month)
#
# for all transactions (not just test nor train)

# import built-ins and libraries
import numpy as np
import pandas as pd
import sys
import cPickle as pickle

# import my stuff
from directory import directory
from Logger import Logger


class Control(object):
    def __init__(self):
        me = 'chart-01-data'
        working = directory('working')
        log = directory('log')

        self.path_out = working + me + '.pickle'
        self.path_out_log = log + me + '.log'
        self.path_in = working + 'transactions-subset2.pickle'

        self.testing = False


def summarize(df):
    '''
    Return dictionaries median_price and num_transactions, both with keys
    (year, month).
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

    return median_price, num_transactions


def main():

    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    df = pd.read_pickle(control.path_in)
    median_price, num_transactions = summarize(df)
    pickle.dump((median_price, num_transactions),
                open(control.path_out, 'wb'))

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v
    print 'done'

if __name__ == '__main__':
    main()
