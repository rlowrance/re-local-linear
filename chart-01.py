# create file WORKING/chart01.pdf containing chart depicting median
# prices by month

# import built-ins and libraries
import numpy as np
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
from datetime import datetime

# import my stuff
from directory import directory
from Logger import Logger


class Control(object):
    def __init__(self):
        me = 'chart-01'
        working = directory('working')
        log = directory('log')

        self.path_out = working + me + '.pdf'
        self.path_out_log = log + me + '.log'
        self.path_in = working + me + '-data.pickle'

        self.testing = False


def make_figure(median_price):
    '''
    Return bar chart Figure
    '''

    # build nparrays used an input to matplotlib
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


def main():

    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    median_price, num_transactions = pickle.load(open(control.path_in, 'rb'))
    figure = make_figure(median_price)
    figure.savefig(control.path_out)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v
    print 'done'

if __name__ == '__main__':
    main()
