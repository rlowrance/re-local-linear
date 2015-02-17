# create files
# WORKING/chart-03.makefile  (empty)
# WORKING/chart-03.data
#   dict: key = (year, month); value = [{'price':,'assesment':},...]
# WORKING/chart-03.txt     containing chart median losses across fold


# import built-ins and libraries
import sys
import pdb
import cPickle as pickle

# import my stuff
from directory import directory
from Logger import Logger


def print_help():
    print 'python chart-03.py SUFFIX'
    print 'where SUFFIX in {"makefile", "data", "txt"}'


class Control(object):
    def __init__(self, arguments):
        self.me = 'chart-03'

        working = directory('working')
        log = directory('log')
        cells = directory('cells')
        src = directory('src')
        self.dir_working = working
        self.dir_log = log
        self.dir_cells = cells
        self.dir_src = src

        # handle command line arguments
        if len(arguments) != 2:
            print print_help()
            raise RuntimeError('missing command line argument')

        self.suffix = arguments[1]

        self.path_out_log = log + self.me + '.log'
        self.path_dir_cells = cells
        self.path_training = working + 'transactions-subset2.pickle'

        # output files
        self.path_data = working + self.me + '.data'
        self.path_txt = working + self.me + '.txt'
        self.path_makefile = src + self.me + '.makefile'

        # components of cell names
        self.model = 'ols'
        self.responses = ['price', 'logprice']
        self.predictors = ['act', 'actlog', 'ct', 'ctlog']
        self.year = '2008'
        self.training_periods = ['30', '60', '90', '120', '150', '180',
                                 '210', '240', '270', '300', '330', '360']

        self.testing = False
        self.debugging = False

    def __str__(self):
        s = ''
        for kv in sorted(self.__dict__.items()):
            k, v = kv
            s = s + ('control.%s = %s\n' % (k, v))
        return s


class Report(object):

    def __init__(self, lines):
        self.lines = lines
        self.format_header = '{:>8s} ' * 4
        self.format_detail = '{:>8d}' * 3 + ' {:>8.2f}'

    def header(self, c0, c1, c2, c3):
        print c0, c1, c2, c3
        s = self.format_header.format(c0, c1, c2, c3)
        self.lines.append(s)

    def detail(self, year, month, count, fraction_zero_error):
        print year, month, count, fraction_zero_error
        s = self.format_detail.format(year, month, count, fraction_zero_error)
        self.lines.append(s)


def create_txt(control):
    '''Return list of lines that are chart 03.txt.
    '''
    def append_description(lines):
        '''Append header lines'''
        lines.append('Fraction of Sales for Exactly Their 2008 Assessement')
        lines.append(' ')

    def read_data():
        '''Return data dict built by create_data() function.'''
        f = open(control.path_data, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def append_header(t):
        t.header('year', 'month', 'nSales', 'fracZero')

    def append_detail_line(item, report):
        time_period, a_list = item
        year, month = time_period
        count = len(a_list)
        count_zero_error = 0
        fraction_zero_error = 0
        for d in a_list:
            assessment = d['assessment']
            price = d['price']
            error = assessment - price
            if error == 0:
                count_zero_error += 1
                fraction_zero_error = count_zero_error / (1.0 * count)
        report.detail(int(year),
                      int(month),
                      int(count),
                      fraction_zero_error)

    def append_detail_lines(report, data):
        '''Append body lines to report using data.'''
        for item in sorted(data.items()):
            append_detail_line(item, report)

    def write_lines(lines):
        f = open(control.path_txt, 'w')
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    lines = []
    append_description(lines)
    report = Report(lines)
    append_header(report)
    data = read_data()
    append_detail_lines(report, data)
    write_lines(lines)


def create_data(control):
    '''Write data file (in pickle format) to working directory.

    The data is a dict
    key =(year, month)
    value = list of {assessment: price:}
    '''
    f = open(control.path_training, 'rb')
    transactions = pickle.load(f)
    f.close()

    print transactions.columns
    print len(transactions)
    data = {}
    for index in xrange(transactions.shape[0]):
        year = transactions.iloc[index]['sale.year']
        month = transactions.iloc[index]['sale.month']
        lv = transactions.iloc[index]['LAND.VALUE.CALCULATED']
        iv = transactions.iloc[index]['IMPROVEMENT.VALUE.CALCULATED']
        assessment = lv + iv
        price = transactions.iloc[index]['SALE.AMOUNT']
        key = (year, month)
        value = {'assessment': assessment, 'price': price}
        current_value = data.get(key, [])
        current_value.append(value)
        data[key] = current_value
        if index % 10000 == 0:
            print index, 'of', len(transactions)

    # write the errors
    print 'data keys and counts'
    for kv in sorted(data.items()):
        k, v = kv
        print k, len(v)

    f = open(control.path_data, 'wb')
    pickle.dump(data, f)
    f.close()


def create_makefile(control):
    '''Write makefile to source directory.

    It is empty.
    '''

    def make_lines():
        '''Produce a comment line only.'''
        lines = ['# makefile for chart-03 (intended to be empty)']
        return lines

    lines = make_lines()
    f = open(control.path_makefile, 'w')
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
    elif control.suffix == 'txt':
        create_txt(control)
    else:
        print_help()
        raise RuntimeError('bad command SUFFIX')

    # wrap up
    print control

    if control.testing:
        print 'DISCARD OUTPUT: TESTING'

    print 'done'

if __name__ == '__main__':
    main()
