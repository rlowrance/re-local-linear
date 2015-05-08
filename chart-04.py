# create files for chart-04: elastic net variables in importance order
# WORKING/chart-NN.makefile
# WORKING/chart-NN.data
# WORKING/chart-NN.txt-SPECIFIC.txt

# import built-ins and libraries
import sys
import pdb
import cPickle as pickle
import os.path

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger


def print_help():
    print 'python chart-04.py SUFFIX'
    print 'where SUFFIX    in {"makefile", "data", "txt"}'


def make_control(argv):
    # return a Bunch

    # derive fields from the invocation
    base_name = argv[0].split('.')[0]
    suffix = argv[1]

    # identifier for the cross validation cell we use
    cvcell_id = 'elasticnet-logprice-ct-2003on-30'

    def make_paths():
        # return Bunch of paths
        in_file = 'transactions-subset2-train.pickle'
        b = Bunch(
            cell=directory('cells') + cvcell_id + '.cvcell',
            cvcell_program='cv-cell.py',
            data=directory('working') + base_name + '.data',
            txt=directory('working') + base_name + '.txt',
            in_training=directory('working') + in_file,
            out_log=directory('log') + base_name + suffix + '.log',
            out_makefile=base_name + '.makefile')
        return b

    if not(2 <= len(argv) == 2):
        print_help()
        print 'argv', argv
        raise RuntimeError('bad invocation')

    # supply common conrol values
    b = Bunch(debugging=False,
              base_name=base_name,
              cvcell_id=cvcell_id,
              me=argv[0],
              path=make_paths(),
              suffix=suffix,
              testing=False)

    return b


class Report(object):

    def __init__(self, lines, table_entry_format):
        self.lines = lines
        self.format_header = '{:>9s}' + (' {:>8s}' * 8)
        self.format_detail = '{:9d}' + ((' {:%s}' % table_entry_format) * 8)
        self.format_legend = '{:80s}'

    def header(self, c0, c1, c2, c3, c4, c5, c6, c7, c8):
        print c0, c1, c2, c3, c4, c5, c6, c7, c8
        s = self.format_header.format(c0, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)

    def detail(self, ndays, *clist):
        # replace large column values with all 9's
        print ndays, clist
        large_value = 99999999
        capped = [x if x <= large_value else large_value
                  for x in clist]
        s = self.format_detail.format(ndays,
                                      capped[0],
                                      capped[1],
                                      capped[2],
                                      capped[3],
                                      capped[4],
                                      capped[5],
                                      capped[6],
                                      capped[7])
        # s = self.format_detail.format(ndays, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)

    def legend(self, txt):
        print 'legend', txt
        s = self.format_legend.format(txt)
        self.lines.append(s)


def create_txt(control):
    '''Return list of lines that are chart 02.txt.
    '''
    def append_description(lines):
        '''Append header lines'''
        lines.append(control.specs.title)
        lines.append('From 10-fold Cross Validation')
        lines.append(' ')
        lines.append('Model: ' + control.specs.model)
        lines.append('Time period: ' + control.specs.year)
        lines.append(' ')

    def read_data():
        '''Return correct data dict built by create_data() function.'''
        path = control.path.dir_working + control.base_name + '.data'
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def append_header(t):
        t.header('response:',
                 'price', 'price', 'price', 'price',
                 'logprice', 'logprice', 'logprice', 'logprice')
        t.header('features:',
                 control.specs.feature_sets[0],
                 control.specs.feature_sets[1],
                 control.specs.feature_sets[2],
                 control.specs.feature_sets[3],
                 control.specs.feature_sets[0],
                 control.specs.feature_sets[1],
                 control.specs.feature_sets[2],
                 control.specs.feature_sets[3])
        t.header('ndays', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ')

    def append_detail_line(report, data, ndays):
        def v(response, features):
            '''Return int or 0, the value in the report.
            '''
            # features := predictor
            # shortened, so save one column in the printout
            key = (response, features, ndays)
            if key in data:
                value = data[key]
                if control.specs.metric == 'mean-wi10':
                    return value
                elif control.specs.metric == 'mean-mean':
                    return int(value)
                elif control.specs.metric == 'median-median':
                    return int(value)
                else:
                    raise RuntimeError('unknown metric: ' +
                                       control.specs.metric)
            else:
                print 'no data for', key
                return 0

        report.detail(int(ndays),
                      v('price', control.specs.feature_sets[0]),
                      v('price', control.specs.feature_sets[1]),
                      v('price', control.specs.feature_sets[2]),
                      v('price', control.specs.feature_sets[3]),
                      v('logprice', control.specs.feature_sets[0]),
                      v('logprice', control.specs.feature_sets[1]),
                      v('logprice', control.specs.feature_sets[2]),
                      v('logprice', control.specs.feature_sets[3]))

    def append_detail_lines(report, values):
        '''Append body lines to report using values.'''

        # one line for each training period
        for ndays in control.specs.training_periods:
            append_detail_line(report, values, ndays)

    feature_set_desc = \
        dict(act='features derived from accessor, census, and taxroll data',
             actlog='like at, but size features in log domain',
             ct='features derived from census and taxroll data',
             ctlog='like ct, but size features in log domain',
             t='features derived from taxroll data',
             tlog='like t, but size features in log domain'
             )

    def append_legend_lines(report):
        # print legend describing features sets actually used
        def r(s):
            report.legend(s)

        r(' ')
        r('features set definitions')
        for feature_set in control.specs.feature_sets:
            r(feature_set + ': ' + feature_set_desc[feature_set])
        r(' ')

    def write_lines(lines):
        f = open(control.path.out_output, 'w')
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    lines = []
    append_description(lines)
    report = Report(lines, control.table_entry_format)
    append_header(report)
    data = read_data()
    append_detail_lines(report, data)
    append_legend_lines(report)
    write_lines(lines)


def create_data(control):
    '''Write data file (in pickle format) to working directory.

    The data is a dict
    key = ERROR (from command line) (one of mrmse mae)
    value = a dicionary with the estimated generalization error
     key =(response, predictor, training_days)
     value = scalar value from each fold
    '''

    def make_file_path(response, predictor, training_days, control):
        '''Return string containing file name.
        '''
        cell_file_name = '%s-%s-%s-%s-%s.cvcell' % (control.specs.model,
                                                    response,
                                                    predictor,
                                                    control.specs.year,
                                                    training_days)
        return control.path.dir_cells + cell_file_name

    def get_cv_result(file_path):
        '''Return CvResult instance.'''
        f = open(file_path, 'rb')
        cv_result = pickle.load(f)
        f.close()
        return cv_result

    # create table containing results from each cross validation
    def cv_result_summary(cv_result):
        if control.specs.metric == 'median-median':
            maybe_value = cv_result.median_of_root_median_squared_errors()
        elif control.specs.metric == 'mean-wi10':
            maybe_value = cv_result.mean_of_fraction_wi10()
        elif control.specs.metric == 'mean-mean':
            maybe_value = cv_result.mean_of_root_mean_squared_errors()
        else:
            print control.specs
            raise RuntimeError('unknown metric: ' + control.specs.metric)
        return maybe_value.value if maybe_value.has_value else None

    data = {}
    for response in control.specs.responses:
        for feature_set in control.specs.feature_sets:
            for training_period in control.specs.training_periods:
                file_path = make_file_path(response,
                                           feature_set,
                                           training_period,
                                           control)
                key = (response, feature_set, training_period)
                if os.path.isfile(file_path):
                    cv_result = get_cv_result(file_path)
                    data[key] = cv_result_summary(cv_result)
                else:
                    print 'no file for', response, feature_set, training_period
                    raise RuntimeError('missing file: ' + file_path)

    # write the data (so that its in the log)
    for k, v in data.iteritems():
        print k, v

    path = control.path.out_output
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def create_makefile(control):
    '''Write makefile to source directory.'''
    def make_file_names():
        '''Return list of cell names.'''
        file_names = []
        pdb.set_trace()
        for response in control.specs.responses:
            for feature_set in control.specs.feature_sets:
                for training_period in control.specs.training_periods:
                    cell_name = '%s-%s-%s-%s-%s' % (control.specs.model,
                                                    response,
                                                    feature_set,
                                                    control.specs.year,
                                                    training_period)
                    file_name = '%s%s.cvcell' % (control.path.dir_cells,
                                                 cell_name)
                    file_names.append(file_name)
                    if control.testing and len(file_names) > 0:
                        return file_names
        return file_names

    def make_lines():
        '''Produce lines for makefile.

        chart-02-cells = <cell1> <cell2> ...
        chart-02.txt: chart-02.data chart-02.py
            $(PYTHON) chart-02.py txt
        chart-02.data: $(chart-02-cells) chart-02.py
            $(PYTHON) chart-02.py data
        #chart-02.makefile: chart02.py
        #   $(PYTHON) chart-02.py makefile
        '''
        lines = []
        lines.append('# makefile generated by python %s.py makefile' %
                     control.base_name)

        # rule to build the cross-validation cell
        start_python = '\t$(PYTHON)'
        lines.append('%s: %s %s' %
                     (control.path.cell,
                      control.path.cvcell_program,
                      control.path.in_training))
        lines.append('%s %s %s' %
                     (start_python,
                      control.path.cvcell_program,
                      control.cvcell_id))

        # rule to create the data for the chart
        lines.append('%s: %s %s' %
                     (control.path.data,
                      control.path.cell,
                      control.me))
        lines.append('%s %s data' %
                     (start_python,
                      control.me))

        # rule to create the chart
        lines.append('%s: %s %s' %
                     (control.path.txt,
                      control.path.data,
                      control.me))
        lines.append('%s %s txt' %
                     (start_python,
                      control.me))

        return lines

    lines = make_lines()
    if True:
        print 'makefile'
        for line in lines:
            print line

    f = open(control.path.out_makefile, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()


def chart(specs, argv):
    '''create files for charts 02-X
    ARGS
    specs: a Bunch of specifications
    argv: the value of sys.argv from the caller (a main program)
    '''
    if False:
        pdb.set_trace()
    control = make_control(specs, argv)
    sys.stdout = Logger(logfile_path=control.path.out_log)
    print control

    suffix = argv[1]
    if suffix == 'makefile':
        create_makefile(control)
    elif suffix == 'data':
        create_data(control)
    elif suffix == 'txt':
        create_txt(control)
    else:
        print_help()
        raise RuntimeError('bad command SUFFIX')

    # clean up
    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'


def main():
    control = make_control(sys.argv)
    sys.stdout = Logger(logfile_path=control.path.out_log)
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

    # clean up
    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'


if __name__ == '__main__':
    main()
