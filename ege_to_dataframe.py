'''convert ege results dict to pandas dataframe

COMMAND LINE: python ege_to_dataframe.py YYYY-MM-DD

API
from age_to_dataframe import ege_to_dataframe
df = ege_to_dataframe(ege_result)
'''

# TODO: read many input files, one for each date

import cPickle as pickle
import datetime
import numpy as np
import pandas as pd
from pprint import pprint  # for debugging
import random
import pdb
import sys

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger
# import parse_command_line
from ege_date import Rf, Ols, ReportRf, ReportOls


def print_help():
    print 'python ege_to_dataframe.py --date YYYY-MM-DD'
    print 'where'
    print ' YYYY-MM-DD is the date'
    print 'input file:  WORKING/ege_date-MMMM-DD-YY.pickle'
    print 'output file: WORKING/ege_date-MMMM-DD-YY.dataframe.pickle'


def make_control(argv):
    'return a Bunch'

    random.seed(123456)

    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    testing = False

    arg_date = argv[1]

    # supply common conrol values
    b = Bunch(debugging=False,
              base_name=base_name,
              me=argv[0],
              path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
              path_in=directory('working') + 'ege_date-' + arg_date + '.pickle',
              path_out=directory('working') + 'ege_to_dataframe-' + arg_date + '.pickle',
              testing=testing)

    return b


def ege_to_dataframe(ege_results):
    '''return DataFrame with summary of the dict all_results

    The reduced data is a dict containing
    - for each fold and sale date:
        - 'num_tests': number of tests
        - 'fitted_coef': linear regression intercept
        - 'actuals': actuals, estimates for each test transaction in the fold
        - 'fitted_models_subset': a subset of the the fitted model
    - for each fold, sale date, and predictor:
        - 'fitted_coef': linear regression coefficient

    Note: there is one model for each sale date and fold
    '''

    def errors(model_result):
        'return median_absolute_error and median_relative_absolute_error'
        actuals = model_result['actuals']
        estimates = model_result['estimates']
        abs_error = np.abs(actuals - estimates)
        median_abs_error = np.median(abs_error)
        rel_abs_error = abs_error / actuals
        median_rel_abs_error = np.median(rel_abs_error)
        return median_abs_error, median_rel_abs_error

    def make_df(fold_number=None, sale_date=None, training_days=None,
                scope=None, model_id=None,
                abs_error=None, rel_error=None):
        'convert scalars (possible None) to DataFrame'

        def maybe(x):
            return [] if x is None else [x]

        d = {'fold_number': pd.Series(maybe(fold_number), dtype=np.float64),
             'sale_date': pd.Series(maybe(sale_date), dtype=object),
             'training_days': pd.Series(maybe(training_days), dtype=np.float64),
             'scope': pd.Series(maybe(scope), dtype=object),
             'model_id': pd.Series(maybe(model_id), dtype=object),
             'abs_error': pd.Series(maybe(abs_error), dtype=np.float64),
             'rel_error': pd.Series(maybe(rel_error), dtype=np.float64)}
        df = pd.DataFrame(d)
        return df

    def append_df(df, fold_number, sale_date, training_days, scope, model_name, result):
        'use fold_number .. errors(result) to append row to df'
        abs_error, rel_error = errors(result)
        new_df = make_df(fold_number, sale_date, training_days, scope, model_name, abs_error, rel_error)
        return df.append(new_df, ignore_index=True)

    def is_zip_code(scope):
        return isinstance(scope, tuple) and scope[0] == 'zip'

    def append_modes(base_name, mode):
        mode_y = mode[1]
        mode_x = mode[3]
        return base_name + ' ' + mode_y[:3] + ' ' + mode_x[:3]

    def rf_model_id(rf_mode):
        # rf_mode is number of trees in the forest
        return 'rf %3d' % rf_mode

    df = make_df()
    for k, v in ege_results.iteritems():
        print k
        k_fold_number, k_sale_date, k_training_days, k_model_name, k_scope = k
        if k_model_name == 'ols':
            # handle the y and x modes by creating an augumented model name
            if k_scope == 'global':
                for mode, result in v.iteritems():
                    model_id = append_modes(k_model_name, mode)
                    df = append_df(df,
                                   k_fold_number, k_sale_date, k_training_days, k_scope, model_id,
                                   result)
            elif is_zip_code(k_scope):
                for mode, result in v.iteritems():
                    model_id = append_modes(k_model_name, mode)
                    zip_code = k_scope[1]
                    df = append_df(df,
                                   k_fold_number, k_sale_date, k_training_days, zip_code, model_id,
                                   result)
            else:
                RuntimeError('bad k_scope: ' + k_scope)
        elif k_model_name == 'rf':
            pdb.set_trace()  # handle the rf variants; follow the ols code
            if k_scope == 'global':
                for mode, result in v.iteritems():
                    model_id = rf_model_id(mode)
                    df = append_df(df,
                                   k_fold_number, k_sale_date, k_training_days, k_scope, model_id,
                                   result)
            elif is_zip_code(k_scope):
                for mode, result in v.iteritems():
                    model_id = rf_model_id(mode)
                    zip_code = k_scope[1]
                    df = append_df(df,
                                   k_fold_number, k_sale_date, k_training_days, zip_code, model_id,
                                   result)
            else:
                RuntimeError('bad k_scope: ' + k_scope)
        else:
            RuntimeError('bad k_model_name: ' + k_model_name)
    print 'df.shape', df.shape
    return df


def main(argv):
    control = make_control(sys.argv)
    sys.stdout = Logger(control.path_log)
    print control

    f = open(control.path_in, 'rb')
    pickled = pickle.load(f)
    f.close()

    df = ege_to_dataframe(pickled['all_results'])

    f = open(control.path_out, 'wb')
    pickle.dump(df, f)
    f.close()

    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        Ols()
        ReportOls()
        Rf()
        ReportRf()
    main(sys.argv)
