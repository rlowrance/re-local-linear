# create a cross validation cell, a file *.cvcell containing cv results
# invocation: MODEL-RESPONSE-PREDICTORS-YEARS-DAYS --in PATHIN --out PATHOUT
# --age YESNO
# where
# MODEL specifies the model, one of {ols, lassocv, ...}
# RESPONSE specifies the response feature, one of {price, logprice}
# PREDICTORS is an argument to function features {ct, act, ...}
# YEARS is one of {2008, 2003on}
# TRAININGDAYS is one of {30, 60, ..., 360}
# INPATH is the path to the input transactions file
#        default=WORKING/transactions-subset2.pickle
# OUTPUT is the path to the output cross validations file
#        defaul=WORKING/cvcell/MODEL-RESPONSE-PREDICTORS-YEARS-DAYS.cvcell
# YESNO whether to synthesize the age feature, default is no
#

# NOTE: The default paths are provided to maintain backwards compatibility
# with existing Makefiles.

# import built-ins and libraries
import datetime
import sys
import pdb
import cPickle as pickle
from sklearn import cross_validation
from sklearn import linear_model
import sklearn
# import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np
import pandas as pd
import warnings

# import my stuff
from Bunch import Bunch
from directory import directory
from features import features
from Logger import Logger
from CvResult import CvResult
from FoldResult import FoldResult
import parse_command_line


def make_control(argv):
    # return a Bunch

    script_name = argv[0]
    cell_specifier = argv[1]

    default_in = directory('working') + 'transactions-subset2.pickle'
    default_out = directory('cv-cell') + argv[1] + '.cvcell'
    model, response, predictors, years, days = cell_specifier.split('-')

    random_seed = 123  # random_state is an np.random.RandomState instance

    b = Bunch(debugging=False,
              testing=False,
              now=datetime.datetime.now(),
              base_name=script_name.split('.')[0],
              me=script_name,
              n_folds=10,
              random_state=sklearn.utils.check_random_state(random_seed),
              arg_age=parse_command_line.default(argv, '--age', 'yes'),
              arg_in=parse_command_line.default(argv, '--in', default_in),
              arg_out=parse_command_line.default(argv, '--out', default_out),
              model=model,
              response=response,
              predictors=predictors,
              test_years=years,
              training_days=days)
    return b


def relevant_test(df, test_years):
    ''' Return DataFrame containing just the relevant test transactions.'''
    if test_years == '2008':
        in_testing = df['sale.year'] == 2008
    elif test_years == '2004on':
        in_testing == df['sale.year'] >= 2004
    else:
        raise NotImplemented('test_years: ' + test_years)
    return df[in_testing]


def relevant_train(df, the_sale_datetime, training_days):
    '''Return rows within training_days of the sale date.'''
    days_before = the_sale_datetime - df['sale.datetime']
    in_training = days_before <= training_days
    return df[in_training]


def check_sale_datetime(dt):
    '''Check that the time components are always zero.'''
    # NOTE: This code doesn't work, as there are not such fields
    # MAYBE FIX: use pd.DateTimeIndex when creating
    pdb.set_trace()
    ok = dt.hour == 0 and \
        dt.minute == 0 and \
        dt.second == 0 and \
        dt.microsecond == 0
    if not ok:
        print 'datetime with non-zero time element', dt


def get_transaction_dates(df):
    return df['sale.python_date']


def select_testingOLD(sale_date, df):
    '''Return DataFrame with only testing transactions.
    '''
    transaction_dates = get_transaction_dates(df)
    test_indices = np.where(transaction_dates == sale_date)
    testing = df.iloc[test_indices]
    return testing


def select_trainingOLD(sale_date, df, training_days):
    '''Return DataFrame containin only training transactions.
    '''
    first_date = sale_date - datetime.timedelta(int(training_days))
    transaction_dates = get_transaction_dates(df)
    in_training = np.logical_and(transaction_dates < sale_date,
                                 transaction_dates >= first_date)
    training = df[in_training]
    return training


def response(df, response_name):
    '''Return numpy 1D array with selected column.'''
    values = np.array(df['SALE.AMOUNT'])
    if response_name == 'price':
        return values
    elif response_name == 'logprice':
        return np.log(values)
    else:
        raise NotImplemented('response: ', response_name)


def predictors(df, predictor_names):
    '''Return numpy 2D array with selected columns.'''
    # build the result matrix m in transposed form
    m = np.empty([len(predictor_names), df.shape[0]])
    column_index = 0
    for column_name in predictor_names:
        values = df[column_name]
        m[column_index] = values
        column_index += 1

    return m.transpose()


def maybe_add_age(df_list, from_to_list, predictor_names, test_date):
    'If DataFrame contains year.built or effective.year.built, add age, age^2.'

    def column_names(from_to):
        name_from, name_to = from_to
        return [name_from, name_to, name_to + '2']

    def helper(df, from_to):
        year_name, age_name, age2_name = column_names(from_to)
        if year_name in df.columns:
            year = df[year_name]
            test_year = pd.to_datetime([test_date]).year
            age = test_year - year  # age in whole number of years
            df[age_name] = age
            df[age2_name] = age * age

    # add age column to DataFrames
    pd.options.mode.chained_assignment = None  # default='warn'
    for df in df_list:
        for from_to in from_to_list:
            helper(df, from_to)

    # adjust column names in predictors
    for from_to in from_to_list:
        year_name, age_name, age2_name = column_names(from_to)
        predictor_names.remove(year_name)
        predictor_names.append(age_name)
        predictor_names.append(age2_name)


def transform_to_log(df_list, transformation):
    '''Mutate each data frame by converting to log domain.'''
    def transform_one(df):
        for feature, how in transformation.iteritems():
            if how == 'none':
                pass
            elif how == 'log':
                new_value = np.log(df[feature])
                del df[feature]  # avoid a warning in the assignment below
                df[feature] = new_value
            elif how == 'log1p':
                new_value = np.log1p(df[feature])
                del df[feature]  # avoid a warning in the assignment below
                df[feature] = new_value
            else:
                raise ValueError('feature, how: ' + feature + ',' + how)

    for df in df_list:
        transform_one(df)


def make_xy(test_date, test, train, control):
    '''Return test_x, train_x, train_y.'''
    pdb.set_trace()
    transformation = features(control.predictors)
    predictor_names = transformation.keys()

    # possibly mutate predictor_names by
    #  dropping year features
    #  adding age and age^2 features
    if control.arg_age == 'yes':
        maybe_add_age([test, train],
                      [['YEAR.BUILT', 'age'],
                       ['EFFECTIVE.YEAR.BUILT', 'effective.age']],
                      predictor_names,
                      test_date)
    if control.predictors[-3:] == 'log':
        # NOTE: don't introduce new predictor names
        transform_to_log([test, train], transformation)

    train_x = predictors(train, predictor_names)
    test_x = predictors(test, predictor_names)
    train_y = response(train, control.response)

    return test_x, train_x, train_y, predictor_names


def make_relevant_test(sale_date, df, control):
    '''Return DataFrame containing only test transactions relevant to sale_date.
    '''
    transaction_dates = get_transaction_dates(df)
    test_indices = np.where(transaction_dates == sale_date)
    relevant = df.iloc[test_indices]
    return relevant


def make_relevant_train(sale_date, df, control):
    '''Return DataFrame containing only train transactions relevant to
    sale_date.
    '''
    training_days = int(control.training_days)
    first_date = sale_date - datetime.timedelta(training_days)
    transaction_dates = get_transaction_dates(df)
    in_training = np.logical_and(transaction_dates < sale_date,
                                 transaction_dates >= first_date)
    relevant = df[in_training]
    return relevant


def make_relevant(sale_date, test, train, control):
    '''
    Return test, train for transactions relevant to sale_date and training days.
    '''
    relevant_test = make_relevant_test(sale_date, test, control)
    relevant_train = make_relevant_train(sale_date, train, control)
    return relevant_test, relevant_train


class Quantile50(object):
    def __init__(self):
        self.q = .5

    def fit(self, train_x, train_y):
        pdb.set_trace()
        m = QuantReg(endog=train_y,  # response variable
                     exog=train_x)   # explanatory variables
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                fitted = m.fit(q=self.q)
            except Warning as w:
                # for argument quantil50-logprice-act-2008-30
                # raises Convergence cycle detected
                print 'raised', w
                raise RuntimeError(w)
        fitted = m.fit(q=self.q)  # take defaults for methods
        if True:
            print 'fitted value from QuantReg'
            print fitted
            print dir(fitted)
            print fitted.summary()
            print 'model degrees of freedom', fitted.df_model
            print 'residual degress of freedom', fitted.df_resid
            print 'endog names', fitted.endog_names
            print 'exog names', fitted.exog_names
        return fitted

    def predict(fitted, test_x):
        pdb.set_trace()
        estimates = QuantReg.predict(params=fitted.params, exog=test_x)
        return estimates


def fit_model(train_x, train_y, control):
    '''Return fitted model instance.

    Pass control structure, because may need hyperparameters.
    '''
    # TODO: convert code to structure similar to quantile50
    # - build a class that does the fitting and predicting
    # - that puts together the peculiar features of each API
    # - note that scikit-learn and statsmodels have different APIs
    if control.model == 'ols':
        m = linear_model.LinearRegression(fit_intercept=True,
                                          normalize=False,
                                          copy_X=True)
        fitted = m.fit(train_x, train_y)
        if False:
            print 'fitted values'
            print 'coefficients', fitted.coef_
            print 'intercept', fitted.intercept_
        return fitted
    elif control.model == 'lassocv':
        num_samples = train_x.shape[0]
        cv = min(num_samples, 10)  # num folds in the cross validation
        verbose = True
        n_jobs = 1  # num CPUs, -1 ==> all
        fit_intercept = True
        normalize = True  # normalize regressors X before fitting
        m = sklearn.linear_model.LassoCV(cv=cv,
                                         verbose=verbose,
                                         n_jobs=n_jobs,
                                         fit_intercept=fit_intercept,
                                         normalize=normalize)
        fitted = m.fit(train_x, train_y)
        if True:
            print 'trained set shape', train_x.shape
            print 'fitted values'
            print 'coefficient', fitted.coef_
            print 'intercept', fitted.intercept_
            print fitted
        return fitted
    elif control.model == 'elasticnet':
        pdb.set_trace()
        alpha = 1.0     # multiply penalty terms by 1
        l1_ratio = 1.0  # use L1 penalty, not L2
        fit_intercept = True
        normalize = False
        normalize = True
        m = sklearn.linear_model.ElasticNet(alpha=alpha,
                                            l1_ratio=l1_ratio,
                                            fit_intercept=fit_intercept,
                                            normalize=normalize)
        fitted = m.fit(train_x, train_y)
        if True:
            print 'trained set shape', train_x.shape
            print 'fitted values'
            print 'coefficient', fitted.coef_
            print 'interecept', fitted.intercept_
            print fitted
        return fitted
    elif control.model == 'quantile50':
        return Quantile50().fit(train_x, train_y)
    elif control.model == 'ransac':
        num_samples, num_features = train_x.shape
        if num_samples < num_features + 1:
            # cannot fit a linear model on too few samples
            return None
        r = linear_model.RANSACRegressor
        m = r()  # take all the defaults
        fitted = m.fit(train_x, train_y)
        return fitted
    elif control.model == 'theilsen':
        ts = linear_model.TheilSenRegressor
        # note: not implemented in scikit learn 15.2
        # implemented in scikit learn 16.0
        verbose = True
        m = ts(fit_intercept=True,
               copy_X=True,  # otherwise, X may be overwritten
               max_subpopulation=1e4,  # use default stochastic sample size
               n_subsamples=None,      # use default
               max_iter=300,           # use default number of iterations
               tol=1e-3,               # use default
               n_jobs=1,               # one job, since make runs many jobs
               verbose=verbose)
        fitted = m.fit(train_x, train_y)
        return fitted
    elif control.model == 'huber100':
        sgd = sklearn.linear_model.SGDRegressor
        eta0 = .01
        eta0 = .02
        eta0 = .005
        eta0 = .001
        eta0 = 1e-5
        # eta0 = 1e-6
        # eta0 = .5
        # eta0 = 1.0
        eta0 = 1e-7
        learning_rate = 'invscaling'
        power_t = 0.25
        loss = 'huber'
        epsilon = 1.0
        # loss = 'squared_loss'
        m = sgd(loss='huber',
                epsilon=epsilon,  # if loss above epsilon, use linear loss
                penalty='none',  # no regularizer
                fit_intercept=True,
                n_iter=10000,  # number of epochs
                shuffle=True,   # shuffle training data after each epoch
                verbose=2,
                learning_rate=learning_rate,
                eta0=eta0,
                warm_start=False)
        fitted = m.fit(train_x, train_y)
        if True:
            print 'fitted values'
            print 'coefficients', fitted.coef_
            print 'intercept', fitted.intercept_
            print 'train_x shape', train_x.shape
            print 'loss', loss
            print 'epsilon', epsilon
            print 'learning rate', learning_rate, power_t
            print 'eta0', eta0
            print 'parameters', m.get_params()
            if loss != 'huber':
                print 'NOT HUBER LOSS'
            pdb.set_trace()
        return fitted
    else:
        print 'control.model: ', control.model
        raise NotImplemented('model: ' + control.model)


def predict_model(test_x, fitted, control):
    '''Return estimes'''
    if fitted is None:
        return None
    if control.model == 'quantile50':
        estimates = Quantile50().predict(fitted, test_x)
    else:
        # protocol for scikit-learn
        estimates = fitted.predict(test_x)
    return estimates


def get_actuals(df, control):
    '''Return actual sale prices'''
    actuals = df['SALE.AMOUNT']
    return actuals


def make_fold_result_for_sale_date(sale_date, test, train, control):
    '''Return actuals, estimates for all test transactions on sale date.
    '''
    pdb.set_trace()
    test_relevant, train_relevant = make_relevant(sale_date,
                                                  test,
                                                  train,
                                                  control)
    # convert DataFrames to x matrices and y vectors
    test_x, train_x, train_y, predictor_names = make_xy(sale_date,
                                                        test_relevant,
                                                        train_relevant,
                                                        control)
    fitted = fit_model(train_x, train_y, control)
    estimates_model_units = predict_model(test_x, fitted, control)
    if estimates_model_units is None:
        # model could not be fit
        # ex: in RANSAC, too few transactions relative to num of features
        return None, None
    if control.response == 'price':
        estimates = estimates_model_units
    elif control.response == 'logprice':
        estimates = np.exp(estimates_model_units)
    else:
        raise ValueError('control.reponse: ' + control.response)
    actuals = get_actuals(test_relevant, control)
    num_test = test_x.shape[0]
    num_train = train_x.shape[0]
    return {'actuals': actuals,
            'estimates': estimates,
            'model': control.model,
            'fitted': fitted,
            'predictor_names': predictor_names,
            'num_test': num_test,
            'num_train': num_train}


def make_sorted_test_sale_dates(df, control):
    '''Return list of sorted sale dates in test years.'''
    if control.test_years == '2008':
        first_date = datetime.date(2008, 1, 1)
        last_date = datetime.date(2008, 12, 31)
    elif control.test_years == '2003on':
        first_date = datetime.date(2003, 1, 1)
        last_date = datetime.date(2009, 3, 31)
    else:
        raise NotImplementedError('control.test_years: ' + control.test_years)
    dates = df['sale.python_date']
    in_test_period = np.logical_and(dates >= first_date,
                                    dates <= last_date)
    test_year_dates = dates[in_test_period]
    unique = test_year_dates.unique()
    ordered = np.sort(unique)
    return ordered


def make_fold_result(fold_number, test, train, control):
    '''Return FoldResult instance from predicting all test transactions.

    For each test transaction,
    1. Build model with all the training data.
    2. Predict value for the test transaction.
    '''
    sorted_sale_dates = make_sorted_test_sale_dates(test, control)
    fold_result = FoldResult()
    for test_sale_date in sorted_sale_dates:
        with warnings.catch_warnings():
            try:
                fr = make_fold_result_for_sale_date(test_sale_date,
                                                    test,
                                                    train,
                                                    control)
            except Warning as w:
                print 'warning raised:', w
                print 'fold_number', fold_number
                print 'test_sale_date', test_sale_date
                print 'skipping this test date'
                continue

        actuals = fr['actuals']
        estimates = fr['estimates']
        if actuals is None or estimates is None:
            print control.command_line, fold_number, test_sale_date, None
        else:
            fold_result.extend(actuals, estimates)
            fold_result.save_raw_fold_result(test_sale_date, fr)
            print \
                control.command_line, \
                fold_number, \
                test_sale_date, \
                len(actuals)
    return fold_result


def make_cv_result(df, control):
    '''
    Return CvResult instance containing results from 10-fold cv.

    Break DataFrame df into folds.
    Determine cross-validation error for each fold.
    '''
    # create iterator across fold indices
    kf = cross_validation.KFold(df.shape[0],
                                n_folds=control.n_folds,
                                shuffle=True,
                                random_state=control.random_state)

    # for each fold
    pdb.set_trace()
    cvresult = CvResult()
    fold_number = 0
    for train_indices, test_indices in kf:
        fold_number += 1

        # split df into train and test
        fold_train = df.iloc[train_indices]
        fold_test = df.iloc[test_indices]

        fold_result = make_fold_result(fold_number,
                                       fold_test,
                                       fold_train,
                                       control)
        cvresult.save_FoldResult(fold_result)

    return cvresult


def main():

    warnings.filterwarnings('error')
    control = make_control(sys.argv)
    path = \
        directory('log') + \
        control.base_name + '.' + control.now.isoformat('T') + '.log'
    sys.stdout = Logger(logfile_path=path)  # print x now logs and prints x
    print control

    # read training data
    print "reading training data"
    f = open(control.arg_in, 'rb')
    df = pickle.load(f)
    f.close()

    # check that no sale.python_date value is not null
    if False and control.debugging_sale_python_date:
        dates = df['sale.python_date']
        if dates.isnull().any():
            print dates
            raise ValueError('a sale.python_date is null')

    print "making the cross validation result"
    pdb.set_trace()
    cv_result = make_cv_result(df=df, control=control)

    # write cross validation result
    f = open(control.arg_out, 'wb')
    pickle.dump(cv_result, f)
    f.close()

    print control
    print 'done'

if __name__ == '__main__':
    main()
