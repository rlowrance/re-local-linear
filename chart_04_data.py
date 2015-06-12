# create WORKING/chart-04.data

# import built-ins and libraries
import collections
import cPickle as pickle
import pdb

# import my stuff
from directory import directory


def create(control):
    '''Write data file (in pickle format) to working directory.

    The data is a dict
    key = ERROR (from command line) (one of mrmse mae)
    value = a dicionary with the estimated generalization error
     key =(response, predictor, training_days)
     value = scalar value from each fold
    '''

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

    # key = (fold_number, sale_date) value = num test transactions
    num_tests = collections.defaultdict(int)
    # key = (fold_number, sale_date, predictor_name) value = coefficient
    test_coef = {}

    path = directory('cells') + control.cvcell_id + '.cvcell'
    cv_result = get_cv_result(path)  # a CvResult instance
    num_folds = len(cv_result.fold_results)
    for fold_number in xrange(num_folds):
        fold_result = cv_result.fold_results[fold_number]
        # fold_actuals = fold_result.actuals
        # fold_estimates = fold_result.estimates
        fold_raw = fold_result.raw_fold_result
        for sale_date, fitted_model in fold_raw.iteritems():
            # sale_date_num_train = fitted_model['num_train']
            # sale_date_estimates = fitted_model['estimates']
            # sale_date_actuals = fitted_model['estimates']
            sale_date_predictor_names = fitted_model['predictor_names']
            sale_date_num_test = fitted_model['num_test']
            # sale_date_model = fitted_model['model']
            sale_date_fitted = fitted_model['fitted']

            # extract info from the fitted_model
            num_tests[(fold_number, sale_date)] += sale_date_num_test

            # save each coefficient
            sale_date_coef = sale_date_fitted.coef_
            assert(len(sale_date_coef) == len(sale_date_predictor_names))
            for i in xrange(len(sale_date_coef)):
                key = (fold_number, sale_date, sale_date_predictor_names[i])
                test_coef[key] = sale_date_coef[i]

    # write the data
    data = {'num_tests': num_tests, 'test_coef': test_coef}
    path = directory('working') + control.base_name + '.data'
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    if False:
        pdb.set_trace()
