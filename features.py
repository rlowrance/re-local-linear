def features(feature_set_name):
    '''Return dictionary[column_name] = transformation for the feature set.


    feature_set_name in [act, actLog, ct, ctLog, t, tLog,
                         bestNN, pcaNN,
                         id, prices
                         best15{census|city|zip}]

    See Features.R for original implementation.
    '''

    # these dictionaries define the features in each feature set and
    # whether and how to convert the feature into the log domain

    id_features = {'RECORDING.DATE': 'none',
                   'SALE.DATE': 'none',
                   'sale.year': 'none',
                   'sale.month': 'none',
                   'sale.day': 'none',
                   'apn.recoded': 'none',
                   'CENSUS.TRACT': 'none',
                   'zip5': 'none',
                   'PROPERTY.CITY': 'none'}

    price_features = {'SALE.AMOUNT': 'none'}

    predictors_assessment = {'IMPROVEMENT.VALUE.CALCULATED': 'log',
                             'LAND.VALUE.CALCULATED': 'log',
                             'fraction.improvement.value': 'none'}

    predictors_census = {'avg.commute': 'none',
                         'census.tract.has.industry': 'none',
                         'census.tract.has.park': 'none',
                         'census.tract.has.retail': 'none',
                         'census.tract.has.school': 'none',
                         'fraction.owner.occupied': 'none',
                         'median.household.income': 'none'}

    predictors_taxroll = {'EFFECTIVE.YEAR.BUILT': 'none',
                          'has.pool': 'none',
                          'is.new.construction': 'none',
                          'YEAR.BUILT': 'none',
                          'zip5.has.industry': 'none',
                          'zip5.has.park': 'none',
                          'zip5.has.retail': 'none',
                          'zip5.has.school': 'none',
                          'LAND.SQUARE.FOOTAGE': 'log',
                          'LIVING.SQUARE.FEET': 'log',
                          'BASEMENT.SQUARE.FEET': 'log1p',
                          'TOTAL.BATHS.CALCULATED': 'log1p',
                          'BEDROOMS': 'log1p',
                          'FIREPLACE.NUMBER': 'log1p',
                          'PARKING.SPACES': 'log1p',
                          'STORIES.NUMBER': 'log1p',
                          'TOTAL.ROOMS': 'log1p'}

    def transform(predictors, use_log):
        '''Return dictionary specifying how to convert to log domains.'''
        result = {}
        for k, v in predictors.iteritems():
            result[k] = v if use_log else 'none'
        return result

    def pca(n):
        '''Return feature set of first n pca features.'''
        pca_features_all = ('median.household.income',
                            'land.square.footage',
                            'living.area',
                            'basement.square.feet')
        pca_features_selected = pca_features_all[:n]
        result = {}
        for feature in pca_features_selected:
            result[feature] = 'none'  # no transformations
        return result

    def best(n):
        '''Return feature set of first n best features.

        Mimic pca(n), eventually.
        The best features will end up in a file and will need to be read in.
        For now, raise an error
        '''
        return NotImplementedError

    pca_feature_set_names = ('pca01', 'pca02', 'pca03', 'pca04')
    best_feature_set_names = ('best01', 'best02', 'best03', 'best04', 'best05',
                              'best06', 'best07', 'best08', 'best09', 'best10',
                              'best11', 'best12', 'best13', 'best14', 'best15',
                              'best16', 'best17', 'best18', 'best19', 'best20')
    if pca_feature_set_names.count(feature_set_name) == 1:
        return pca(pca_feature_set_names.index(feature_set_name) + 1)

    elif best_feature_set_names.count(feature_set_name) == 1:
        return best(best_feature_set_names.index(feature_set_name) + 1)

    elif feature_set_name == 'act':
        d = predictors_assessment  # NOTE: mutates predictors_assessment
        d.update(predictors_census)
        d.update(predictors_taxroll)
        return transform(d, use_log=False)

    elif feature_set_name == 'actLog':
        d = predictors_assessment
        d.update(predictors_census)
        d.update(predictors_taxroll)
        return transform(d, use_log=True)

    elif feature_set_name == 'ct':
        d = predictors_census
        d.update(predictors_taxroll)
        return transform(d, use_log=False)

    elif feature_set_name == 'ctLog':
        d = predictors_census
        d.update(predictors_taxroll)
        return transform(d, use_log=True)

    elif feature_set_name == 't':
        d = predictors_taxroll
        return transform(d, use_log=False)

    elif feature_set_name == 'tLog':
        d = predictors_taxroll
        return transform(d, use_log=True)

    elif feature_set_name == 'id':
        return id_features

    elif feature_set_name == 'prices':
        return price_features

    elif feature_set_name == 'best15census':
        return NotImplementedError

    elif feature_set_name == 'best15city':
        return NotImplementedError

    elif feature_set_name == 'best15zip':
        return NotImplementedError

    else:
        raise RuntimeError('invalid feature_set_name ' + feature_set_name)

if __name__ == '__main__':
    # unit test
    # for now, just print
    import unittest

    def get(feature_set_name):
        ''' get features and possibly print them.'''
        f = features(feature_set_name)
        if False:
            print(feature_set_name)
            print(f)
        return f

    class TestFeatures(unittest.TestCase):
        def setUp(self):
            pass

        def test_id(self):
            f = get('id')
            self.assertEqual(len(f), 9)

        def test_prices(self):
            f = get('prices')
            self.assertEqual(len(f), 1)

        def test_pca02(self):
            f = get('pca02')
            self.assertEqual(len(f), 2)

        def test_best20(self):
            self.failIf(False)  # not yet implemented
            # f = get('best20')
            # self.assertEqual(len(f), 21)

        def test_act(self):
            f = get('act')
            self.assertEqual(len(f), 27)

        def test_actLog(self):
            f = get('actLog')
            self.assertEqual(len(f), 27)

        def test_ct(self):
            f = get('ct')
            self.assertEqual(len(f), 24)

        def test_ctLog(self):
            f = get('ctLog')
            self.assertEqual(len(f), 24)

        def test_t(self):
            f = get('t')
            self.assertEqual(len(f), 17)

        def test_tLog(self):
            f = get('tLog')
            self.assertEqual(len(f), 17)

    unittest.main()
