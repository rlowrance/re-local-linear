def year_month_day(yyyymmdd):
    '''
    Return numpy array with yyyy values (the years).
    '''
    import numpy as np

    year = np.floor(yyyymmdd / 1e4)
    month = np.floor((yyyymmdd - year * 10000) / 1e2)
    day = yyyymmdd - year * 10000 - month * 100
    return year, month, day


if __name__ == '__main__':
    import unittest
    import numpy as np
    from np_equal_considering_nans import np_equal_considering_nans

    class TestSaleYear(unittest.TestCase):
        def test_synthetic(self):
            dates = np.array([12345678, 12339988])
            years, months, days = year_month_day(dates)
            # print years, months, days
            self.assertEqual(years[0], 1234)
            self.assertEqual(years[1], 1233)
            self.assertEqual(months[0], 56)
            self.assertEqual(months[1], 99)
            self.assertEqual(days[0], 78)
            self.assertEqual(days[1], 88)

        def test_actual(self):
            dates = np.float64([np.nan,
                                20001220,
                                20051223,
                                np.nan,
                                20020926,
                                20030519,
                                20060720,
                                19891200,
                                np.nan,
                                20050111])
            years, months, days = year_month_day(dates)
            expected_years = np.float64([np.nan,
                                         2000,
                                         2005,
                                         np.nan,
                                         2002,
                                         2003,
                                         2006,
                                         1989,
                                         np.nan,
                                         2005])
            expected_months = np.float64([np.nan,
                                          12,
                                          12,
                                          np.nan,
                                          9,
                                          5,
                                          7,
                                          12,
                                          np.nan,
                                          1])
            expected_days = np.float64([np.nan,
                                        20,
                                        23,
                                        np.nan,
                                        26,
                                        19,
                                        20,
                                        0,
                                        np.nan,
                                        11])

            def all_ok(a, b):
                result = True
                for index, a_item in np.ndenumerate(a):
                    b_item = b[index]
                    if np.isnan(a_item) | np.isnan(b_item):
                        pass
                    else:
                        result &= a_item == b_item
                        if not result:
                            return False
                return True

            self.assertTrue(all_ok(years, expected_years))
            self.assertTrue(np_equal_considering_nans(years, expected_years))

            self.assertTrue(all_ok(months, expected_months))
            self.assertTrue(np_equal_considering_nans(months, expected_months))

            self.assertTrue(all_ok(days, expected_days))
            self.assertTrue(np_equal_considering_nans(days, expected_days))

    unittest.main()
