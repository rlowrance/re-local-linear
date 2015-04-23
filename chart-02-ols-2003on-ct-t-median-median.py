# create files for chart-02-*
# with these choices
#  metric     in median-root-median-squared-errors
#  model      in ols
#  ndays      in 30 60 ... 360
#  predictors in ct ctlog t tlog
#  responses  in price logprice
#  year       in 2003on
# invocations and files created
#  python chart-02X.py makefile -> src/chart-02X.makefile
#  python chart-02X.py data     -> data/working/chart-02X.data
#  python chart-02X.py txt      -> data/working/chart-02X.txt
#  python chart-02X.py txtY     -> data/working/chart-02X-Y.txt

import sys


from Bunch import Bunch
from chart_02_template import chart


def main():
    specs = Bunch(metric='median-of-root-median-squared-errors',
                  title='Median of Root Median Squared Errors',
                  model='ols',
                  training_periods=['30', '60', '90', '120', '150', '180',
                                    '210', '240', '270', '300', '330', '360'],
                  feature_sets=['ct', 'ctlog', 't', 'tlog'],
                  responses=['price', 'logprice'],
                  year='2003on')
    chart(specs=specs,
          argv=sys.argv)


if __name__ == '__main__':
    main()
