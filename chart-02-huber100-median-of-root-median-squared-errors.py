# create files for chart-02-huber100-median-of-root-mdian-squared-errors
# with these choices
#  metric     in median-root-median-squared-errors
#  model      in q50 (quantile-50)
#  ndays      in 30 60 ... 360
#  predictors in act actlog ct ctlog
#  responses  in price logprice
#  usetax     in yes no
#  year       in 2008
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
                  model='huber100',  # huber loss with eps = 100000
                  training_periods=['30', '60', '90', '120', '150', '180',
                                    '210', '240', '270', '300', '330', '360'],
                  feature_sets=['act', 'actlog', 'ct', 'ctlog'],
                  responses=['price', 'logprice'],
                  usetax=['yes', 'no'],
                  year='2008')
    chart(specs=specs,
          argv=sys.argv)


if __name__ == '__main__':
    main()
