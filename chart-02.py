# create files (selective on how invoked)
# WORKING/chart-02.makefile
# WORKING/chart-02.data
# WORKING/chart-02.txt-mean-root-mean-squared-errors.txt
# WORKING/chart-02.txt-median-root-median-squared-errors.txt

import sys
import chart_02_04


def main():
    args = sys.argv
    chart_02_04.chart_02_04(chart_id='02',
                            suffix=args[1],
                            specific=args[2] if len(args) == 3 else None)

if __name__ == '__main__':
    main()
