# create file WORKING/record-counts.tex containing record counts

# import built-ins and libraries
import pandas as pd
import sys

# import my stuff
from directory import directory
from Logger import Logger
from round_to_n import round_to_n


class Control(object):
    def __init__(self):
        me = 'record-counts'
        working = directory('working')
        log = directory('log')

        self.path_out = working + me + '.tex'
        self.path_out_log = log + me + '.log'
        self.path_in_parcels = working + 'parcels-sfr-counts.csv'
        self.path_in_deeds = working + 'deeds-al-g-counts.csv'
        self.path_in_transactions = working + 'transactions-counts.csv'

        self.testing = False


def main():

    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    # build the counts dictionary from the content of all of the input files

    counts = {}

    def process_input_file(tag, path):
        def save_row(kind, row):
            print 'save_row', kind, row
            counts[(kind, row.file_name)] = row.record_count

        df = pd.io.parsers.read_csv(path)
        df.apply(lambda row: save_row(tag, row),
                 axis=1)

    process_input_file('parcels', control.path_in_parcels)
    process_input_file('deeds', control.path_in_deeds)
    process_input_file('transactions', control.path_in_transactions)
    import pdb
    pdb.set_trace()

    # create the commands

    def declare(command_name, command_text):
        return r'\DeclareRobustCommand{\%s}{%s}' % (command_name, command_text)

    def count_formatted(k, v, formatted_name, formatter):
        return declare('%s%s%s' % (k[0].title(),
                                   k[1].title(),
                                   formatted_name),
                       formatter(v))

    def exact_count(k, v):
        def exact_count_formatter(n):
            return format(n, ',')

        return count_formatted(k, v, 'ExactCount', exact_count_formatter)

    def rounded_count(k, v):

        def rounded_count_formatter(n):
            rounded = int(round_to_n(n, 2))
            return format(rounded, ',')

        return count_formatted(k, v, 'RoundedCount', rounded_count_formatter)

    commands = []
    for k, v in counts.iteritems():
        command1 = exact_count(k, v)
        commands.append(command1)
        command2 = rounded_count(k, v)
        commands.append(command2)

    # write the lines
    f = open(control.path_out, 'w')
    print
    print 'commands:'
    for command in commands:
        print command
        f.write('%s\n' % command)
    f.close()
    print

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    print 'done'

if __name__ == '__main__':
    main()
