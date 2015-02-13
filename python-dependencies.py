# python-dependencies.py
# create file WORKING/python-dependencies.makefile
# containing dependencies for python source code

# detect only these import statement
# import <module>
# from <module> import <something>

import os
import pdb
import sys

from directory import directory
import Maybe
from Logger import Logger


class Control(object):
    def __init__(self):
        me = 'python-dependencies'
        working = directory('working')
        log = directory('log')
        src = directory('src')

        self.path_out = working + me + '.makefile'
        self.path_out_log = log + me + '.log'
        self.path_in_src = src
        self.this_file = me + '.py'

        self.testing = False
        self.debugging = False


def all_python_source_files(control):
    '''Return list of all python files in src directory.'''
    result = []
    for file_name in os.listdir(control.path_in_src):
        print file_name
        file_name_components = file_name.split('.')
        if file_name_components[-1] == 'py':
            result.append(file_name)
    return result


def extract_import_name(s):
    '''Return Maybe(module_name) for the source code string s.

    Detect only
    import NAME
    from NAME import BLAH
    '''
    pieces = s.split(' ')
    if pieces[0] == 'import':
        return Maybe.Maybe(pieces[1])
    elif pieces[0] == 'from':
        return Maybe.Maybe(pieces[1])
    else:
        return Maybe.NoValue()


def make_dependencies(python_source_file_names, control):
    '''Return dict[source_file_name] = <list of imported files>.

    Include as dependencies only files in the python source file list.
    '''
    d = {}
    for file_name in python_source_file_names:
        file_path = control.path_in_src + file_name
        dependencies = []
        with open(file_path) as f:
            lines = f.readlines()
        for line in lines:
            maybe_name = extract_import_name(line)
            if maybe_name.has_value:
                name = maybe_name.value + '.py'
                if name in python_source_file_names:
                    dependencies.append(name)
        d[file_name] = dependencies

    return d


def main():
    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    python_files = all_python_source_files(control)
    other_python_files = [f
                          for f in python_files
                          if f != control.this_file]
    pdb.set_trace()
    d = make_dependencies(python_files, control)

    # generate dependency lines
    # make this file dependent on all other files
    dependencies = []
    for source in sorted(d.iterkeys()):
        if source == control.this_file:
            print source
            line = source + ': ' + ' '.join(sorted(other_python_files))
        else:
            line = source + ': ' + ' '.join(sorted(d[source]))
        dependencies.append(line)
    pdb.set_trace()

    # write lines for dependencies; x
    f = open(control.path_out, 'w')
    for line in dependencies:
        f.write(line + '\n')
    f.close()
    pdb.set_trace()

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    print 'done'

if __name__ == '__main__':
    main()
