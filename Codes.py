"provide contents of CoreLogic's code files"


import pandas as pd
import pdb
from pprint import pprint
import unittest


class Codes(object):
    def __init__(self,
                 path_parcels='../data/input/corelogic-code-tables/2580_Code_Tables.csv',
                 path_deeds='../data/input/corelogic-code-tables/1080_Code_Tables.csv',
                 verbose=False):
        self._verbose = verbose
        self._parcels_to_description = {}
        self._parcels_to_code = {}
        self._deeds_to_description = {}
        self._deeds_to_code = {}
        # read files and build data structures needed for the queries
        self._read_deeds(path_deeds)
        self._read_parcels(path_parcels)

    def _read_table_file(self, path, recoded_descriptions, ignore):
        '''return to_code, to_description dicts

        keys = (table, code) or (table, description)
        '''

        def safely_insert(table, k, v, ignore, dest):
            key = (table, k)
            if ignore(key, v):
                return
            if key in dest:
                print table, k, v
                pdb.set_trace()
                print 'duplicate key'
            else:
                dest[key] = v

        df = pd.read_csv(path)
        to_code = {}
        to_description = {}
        seen = set()
        for id_row in df.iterrows():
            row = id_row[1]
            table = row[0]
            code = row[1]
            original_description = row[2]
            if (table, code, original_description) in seen:
                continue  # some rows are duplicated in the code tables
            description = (original_description + '-' + code
                           if (table, original_description) in recoded_descriptions
                           else original_description)
            if original_description != description:
                if self._verbose:
                    print 'recoded', table, code, original_description, description
            safely_insert(table, code, description, ignore, to_description)
            safely_insert(table, description, code, ignore, to_code)
            seen.add((table, code, original_description))
        return to_code, to_description

    def _read_deeds(self, path_deeds):

        def ignore(key, v):
            'ignore some (table, code) pairs; return boolean'
            return key == ('DEEDC', 'L') and v == 'LIS PENDENS - NON CALIFORNIA'

        recoded_descriptions = set((('OWNSH', 'SINGLE'),
                                    ('SCODE', 'SALE PRICE (PARTIAL)'),
                                    ('LOCIN', 'TYPE UNKNOWN'),
                                    ))
        to_code, to_description = self._read_table_file(path_deeds, recoded_descriptions, ignore)
        self._deeds_to_code = to_code
        self._deed_to_description = to_description

    def _read_parcels(self, path_parcels):

        def ignore(key, v):
            return False

        recoded_descriptions = set((('FIREP', 'TYPE UNKNOWN'),
                                    ('RFFRM', 'BAR JOIST'),
                                    ('RFFRM', 'CONCRETE'),
                                    ('RFFRM', 'BAR JOIST & WOOD DECK'),
                                    ('RFFRM', 'METAL PIPE'),
                                    ('RFFRM', 'WOOD FRAME'),
                                    ('RFFRM', 'WOOD FRAME'),
                                    ('RFFRM', 'BOWSTRING'),
                                    ('RFFRM', 'BAR JOIST & CONCRETE DECK'),
                                    ('RFFRM', 'FLEXIBLE/FLEXICORE'),
                                    ('RFFRM', 'METAL'),
                                    ('RFFRM', 'REINFORCED CONCRETE'),
                                    ('RFFRM', 'LONGSPAN TRUSS'),
                                    ('RFFRM', 'PRESTRESS CONCRETE'),
                                    ('RFFRM', 'BAR JOIST & RIGID FRAME'),
                                    ('RFFRM', 'STEEL'),
                                    ('RFFRM', 'TRUSS/JOIST'),
                                    ('RFFRM', 'WOOD BEAM'),
                                    ('RFFRM', 'WOOD'),
                                    ('RFFRM', 'WOOD JOIST'),
                                    ('RFFRM', 'WOOD ON STEEL'),
                                    ('RFFRM', 'WOOD TRUSS'),
                                    ('PARKG', 'VINYL GARAGE'),
                                    ('VIEW', 'TYPE UNKNOWN'),
                                    ('LOCIN', 'TYPE UNKNOWN'),
                                    ('SCODE', 'SALE PRICE (PARTIAL)'),
                                    ('OWNSH', 'SINGLE'),
                                    ('MTGTP', 'COMMUNITY DEVELOPMENT AUTHORITY'),
                                    ('MTGTP', 'CONVENTIONAL'),
                                    ('MTGTP', 'LEASE HOLD MORTGAGE'),
                                    ('MTGTP', 'PRIVATE PARTY LENDER'),
                                    ('MTGTP', 'SMALL BUSINESS ADMINISTRATION'),
                                    ('MTGTP', 'VETERANS ADMINISTRATION'),
                                    ('MTGTP', 'WRAP-AROUND MORTGAGE'),
                                    ))
        to_code, to_description = self._read_table_file(path_parcels, recoded_descriptions, ignore)
        self._parcels_to_code = to_code
        self._parcels_to_description = to_description

    def code_to_description(self, filename, table, code):
        key = (table, code)
        lookup = self._deed_to_description if filename == 'deeds' else self._parcels_to_description
        description = lookup.get(key, None)
        if description is None:
            print 'missing', filename, table, code
            pdb.set_trace()
        return description

    def description_to_code(self, filename, table, description):
        key = (table, description)
        lookup = self._deeds_to_code if filename == 'deeds' else self._parcels_to_code
        code = lookup.get(key, None)
        if code is None:
            print 'missing', filename, table, description
            pdb.set_trace()
        return code


class CodesTest(unittest.TestCase):
    def setUp(self):
        self.codes = Codes()

    def test_init(self):
        self.assertTrue(isinstance(self.codes, Codes))

    def test_code_to_description(self):
        self.assertEqual(
            self.codes.code_to_description('deeds', 'MTGTP', 'WRP'),
            'WRAP-AROUND MORTGAGE')
        self.assertEqual(
            self.codes.code_to_description('parcels', 'LUSEI', '127'),
            'HOTEL')

    def test_description_to_code(self):
        self.assertEqual(
            self.codes.description_to_code('deeds', 'MTGTP', 'WRAP-AROUND MORTGAGE'),
            'WRP')
        self.assertEqual(
            self.codes.description_to_code('parcels', 'LUSEI', 'HOTEL'),
            '127')


if __name__ == '__main__':
    if False:
        pprint(' ')
    unittest.main()
