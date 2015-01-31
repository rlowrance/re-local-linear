class TRNTP:
    '''Transaction type code'''
    def __init__(self):
        self.ikind = {1: 'resale',
                      2: 'refinance',
                      3: 'new construction',
                      4: 'timeshare',
                      6: 'construction loan',
                      7: 'seller carryback',
                      9: 'nominal'}

    def is_resale(self, v):
        return v == 1

    def is_new_construction(self, v):
        return v == 3

    def is_resale_or_new_construction(self, v):
        return self.is_resale(v) | self.is_new_construction(v)
