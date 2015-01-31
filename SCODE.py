class SCODE:
    '''Sales Code'''
    def __init__(self):
        self.kinds = {'C': 'confirmed',
                      'E': 'estimated',
                      'F': 'sale price full',
                      'L': 'sale price partial 1',
                      'N': 'not of public record',
                      'P': 'sale price partical 2',
                      'R': 'lease',
                      'U': 'known',
                      'V': 'verified'}

    def is_sale_price_full(self, v):
        return v == 'F'
