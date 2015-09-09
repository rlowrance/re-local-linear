'decoding of files provided by CoreLogic'

import pdb


def is_grant_deed(deed_df):
    'return mask for deeds dataframe'
    return deed_df['DOCUMENT TYPE CODE'] == 'G'  # coding is DEEDC


def is_arms_length(deed_df):
    'return mask for deeds dataframe'
    return deed_df['PRI CAT CODE'] == 'A'


def is_sfr(parcel_df):
    'return mask for parcel dataframe'
    return parcel_df['UNIVERSAL LAND USE CODE'] == 163


def is_industry(parcel_df):
    'conform to definition in R code: any.industrial in PROPN.R'
    pdb.set_trace()
    code = parcel_df['PROPERTY INDICATOR CODE']
    return (code == 50 | code == 51 | code == 52)


def is_park(parcel_df):
    'conform to definition in R code: park in LUSEI.R'
    pdb.set_trace()
    code = parcel_df['UNIVERSAL LAND USE CODE']
    return code == 25


def is_retail(parcel_df):
    'conform to definition in R code: retail in PROPN.R'
    pdb.set_trace()
    code = parcel_df['PROPERTY INDICATOR CODE']
    return code == 757


def is_school(parcel_df):
    'conform to definition in R code: any.school in LUSEI.R'
    pdb.set_trace()
    code = parcel_df['UNIVERSAL LAND USE CODE']
    return code >= 650 & code <= 665  # not universities


# PROPN decoder
property_indicator_code = {
    10: 'single family residence',
    11: 'condominium',
    21: 'duplex',
    22: 'apartment',
    23: 'hotel',
    24: 'commercial',
    25: 'retail',
    26: 'service',
    27: 'office.building',
    28: 'warehouse',
    29: 'financial institution',
    30: 'hospital',
    31: 'parking',
    32: 'amusement',
    50: 'industrial',
    51: 'industrial light',
    52: 'industrial heavy',
    53: 'transport',
    54: 'utilities',
    70: 'agricultural',
    80: 'vacant',
    90: 'exempt',
    0: 'missing'
}


# these functions are not used.
# they rely on deconding the UNIVERSAL LAND USE CODE via the LUSEI.R file

def is_retail2(parcel_df):
    'determine if its nice commercial (aka, retail)'
    code = parcel_df['UNIVERSAL LAND USE CODE']
    # don't count as retail:
    #  store franchise, fast food franchise
    return (code == 261 |  # restaurant
            code == 262 |  # restaurant drive in
            code == 276 |  # apparel
            code == 278 |  # store buildings
            code == 279 |  # stores and offices
            code == 281 |  # stores and residential
            code == 282 |  # retail trade
            code == 283 |  # supermarket
            code == 284 |  # food stores
            code == 285)   # tavern


def is_school2(parcel_df):
    pdb.set_trace()
    code = parcel_df['UNIVERSAL LAND USE CODE']
    return code >= 650 & code <= 680   # school .. university


def is_park2(parcel_df):
    pdb.set_trace()
    code = parcel_df['UNIVERSAL LAND USE CODE']
    return code == 757  # park


def is_industry2(parcel_df):
    'is industrial or commercial, except for retail'
    code = parcel_df['UNIVERSAL LAND USE CODE']
    return (code >= 200 &
            (not is_park(parcel_df)) &
            (not is_retail(parcel_df)) &
            (not is_school(parcel_df)))
