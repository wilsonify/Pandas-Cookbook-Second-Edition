# # Selecting Subsets of Data 

import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Selecting Series data

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')
city = college['CITY']
city


city['Alabama A & M University']


city.loc['Alabama A & M University']


city.iloc[0]


city[['Alabama A & M University', 'Alabama State University']]


city.loc[['Alabama A & M University', 'Alabama State University']]


city.iloc[[0, 4]]


city['Alabama A & M University': 'Alabama State University']


city[0:5]


city.loc['Alabama A & M University': 'Alabama State University']


city.iloc[0:5]


alabama_mask = city.isin(['Birmingham', 'Montgomery'])
city[alabama_mask]


# ### How it works...

s = pd.Series([10, 20, 35, 28], index=[5,2,3,1])
s


s[0:4]


s[5]


s[1]


# ### There's more...

college.loc['Alabama A & M University', 'CITY']


college.iloc[0, 0]


college.loc[['Alabama A & M University', 
    'Alabama State University'], 'CITY']


college.iloc[[0, 4], 0]


college.loc['Alabama A & M University':
    'Alabama State University', 'CITY']


college.iloc[0:5, 0]


city.loc['Reid State Technical College':
         'Alabama State University']


# ## Selecting DataFrame rows

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.sample(5, random_state=42)


college.iloc[60]


college.loc['University of Alaska Anchorage']


college.iloc[[60, 99, 3]]


labels = ['University of Alaska Anchorage',
          'International Academy of Hair Design',
          'University of Alabama in Huntsville']
college.loc[labels]


college.iloc[99:102]


start = 'International Academy of Hair Design'
stop = 'Mesa Community College'
college.loc[start:stop]


# ### How it works...

# ### There's more...

college.iloc[[60, 99, 3]].index.tolist()


# ## Selecting DataFrame rows and columns simultaneously

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.iloc[:3, :4]


college.loc[:'Amridge University', :'MENONLY']


college.iloc[:, [4,6]].head()


college.loc[:, ['WOMENONLY', 'SATVRMID']].head()


college.iloc[[100, 200], [7, 15]]


rows = ['GateWay Community College',
        'American Baptist Seminary of the West']
columns = ['SATMTMID', 'UGDS_NHPI']
college.loc[rows, columns]


college.iloc[5, -4]


college.loc['The University of Alabama', 'PCTFLOAN']


college.iloc[90:80:-2, 5]


start = 'Empire Beauty School-Flagstaff'
stop = 'Arizona State University-Tempe'
college.loc[start:stop:-2, 'RELAFFIL']


# ### How it works...

# ### There's more...

# ## Selecting data with both integers and labels

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')


col_start = college.columns.get_loc('UGDS_WHITE')
col_end = college.columns.get_loc('UGDS_UNKN') + 1
col_start, col_end


college.iloc[:5, col_start:col_end]


# ### How it works...

# ### There's more...

row_start = college.index[10]
row_end = college.index[15]
college.loc[row_start:row_end, 'UGDS_WHITE':'UGDS_UNKN']


college.ix[10:16, 'UGDS_WHITE':'UGDS_UNKN']


college.iloc[10:16].loc[:, 'UGDS_WHITE':'UGDS_UNKN']


# ## Slicing lexicographically

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')


college.loc['Sp':'Su']


college = college.sort_index()


college.loc['Sp':'Su']


# ### How it works...

# ### There's more...

college = college.sort_index(ascending=False)
college.index.is_monotonic_decreasing


college.loc['E':'B']


