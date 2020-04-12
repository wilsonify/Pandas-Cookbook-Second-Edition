# -*- coding: utf-8 -*-
# # Restructuring Data into a Tidy Form

import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Tidying variable values as column names with stack

state_fruit = pd.read_csv('data/state_fruit.csv', index_col=0)
state_fruit


# ### How to do it...

state_fruit.stack()


(state_fruit
   .stack()
   .reset_index()
)


(state_fruit
   .stack()
   .reset_index()
   .rename(columns={'level_0':'state', 
      'level_1': 'fruit', 0: 'weight'})
)


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
)


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
    .reset_index(name='weight')
)


# ### How it works...

# ### There's more...

state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


state_fruit2.stack()


state_fruit2.set_index('State').stack()


# ## Tidying variable values as column names with melt

# ### How to do it...

state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


state_fruit2.melt(id_vars=['State'],
    value_vars=['Apple', 'Orange', 'Banana'])


state_fruit2.melt(id_vars=['State'],
                   value_vars=['Apple', 'Orange', 'Banana'],
                   var_name='Fruit',
                   value_name='Weight')


# ### How it works...

# ### There's more...

state_fruit2.melt()


state_fruit2.melt(id_vars='State')


# ## Stacking multiple groups of variables simultaneously

movie = pd.read_csv('data/movie.csv')
actor = movie[['movie_title', 'actor_1_name',
               'actor_2_name', 'actor_3_name',
               'actor_1_facebook_likes',
               'actor_2_facebook_likes',
               'actor_3_facebook_likes']]
actor.head()


# ### How to do it...

def change_col_name(col_name):
    col_name = col_name.replace('_name', '')
    if 'facebook' in col_name:
        fb_idx = col_name.find('facebook')
        col_name = (col_name[:5] + col_name[fb_idx - 1:] 
               + col_name[5:fb_idx-1])
    return col_name


actor2 = actor.rename(columns=change_col_name)
actor2


stubs = ['actor', 'actor_facebook_likes']
actor2_tidy = pd.wide_to_long(actor2,
    stubnames=stubs,
    i=['movie_title'],
    j='actor_num',
    sep='_')
actor2_tidy.head()


# ### How it works...

# ### There's more...

df = pd.read_csv('data/stackme.csv')
df


df.rename(columns = {'a1':'group1_a1', 'b2':'group1_b2',
                     'd':'group2_a1', 'e':'group2_b2'})


pd.wide_to_long(
       df.rename(columns = {'a1':'group1_a1', 
                 'b2':'group1_b2',
                 'd':'group2_a1', 'e':'group2_b2'}),
    stubnames=['group1', 'group2'],
    i=['State', 'Country', 'Test'],
    j='Label',
    suffix='.+',
    sep='_')


# ## Inverting stacked data

# ### How to do it...

usecol_func = lambda x: 'UGDS_' in x or x == 'INSTNM'
college = pd.read_csv('data/college.csv',
    index_col='INSTNM',
    usecols=usecol_func)
college


college_stacked = college.stack()
college_stacked


college_stacked.unstack()


college2 = pd.read_csv('data/college.csv',
   usecols=usecol_func)
college2


college_melted = college2.melt(id_vars='INSTNM',
    var_name='Race',
    value_name='Percentage')
college_melted


melted_inv = college_melted.pivot(index='INSTNM',
    columns='Race',
    values='Percentage')
melted_inv


college2_replication = (melted_inv
    .loc[college2['INSTNM'], college2.columns[1:]]
    .reset_index()
)
college2.equals(college2_replication)


# ### How it works...

# ### There's more...

college.stack().unstack(0)


college.T
college.transpose()


# ## Unstacking after a groupby aggregation

# ### How to do it...

employee = pd.read_csv('data/employee.csv')
(employee
    .groupby('RACE')
    ['BASE_SALARY']
    .mean()
    .astype(int)
)


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
)


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('GENDER')
)


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('RACE')
)


# ### How it works...

# ### There's more...

(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
)


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
    .unstack('GENDER')
)


# ## Replicating pivot_table with a groupby aggregation

# ### How to do it...

flights = pd.read_csv('data/flights.csv')
fpt = flights.pivot_table(index='AIRLINE',
    columns='ORG_AIR',
    values='CANCELLED',
    aggfunc='sum',
    fill_value=0).round(2)
fpt


(flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
)


fpg = (flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
    .unstack('ORG_AIR', fill_value=0)
)


fpt.equals(fpg)


# ### How it works...

# ### There's more...

flights.pivot_table(index=['AIRLINE', 'MONTH'],
    columns=['ORG_AIR', 'CANCELLED'],
    values=['DEP_DELAY', 'DIST'],
    aggfunc=['sum', 'mean'],
    fill_value=0)


(flights
    .groupby(['AIRLINE', 'MONTH', 'ORG_AIR', 'CANCELLED']) 
    ['DEP_DELAY', 'DIST'] 
    .agg(['mean', 'sum']) 
    .unstack(['ORG_AIR', 'CANCELLED'], fill_value=0) 
    .swaplevel(0, 1, axis='columns')
)


# ## Renaming axis levels for easy reshaping

# ### How to do it...

college = pd.read_csv('data/college.csv')
(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR',
       axis='index')
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR', axis='index') 
    .sort_index(level='RELAFFIL', axis='index') 
    .sort_index(level='AGG_COLS', axis='columns')
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .unstack(['RELAFFIL', 'STABBR'])
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack(['AGG_FUNCS', 'AGG_COLS'])
)


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .unstack(['STABBR', 'RELAFFIL']) 
)


# ### How it works...

# ### There's more...

(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis([None, None], axis='index') 
    .rename_axis([None, None], axis='columns')
)


# ## Tidying when multiple variables are stored as column names

# ### How to do it...

weightlifting = pd.read_csv('data/weightlifting_men.csv')
weightlifting


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
)


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
)


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0])
)


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
tidy = pd.concat([melted
           ['sex_age']
           .str.split(expand=True)
           .rename(columns={0:'Sex', 1:'Age Group'})
           .assign(Sex=lambda df_: df_.Sex.str[0]),
          melted[['Weight Category', 'Qual Total']]],
          axis='columns'
)
tidy


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
(melted
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0],
            Category=melted['Weight Category'],
            Total=melted['Qual Total'])
)


# ### How it works...

# ### There's more...

tidy2 = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    .assign(Sex=lambda df_:df_.sex_age.str[0],
            **{'Age Group':(lambda df_: (df_
                .sex_age
                .str.extract(r'(\d{2}[-+](?:\d{2})?)',
                             expand=False)))})
    .drop(columns='sex_age')
)


tidy2


tidy.sort_index(axis=1).equals(tidy2.sort_index(axis=1))


# ## Tidying when multiple variables are stored is a single column

# ### How to do it...

inspections = pd.read_csv('data/restaurant_inspections.csv',
    parse_dates=['Date'])
inspections


inspections.pivot(index=['Name', 'Date'],
    columns='Info', values='Value')


inspections.set_index(['Name','Date', 'Info'])


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
)


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
)


def flatten0(df_):
    df_.columns = df_.columns.droplevel(0).rename(None)
    return df_


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
    .pipe(flatten0)
)


(inspections
    .set_index(['Name','Date', 'Info']) 
    .squeeze() 
    .unstack('Info') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

(inspections
    .pivot_table(index=['Name', 'Date'],
                 columns='Info',
                 values='Value',
                 aggfunc='first') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ## Tidying when two or more values are stored in the same cell

# ### How to do it..

cities = pd.read_csv('data/texas_cities.csv')
cities


geolocations = cities.Geolocation.str.split(pat='. ',
    expand=True)
geolocations.columns = ['latitude', 'latitude direction',
    'longitude', 'longitude direction']


geolocations = geolocations.astype({'latitude':'float',
   'longitude':'float'})
geolocations.dtypes


(geolocations
    .assign(city=cities['City'])
)


# ### How it works...

geolocations.apply(pd.to_numeric, errors='ignore')


# ### There's more...

cities.Geolocation.str.split(pat=r'° |, ', expand=True)


cities.Geolocation.str.extract(r'([0-9.]+). (N|S), ([0-9.]+). (E|W)',
   expand=True)


# ## Tidying when variables are stored in column names and values

# ### Getting ready

sensors = pd.read_csv('data/sensors.csv')
sensors


sensors.melt(id_vars=['Group', 'Property'], var_name='Year')


(sensors
    .melt(id_vars=['Group', 'Property'], var_name='Year') 
    .pivot_table(index=['Group', 'Year'],
                 columns='Property', values='value') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

(sensors
    .set_index(['Group', 'Property']) 
    .stack() 
    .unstack('Property') 
    .rename_axis(['Group', 'Year'], axis='index') 
    .rename_axis(None, axis='columns') 
    .reset_index()
)


