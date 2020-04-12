# -*- coding: utf-8 -*-
# # Debugging and Testing Pandas

# ## Code to Transform Data

# ### How to do it...

import pandas as pd
import numpy as np
import zipfile
url = 'data/kaggle-survey-2018.zip'


with zipfile.ZipFile(url) as z:
    print(z.namelist())
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


df.T


df.dtypes


df.Q1.value_counts(dropna=False)


def tweak_kag(df):
    na_mask = df.Q9.isna()
    hide_mask = df.Q9.str.startswith('I do not').fillna(False)
    df = df[~na_mask & ~hide_mask]


    q1 = (df.Q1
      .replace({'Prefer not to say': 'Another',
               'Prefer to self-describe': 'Another'})
      .rename('Gender')
    )
    q2 = df.Q2.str.slice(0,2).astype(int).rename('Age')
    def limit_countries(val):
        if val in  {'United States of America', 'India', 'China'}:
            return val
        return 'Another'
    q3 = df.Q3.apply(limit_countries).rename('Country')


    q4 = (df.Q4
     .replace({'Master’s degree': 18,
     'Bachelor’s degree': 16,
     'Doctoral degree': 20,
     'Some college/university study without earning a bachelor’s degree': 13,
     'Professional degree': 19,
     'I prefer not to answer': None,
     'No formal education past high school': 12})
     .fillna(11)
     .rename('Edu')
    )


    def only_cs_stat_val(val):
        if val not in {'cs', 'eng', 'stat'}:
            return 'another'
        return val


    q5 = (df.Q5
            .replace({
                'Computer science (software engineering, etc.)': 'cs',
                'Engineering (non-computer focused)': 'eng',
                'Mathematics or statistics': 'stat'})
             .apply(only_cs_stat_val)
             .rename('Studies'))
    def limit_occupation(val):
        if val in {'Student', 'Data Scientist', 'Software Engineer', 'Not employed',
                  'Data Engineer'}:
            return val
        return 'Another'


    q6 = df.Q6.apply(limit_occupation).rename('Occupation')


    q8 = (df.Q8
      .str.replace('+', '')
      .str.split('-', expand=True)
      .iloc[:,0]
      .fillna(-1)
      .astype(int)
      .rename('Experience')
    )


    q9 = (df.Q9
     .str.replace('+','')
     .str.replace(',','')
     .str.replace('500000', '500')
     .str.replace('I do not wish to disclose my approximate yearly compensation','')
     .str.split('-', expand=True)
     .iloc[:,0]
     .astype(int)
     .mul(1000)
     .rename('Salary'))
    return pd.concat([q1, q2, q3, q4, q5, q6, q8, q9], axis=1)


tweak_kag(df)


tweak_kag(df).dtypes


# ### How it works...

kag = tweak_kag(df)
(kag
    .groupby('Country')
    .apply(lambda g: g.Salary.corr(g.Experience))
)


# ## Apply Performance

# ### How to do it...

def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# %%timeit
q3 = df.Q3.apply(limit_countries).rename('Country')


# %%timeit
other_values = df.Q3.value_counts().iloc[3:].index
q3_2 = df.Q3.replace(other_values, 'Another')


# %%timeit
values = {'United States of America', 'India', 'China'}
q3_3 = df.Q3.where(df.Q3.isin(values), 'Another')


# %%timeit
values = {'United States of America', 'India', 'China'}
q3_4 = pd.Series(np.where(df.Q3.isin(values), df.Q3, 'Another'), 
     index=df.index)


q3.equals(q3_2)


q3.equals(q3_3)


q3.equals(q3_4)


# ### How it works...

# ### There's more...

def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


q3 = df.Q3.apply(limit_countries).rename('Country')


def debug(something):
    # what is something? A cell, series, dataframe?
    print(type(something), something)
    1/0


q3.apply(debug)


the_item = None
def debug(something):
    global the_item
    the_item = something
    return something


_ = q3.apply(debug)


the_item


# ## Improving Apply Performance with Dask, Pandarell, Swifter, and More

# ### How to do it...

from pandarallel import pandarallel
pandarallel.initialize()


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# %%timeit
res_p = df.Q3.parallel_apply(limit_countries).rename('Country')


import swifter


# %%timeit
res_s = df.Q3.swifter.apply(limit_countries).rename('Country')


import dask


# %%timeit
res_d = (dask.dataframe.from_pandas(
       df, npartitions=4)
   .map_partitions(lambda df: df.Q3.apply(limit_countries))
   .rename('Countries')
)


np_fn = np.vectorize(limit_countries)


# %%timeit
res_v = df.Q3.apply(np_fn).rename('Country')


from numba import jit


@jit
def limit_countries2(val):
     if val in  ['United States of America', 'India', 'China']:
         return val
     return 'Another'


# %%timeit
res_n = df.Q3.apply(limit_countries2).rename('Country')


# ### How it works...

# ## Inspecting Code 

# ### How to do it...

import zipfile
url = 'data/kaggle-survey-2018.zip'


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


df.Q3.apply?


df.Q3.apply??


import pandas.core.series
pandas.core.series.lib


pandas.core.series.lib.map_infer??


# ### How it works...

# ### There's more...

# ## Debugging in Jupyter

# ### How to do it...

import zipfile
url = 'data/kaggle-survey-2018.zip'


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


def add1(x):
    return x + 1


df.Q3.apply(add1)


from IPython.core.debugger import set_trace


def add1(x):
    set_trace()
    return x + 1


df.Q3.apply(add1)


# ### How it works...

# ### There's more...

# ##  Managing data integrity with Great Expectations

# ### How to do it...

kag = tweak_kag(df)


import great_expectations as ge
kag_ge = ge.from_pandas(kag)


sorted([x for x in set(dir(kag_ge)) - set(dir(kag))
    if not x.startswith('_')])


kag_ge.expect_column_to_exist('Salary')


kag_ge.expect_column_mean_to_be_between(
   'Salary', min_value=10_000, max_value=100_000)


kag_ge.expect_column_values_to_be_between(
   'Salary', min_value=0, max_value=500_000)


kag_ge.expect_column_values_to_not_be_null('Salary')


kag_ge.expect_column_values_to_match_regex(
    'Country', r'America|India|Another|China')


kag_ge.expect_column_values_to_be_of_type(
   'Salary', type_='int')


kag_ge.save_expectation_suite('kaggle_expectations.json')


kag_ge.to_csv('kag.csv')
import json
ge.validate(ge.read_csv('kag.csv'), 
    expectation_suite=json.load(
        open('kaggle_expectations.json')))


# ### How it works...

# ## Using pytest with pandas

# ### How to do it...

# ### How it works...

# ### There's more...

# ## Generating Tests with Hypothesis

# ### How to do it...

# ### How it works...

