# # Beginning Data Analysis 

import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Developing a data analysis routine

# ### How to do it...

college = pd.read_csv('data/college.csv')
college.sample(random_state=42)


college.shape


college.info()


college.describe(include=[np.number]).T


college.describe(include=[np.object, pd.Categorical]).T


# ### How it works...

# ### There's more...

college.describe(include=[np.number],
   percentiles=[.01, .05, .10, .25, .5,
                .75, .9, .95, .99]).T


# ## Data dictionaries

pd.read_csv('data/college_data_dictionary.csv')


# ## Reducing memory by changing data types

# ### How to do it...

college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER',
   'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.head()


col2.dtypes


original_mem = col2.memory_usage(deep=True)
original_mem


col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)    


col2.dtypes


college[different_cols].memory_usage(deep=True)


col2.select_dtypes(include=['object']).nunique()


col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes


new_mem = col2.memory_usage(deep=True)
new_mem


new_mem / original_mem


# ### How it works...

# ### There's more...

college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep=True)


college['MENONLY'].dtype


college['MENONLY'].astype(np.int8)


college.assign(MENONLY=college['MENONLY'].astype('float16'),
    RELAFFIL=college['RELAFFIL'].astype('int8'))


college.index = pd.Int64Index(college.index)
college.index.memory_usage() # previously was just 80


# ## Selecting the smallest of the largest

# ### How to do it...

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()


movie2.nlargest(100, 'imdb_score').head()


(movie2
  .nlargest(100, 'imdb_score')
  .nsmallest(5, 'budget')
)


# ### How it works...

# ### There's more...

# ## Selecting the largest of each group by sorting

# ### How to do it...

movie = pd.read_csv('data/movie.csv')
movie[['movie_title', 'title_year', 'imdb_score']]


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values('title_year', ascending=False)
)


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
)


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
  .drop_duplicates(subset='title_year')
)


# ### How it works...

# ## There's more...

(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .groupby('title_year', as_index=False)
  .apply(lambda df: df.sort_values('imdb_score',
         ascending=False).head(1))
  .sort_values('title_year', ascending=False)
)


(movie
  [['movie_title', 'title_year',
    'content_rating', 'budget']]
   .sort_values(['title_year',
       'content_rating', 'budget'],
       ascending=[False, False, True])
   .drop_duplicates(subset=['title_year',
        'content_rating'])
)


# ## Replicating nlargest with sort_values

# ### How to do it...

movie = pd.read_csv('data/movie.csv')
(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score') 
   .nsmallest(5, 'budget')
)


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100)
)


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100) 
   .sort_values('budget')
   .head(5)
)


# ### How it works...

(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score')
   .tail()
)


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False) 
   .head(100)
   .tail()
)


# ## Calculating a trailing stop order price

# ### How to do it...

import datetime
import pandas_datareader.data as web
import requests_cache
session = requests_cache.CachedSession(
   cache_name='cache', backend='sqlite', 
   expire_after=datetime.timedelta(days=90))


tsla = web.DataReader('tsla', data_source='yahoo',
   start='2017-1-1', session=session)
tsla.head(8)


tsla_close = tsla['Close']


tsla_cummax = tsla_close.cummax()
tsla_cummax.head()


(tsla
  ['Close']
  .cummax()
  .mul(.9)
  .head()
)


# ### How it works...

# ### There's more...

