# # Filtering Rows

import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Calculating boolean statistics

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie[['duration']].head()


movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)


movie_2_hours.sum()


movie_2_hours.mean()


movie['duration'].dropna().gt(120).mean()


movie_2_hours.describe()


# ### How it works...

movie_2_hours.value_counts(normalize=True)


movie_2_hours.astype(int).describe()


# ### There's more...

actors = movie[['actor_1_facebook_likes',
   'actor_2_facebook_likes']].dropna()
(actors['actor_1_facebook_likes'] >
      actors['actor_2_facebook_likes']).mean()


# ## Constructing multiple boolean conditions

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')


criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = ((movie.title_year < 2000) |
   (movie.title_year > 2009))


criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()


# ### How it works...

# ### There's more...

5 < 10 and 3 > 4


5 < 10 and 3 > 4


True and 3 > 4


True and False


False


movie.title_year < 2000 | movie.title_year > 2009


# ## Filtering with boolean arrays

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3


crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = ((movie.title_year >= 2000) &
(movie.title_year <= 2010))
final_crit_b = crit_b1 & crit_b2 & crit_b3


final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()


movie[final_crit_all].head()


movie.loc[final_crit_all].head()


cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)


# ### How it works...

movie.iloc[final_crit_all]


movie.iloc[final_crit_all.values]


# ### There's more...

final_crit_a2 = ((movie.imdb_score > 8) & 
   (movie.content_rating == 'PG-13') & 
   ((movie.title_year < 2000) |
    (movie.title_year > 2009))
)
final_crit_a2.equals(final_crit_a)


# ## Comparing Row Filtering and Index Filtering

# ### How to do it...

college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()


college2 = college.set_index('STABBR')
college2.loc['TX'].head()


# %timeit college[college['STABBR'] == 'TX']


# %timeit college2.loc['TX']


# %timeit college2 = college.set_index('STABBR')


# ### How it works...

# ### There's more...

states = ['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]


college2.loc[states]


# ## Selecting with unique and sorted indexes

# ### How to do it...

college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')
college2.index.is_monotonic


college3 = college2.sort_index()
college3.index.is_monotonic


# %timeit college[college['STABBR'] == 'TX']


# %timeit college2.loc['TX']


# %timeit college3.loc['TX']


college_unique = college.set_index('INSTNM')
college_unique.index.is_unique


college[college['INSTNM'] == 'Stanford University']


college_unique.loc['Stanford University']


college_unique.loc[['Stanford University']]


# %timeit college[college['INSTNM'] == 'Stanford University']


# %timeit college_unique.loc[['Stanford University']]


# ### How it works...

# ### There's more...

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()


college.loc['Miami, FL'].head()


# %%timeit
crit1 = college['CITY'] == 'Miami'
crit2 = college['STABBR'] == 'FL'
college[crit1 & crit2]


# %timeit college.loc['Miami, FL']


# ## Translating SQL WHERE clauses

# ### How to do it...

employee = pd.read_csv('data/employee.csv')


employee.dtypes


employee.DEPARTMENT.value_counts().head()


employee.GENDER.value_counts()


employee.BASE_SALARY.describe()


depts = ['Houston Police Department-HPD',
   'Houston Fire Department (HFD)']
criteria_dept = employee.DEPARTMENT.isin(depts)
criteria_gender = employee.GENDER == 'Female'
criteria_sal = ((employee.BASE_SALARY >= 80000) & 
   (employee.BASE_SALARY <= 120000))


criteria_final = (criteria_dept &
   criteria_gender &
   criteria_sal)


select_columns = ['UNIQUE_ID', 'DEPARTMENT',
                  'GENDER', 'BASE_SALARY']
employee.loc[criteria_final, select_columns].head()


# ### How it works...

# ### There's more...

criteria_sal = employee.BASE_SALARY.between(80_000, 120_000)


top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria]


# ## Improving readability of boolean indexing with the query method

# ### How to do it...

employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 'DEPARTMENT',
                  'GENDER', 'BASE_SALARY']


qs = "DEPARTMENT in @depts "\
     " and GENDER == 'Female' "\
     " and 80000 <= BASE_SALARY <= 120000"
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()


# ### How it works...

# ### There's more...

top10_depts = (employee.DEPARTMENT.value_counts() 
   .index[:10].tolist()
)
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()


# ## Preserving Series size with the where method

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()


fb_likes.describe()


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fb_likes.hist(ax=ax)
fig.savefig('/tmp/c7-hist.png', dpi=300)     # doctest: +SKIP


criteria_high = fb_likes < 20_000
criteria_high.mean().round(2)


fb_likes.where(criteria_high).head()


fb_likes.where(criteria_high, other=20000).head()


criteria_low = fb_likes > 300
fb_likes_cap = (fb_likes
   .where(criteria_high, other=20_000)
   .where(criteria_low, 300)
)
fb_likes_cap.head()


len(fb_likes), len(fb_likes_cap)


fig, ax = plt.subplots(figsize=(10, 8))
fb_likes_cap.hist(ax=ax)
fig.savefig('/tmp/c7-hist2.png', dpi=300)     # doctest: +SKIP


# ### How it works...

# ### There's more...

fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)


# ## Masking DataFrame rows

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isna()
criteria = c1 | c2


movie.mask(criteria).head()


movie_mask = (movie
    .mask(criteria)
    .dropna(how='all')
)
movie_mask.head()


movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)


movie_mask.shape == movie_boolean.shape


movie_mask.dtypes == movie_boolean.dtypes


from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask,
       check_dtype=False)


# ### How it works...

# ### There's more...

# %timeit movie.mask(criteria).dropna(how='all')


# %timeit movie[movie['title_year'] < 2010]


# ## Selecting with booleans, integer location, and labels

# ### How to do it...

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2


movie_loc = movie.loc[criteria]
movie_loc.head()


movie_loc.equals(movie[criteria])


movie_iloc = movie.iloc[criteria]


movie_iloc = movie.iloc[criteria.values]
movie_iloc.equals(movie_loc)


criteria_col = movie.dtypes == np.int64
criteria_col.head()


movie.loc[:, criteria_col].head()


movie.iloc[:, criteria_col.values].head()


cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')


col_index = [movie.columns.get_loc(col) for col in cols]
col_index


movie.iloc[criteria.values, col_index].sort_values('imdb_score')


# ### How it works...

a = criteria.values
a[:5]


len(a), len(criteria)


movie.select_dtypes(int)


