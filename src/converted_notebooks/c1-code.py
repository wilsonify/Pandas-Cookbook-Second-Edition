# # Chapter 1: Pandas Foundations

import pandas as pd
import numpy as np


# ## Introduction

# ## Dissecting the anatomy of a DataFrame

pd.set_option('max_columns', 4, 'max_rows', 10)


movies = pd.read_csv('data/movie.csv')
movies.head()


# ### How it works...

# ## DataFrame Attributes

# ### How to do it... {#how-to-do-it-1}

movies = pd.read_csv('data/movie.csv')
columns = movies.columns
index = movies.index
data = movies.values


columns


index


data


type(index)


type(columns)


type(data)


issubclass(pd.RangeIndex, pd.Index)


# ### How it works...

# ### There's more

index.values


columns.values


# ## Understanding data types

# ### How to do it... {#how-to-do-it-2}

movies = pd.read_csv('data/movie.csv')


movies.dtypes


movies.get_dtype_counts()


movies.info()


# ### How it works...

pd.Series(['Paul', np.nan, 'George']).dtype


# ### There's more...

# ### See also

# ## Selecting a Column

# ### How to do it... {#how-to-do-it-3}

movies = pd.read_csv('data/movie.csv')
movies['director_name']


movies.director_name


movies.loc[:, 'director_name']


movies.iloc[:, 1]


movies['director_name'].index


movies['director_name'].dtype


movies['director_name'].size


movies['director_name'].name


type(movies['director_name'])


movies['director_name'].apply(type).unique()


# ### How it works...

# ### There's more

# ### See also

# ## Calling Series Methods

s_attr_methods = set(dir(pd.Series))
len(s_attr_methods)


df_attr_methods = set(dir(pd.DataFrame))
len(df_attr_methods)


len(s_attr_methods & df_attr_methods)


# ### How to do it... {#how-to-do-it-4}

movies = pd.read_csv('data/movie.csv')
director = movies['director_name']
fb_likes = movies['actor_1_facebook_likes']


director.dtype


fb_likes.dtype


director.head()


director.sample(n=5, random_state=42)


fb_likes.head()


director.value_counts()


fb_likes.value_counts()


director.size


director.shape


len(director)


director.unique()


director.count()


fb_likes.count()


fb_likes.quantile()


fb_likes.min()


fb_likes.max()


fb_likes.mean()


fb_likes.median()


fb_likes.std()


fb_likes.describe()


director.describe()


fb_likes.quantile(.2)


fb_likes.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])


director.isna()


fb_likes_filled = fb_likes.fillna(0)
fb_likes_filled.count()


fb_likes_dropped = fb_likes.dropna()
fb_likes_dropped.size


# ### How it works...

# ### There's more...

director.value_counts(normalize=True)


director.hasnans


director.notna()


# ### See also

# ## Series Operations

5 + 9    # plus operator example. Adds 5 and 9


# ### How to do it... {#how-to-do-it-5}

movies = pd.read_csv('data/movie.csv')
imdb_score = movies['imdb_score']
imdb_score


imdb_score + 1


imdb_score * 2.5


imdb_score // 7


imdb_score > 7


director = movies['director_name']
director == 'James Cameron'


# ### How it works...

# ### There's more...

imdb_score.add(1)   # imdb_score + 1


imdb_score.gt(7)   # imdb_score > 7


# ### See also

# ## Chaining Series Methods

# ### How to do it... {#how-to-do-it-6}

movies = pd.read_csv('data/movie.csv')
fb_likes = movies['actor_1_facebook_likes']
director = movies['director_name']


director.value_counts().head(3)


fb_likes.isna().sum()


fb_likes.dtype


(fb_likes.fillna(0)
         .astype(int)
         .head()
)


# ### How it works...

# ### There's more...

(fb_likes.fillna(0)
         #.astype(int)
         #.head()
)


(fb_likes.fillna(0)
         .astype(int)
         #.head()
)


fb_likes.isna().mean()


fb_likes.fillna(0) \
        .astype(int) \
        .head()


def debug_df(df):
    print("BEFORE")
    print(df)
    print("AFTER")
    return df


(fb_likes.fillna(0)
         .pipe(debug_df)
         .astype(int) 
         .head()
)


intermediate = None
def get_intermediate(df):
    global intermediate
    intermediate = df
    return df


res = (fb_likes.fillna(0)
         .pipe(get_intermediate)
         .astype(int) 
         .head()
)


intermediate


# ## Renaming Column Names

# ### How to do it...

movies = pd.read_csv('data/movie.csv')


col_map = {'director_name':'Director Name', 
             'num_critic_for_reviews': 'Critical Reviews'} 


movies.rename(columns=col_map).head()


# ### How it works... {#how-it-works-8}

# ### There's more {#theres-more-7}

idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .set_index('movie_title')
   .rename(index=idx_map, columns=col_map)
   .head(3)
)


movies = pd.read_csv('data/movie.csv', index_col='movie_title')
ids = movies.index.tolist()
columns = movies.columns.tolist()


# # rename the row and column labels with list assignments

ids[0] = 'Ratava'
ids[1] = 'POC'
ids[2] = 'Ertceps'
columns[1] = 'director'
columns[-2] = 'aspect'
columns[-1] = 'fblikes'
movies.index = ids
movies.columns = columns


movies.head(3)


def to_clean(val):
    return val.strip().lower().replace(' ', '_')


movies.rename(columns=to_clean).head(3)


cols = [col.strip().lower().replace(' ', '_')
        for col in movies.columns]
movies.columns = cols
movies.head(3)


# ## Creating and Deleting columns

# ### How to do it... {#how-to-do-it-9}

movies = pd.read_csv('data/movie.csv')
movies['has_seen'] = 0


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .rename(index=idx_map, columns=col_map)
   .assign(has_seen=0)
)


total = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])


total.head(5)


cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
    'actor_3_facebook_likes','director_facebook_likes']
sum_col = movies[cols].sum(axis='columns')
sum_col.head(5)


movies.assign(total_likes=sum_col).head(5)


def sum_likes(df):
   return df[[c for c in df.columns
              if 'like' in c]].sum(axis=1)


movies.assign(total_likes=sum_likes).head(5)


(movies
   .assign(total_likes=sum_col)
   ['total_likes']
   .isna()
   .sum()
)


(movies
   .assign(total_likes=total)
   ['total_likes']
   .isna()
   .sum()
)


(movies
   .assign(total_likes=total.fillna(0))
   ['total_likes']
   .isna()
   .sum()
)


def cast_like_gt_actor_director(df):
    return df['cast_total_facebook_likes'] >= \
           df['total_likes']


df2 = (movies
   .assign(total_likes=total,
           is_cast_likes_more = cast_like_gt_actor_director)
)


df2['is_cast_likes_more'].all()


df2 = df2.drop(columns='total_likes')


actor_sum = (movies
   [[c for c in movies.columns if 'actor_' in c and '_likes' in c]]
   .sum(axis='columns')
)


actor_sum.head(5)


movies['cast_total_facebook_likes'] >= actor_sum


movies['cast_total_facebook_likes'].ge(actor_sum)


movies['cast_total_facebook_likes'].ge(actor_sum).all()


pct_like = (actor_sum
    .div(movies['cast_total_facebook_likes'])
)


pct_like.describe()


pd.Series(pct_like.values,
    index=movies['movie_title'].values).head()


# ### How it works... {#how-it-works-9}

# ### There's more... {#theres-more-8}

profit_index = movies.columns.get_loc('gross') + 1
profit_index


movies.insert(loc=profit_index,
              column='profit',
              value=movies['gross'] - movies['budget'])


del movies['director_name']


# ### See also

