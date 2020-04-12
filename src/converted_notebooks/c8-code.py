# # Index Alignment

import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Examining the Index object

# ### How to do it...

college = pd.read_csv('data/college.csv')
columns = college.columns
columns


columns.values


columns[5]


columns[[1,8,10]]


columns[-7:-4]


columns.min(), columns.max(), columns.isnull().sum()


columns + '_A'


columns > 'G'


columns[1] = 'city'


# ### How it works...

# ### There's more...

c1 = columns[:4]
c1


c2 = columns[2:6]
c2


c1.union(c2) # or `c1 | c2`


c1.symmetric_difference(c2) # or `c1 ^ c2`


# ## Producing Cartesian products

# ### How to do it...

s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1


s2 = pd.Series(index=list('cababb'), data=np.arange(6))
s2


s1 + s2


# ### How it works...

# ### There's more...

s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1 + s2


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 + s2


s3 = pd.Series(index=list('ab'), data=np.arange(2))
s4 = pd.Series(index=list('ba'), data=np.arange(2))
s3 + s4


# ## Exploding indexes

# ### How to do it...

employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()


salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2


salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2


salary1 = salary1.sort_index()
salary1.head()


salary2.head()


salary_add = salary1 + salary2


salary_add.head()


salary_add1 = salary1 + salary1
len(salary1), len(salary2), len(salary_add), len(salary_add1)


# ### How it works...

# ### There's more...

index_vc = salary1.index.value_counts(dropna=False)
index_vc


index_vc.pow(2).sum()


# ## Filling values with unequal indexes

baseball_14 = pd.read_csv('data/baseball14.csv',
   index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv',
   index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv',
   index_col='playerID')
baseball_14.head()


baseball_14.index.difference(baseball_15.index)


baseball_14.index.difference(baseball_16.index)


hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()


(hits_14 + hits_15).head()


hits_14.add(hits_15, fill_value=0).head()


hits_total = (hits_14
   .add(hits_15, fill_value=0)
   .add(hits_16, fill_value=0)
)
hits_total.head()


hits_total.hasnans


# ### How it works...

s = pd.Series(index=['a', 'b', 'c', 'd'],
              data=[np.nan, 3, np.nan, 1])
s


s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1


s.add(s1, fill_value=5)


# ### There's more...

df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()


df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()


(df_14 + df_15).head(10).style.highlight_null('yellow')


(df_14
.add(df_15, fill_value=0)
.head(10)
.style.highlight_null('yellow')
)

# ## Adding columns from different DataFrames

# ### How to do it...

employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]


dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
    ascending=[True, False])


max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()


max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')


employee = (employee
   .assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY'])
)
employee


employee.query('BASE_SALARY > MAX_DEPT_SALARY')


employee = pd.read_csv('data/employee.csv')
max_dept_sal = (employee
    [['DEPARTMENT', 'BASE_SALARY']]
    .sort_values(['DEPARTMENT', 'BASE_SALARY'],
        ascending=[True, False])
    .drop_duplicates(subset='DEPARTMENT')
    .set_index('DEPARTMENT')
)


(employee
   .set_index('DEPARTMENT')
   .assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY'])
)


# ### How it works...

random_salary = (dept_sal
    .sample(n=10, random_state=42)
    .set_index('DEPARTMENT')
)
random_salary


employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']


# ### There's more...

(employee
    .set_index('DEPARTMENT')
    .assign(MAX_SALARY2=max_dept_sal['BASE_SALARY'].head(3))
    .MAX_SALARY2
    .value_counts()
)


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .transform('max')
)


(employee
    .assign(MAX_DEPT_SALARY=max_sal)
)


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .max()
)


(employee
    .merge(max_sal.rename('MAX_DEPT_SALARY'),
           how='left', left_on='DEPARTMENT',
           right_index=True)
)


# ## Highlighting the maximum value from each column

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.dtypes


college.MD_EARN_WNE_P10.sample(10, random_state=42)


college.GRAD_DEBT_MDN_SUPP.sample(10, random_state=42)


college.MD_EARN_WNE_P10.value_counts()


set(college.MD_EARN_WNE_P10.apply(type))


college.GRAD_DEBT_MDN_SUPP.value_counts()


cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')


college.dtypes.loc[cols]


college_n = college.select_dtypes('number')
college_n.head()


binary_only = college_n.nunique() == 2
binary_only.head()


binary_cols = binary_only[binary_only].index.tolist()
binary_cols


college_n2 = college_n.drop(columns=binary_cols)
college_n2.head()


max_cols = college_n2.idxmax()
max_cols


unique_max_cols = max_cols.unique()
unique_max_cols[:5]


college_n2.loc[unique_max_cols] #.style.highlight_max()

college_n2.loc[unique_max_cols].style.highlight_max()


def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


def select_rows_with_max_cols(df):
    max_cols = df.idxmax()
    unique = max_cols.unique()
    return df.loc[unique]


(college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
   .pipe(select_rows_with_max_cols)
)


# ### How it works...

# ### There's more...

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()

college_ugds.style.highlight_max(axis='columns')


# ## Replicating idxmax with method chaining

# ### How to do it...

def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


college_n = (college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
)


college_n.max().head()


college_n.eq(college_n.max()).head()


has_row_max = (college_n
    .eq(college_n.max())
    .any(axis='columns')
)
has_row_max.head()


college_n.shape


has_row_max.sum()


college_n.eq(college_n.max()).cumsum()


(college_n
    .eq(college_n.max())
    .cumsum()
    .cumsum()
)


has_row_max2 = (college_n
    .eq(college_n.max()) 
    .cumsum() 
    .cumsum() 
    .eq(1) 
    .any(axis='columns')
)


has_row_max2.head()


has_row_max2.sum()


idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols


set(college_n.idxmax().unique()) == set(idxmax_cols)


def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
     )
     return has_row_max[has_row_max].index


idx_max(college_n)


# ### How it works...

# ### There's more...

def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
         [lambda df_: df_]
         .index
     )
     return has_row_max

# %timeit college_n.idxmax().values


# %timeit idx_max(college_n)


# ## Finding the most common maximum of columns

# ### How to do it...

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()


highest_percentage_race.value_counts(normalize=True)


# ### How it works...

# ### There's more...

(college_ugds
    [highest_percentage_race == 'UGDS_BLACK']
    .drop(columns='UGDS_BLACK')
    .idxmax(axis='columns')
    .value_counts(normalize=True)
)


