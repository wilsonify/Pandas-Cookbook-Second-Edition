# ## Combining Pandas Objects

import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 9, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


# ## Introduction

# ## Appending new rows to DataFrames

# ### How to do it...

names = pd.read_csv('data/names.csv')
names


new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names


names.loc['five'] = ['Zach', 3]
names


names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
names


names.loc[len(names)] = pd.Series({'Age':32, 'Name':'Dean'})
names


names = pd.read_csv('data/names.csv')
names.append({'Name':'Aria', 'Age':1})


names.append({'Name':'Aria', 'Age':1}, ignore_index=True)


names.index = ['Canada', 'Canada', 'USA', 'USA']
names


s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s


names.append(s)


s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
names.append([s1, s2])


bball_16 = pd.read_csv('data/baseball16.csv')
bball_16


data_dict = bball_16.iloc[0].to_dict()
data_dict


new_data_dict = {k: '' if isinstance(v, str) else
    np.nan for k, v in data_dict.items()}
new_data_dict


# ### How it works...

# ### There's more...

random_data = []
for i in range(1000):   # doctest: +SKIP
    d = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))
random_data[0]


# ## Concatenating multiple DataFrames together

# ### How to do it...

stocks_2016 = pd.read_csv('data/stocks_2016.csv',
    index_col='Symbol')
stocks_2017 = pd.read_csv('data/stocks_2017.csv',
    index_col='Symbol')


stocks_2016


stocks_2017


s_list = [stocks_2016, stocks_2017]
pd.concat(s_list)


pd.concat(s_list, keys=['2016', '2017'],
   names=['Year', 'Symbol'])  


pd.concat(s_list, keys=['2016', '2017'],
    axis='columns', names=['Year', None])    


pd.concat(s_list, join='inner', keys=['2016', '2017'],
    axis='columns', names=['Year', None])


# ### How it works...

# ### There's more...

stocks_2016.append(stocks_2017)


# ## Understanding the differences between concat, join, and merge

# ### How to do it...

from IPython.display import display_html
years = 2016, 2017, 2018
stock_tables = [pd.read_csv(
    'data/stocks_{}.csv'.format(year), index_col='Symbol')
    for year in years]
stocks_2016, stocks_2017, stocks_2018 = stock_tables
stocks_2016


stocks_2017


stocks_2018


pd.concat(stock_tables, keys=[2016, 2017, 2018])


pd.concat(dict(zip(years, stock_tables)), axis='columns')


stocks_2016.join(stocks_2017, lsuffix='_2016',
    rsuffix='_2017', how='outer')


other = [stocks_2017.add_suffix('_2017'),
    stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')


stock_join = stocks_2016.add_suffix('_2016').join(other,
    how='outer')
stock_concat = pd.concat(dict(zip(years,stock_tables)),
    axis='columns')
level_1 = stock_concat.columns.get_level_values(1)
level_0 = stock_concat.columns.get_level_values(0).astype(str)
stock_concat.columns = level_1 + '_' + level_0
stock_join.equals(stock_concat)


stocks_2016.merge(stocks_2017, left_index=True,
    right_index=True)


step1 = stocks_2016.merge(stocks_2017, left_index=True,
    right_index=True, how='outer',
    suffixes=('_2016', '_2017'))
stock_merge = step1.merge(stocks_2018.add_suffix('_2018'),
    left_index=True, right_index=True,
    how='outer')
stock_concat.equals(stock_merge)


names = ['prices', 'transactions']
food_tables = [pd.read_csv('data/food_{}.csv'.format(name))
    for name in names]
food_prices, food_transactions = food_tables
food_prices


food_transactions


food_transactions.merge(food_prices, on=['item', 'store'])    


food_transactions.merge(food_prices.query('Date == 2017'),
    how='left')


food_prices_join = food_prices.query('Date == 2017') \
   .set_index(['item', 'store'])
food_prices_join    


food_transactions.join(food_prices_join, on=['item', 'store'])


pd.concat([food_transactions.set_index(['item', 'store']),
           food_prices.set_index(['item', 'store'])],
          axis='columns')


# ### How it works...

# ### There's more...

import glob
df_list = []
for filename in glob.glob('data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, index_col='Week',
    parse_dates=['Week']))
gas = pd.concat(df_list, axis='columns')
gas


# ## Connecting to SQL databases

# ### How to do it...

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')


tracks = pd.read_sql_table('tracks', engine)
tracks


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
)


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
     .groupby('Name')
     ['Milliseconds']
     .mean()
     .pipe(lambda s_: pd.to_timedelta(s_, unit='ms'))
     .dt.floor('s')
     .sort_values()
)


cust = pd.read_sql_table('customers', engine,
    columns=['CustomerId','FirstName',
    'LastName'])
invoice = pd.read_sql_table('invoices', engine,
    columns=['InvoiceId','CustomerId'])
ii = pd.read_sql_table('invoice_items', engine,
    columns=['InvoiceId', 'UnitPrice', 'Quantity'])
(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
)


(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
    .assign(Total=lambda df_:df_.Quantity * df_.UnitPrice)
    .groupby(['CustomerId', 'FirstName', 'LastName'])
    ['Total']
    .sum()
    .sort_values(ascending=False) 
)


# ### How it works...

# ### There's more...

sql_string1 = '''
SELECT
    Name,
    time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
FROM (
      SELECT
          g.Name,
          t.Milliseconds
      FROM
          genres as g
      JOIN
          tracks as t on
          g.genreid == t.genreid
     )
GROUP BY Name
ORDER BY avg_time'''
pd.read_sql_query(sql_string1, engine)


sql_string2 = '''
   SELECT
         c.customerid,
         c.FirstName,
         c.LastName,
         sum(ii.quantity * ii.unitprice) as Total
   FROM
        customers as c
   JOIN
        invoices as i
        on c.customerid = i.customerid
   JOIN
       invoice_items as ii
       on i.invoiceid = ii.invoiceid
   GROUP BY
       c.customerid, c.FirstName, c.LastName
   ORDER BY
       Total desc'''


pd.read_sql_query(sql_string2, engine)


