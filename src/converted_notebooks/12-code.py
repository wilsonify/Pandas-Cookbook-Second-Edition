# # Time Series Analysis

import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 12, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


pd.set_option(#'max_columns', 4,
    'max_rows', 10)
from io import StringIO
def txt_repr(df, width=40, rows=None):
    buf = StringIO()
    rows = rows if rows is not None else pd.options.display.max_rows
    num_cols = len(df.columns)
    with pd.option_context('display.width', 100):
        df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
        out = buf.getvalue()
        for line in out.split('\n'):
            if len(line) > width or line.strip().endswith('\\'):
                break
        else:
            return out
        done = False
        while not done:
            buf = StringIO()
            df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
            for line in buf.getvalue().split('\n'):
                if line.strip().endswith('\\'):
                    num_cols = min([num_cols - 1, int(num_cols*.8)])
                    break
            else:
                break
        return buf.getvalue()
pd.DataFrame.__repr__ = lambda self, *args: txt_repr(self, 65, 10)


# ## Introduction

# ## Understanding the difference between Python and pandas date tools

# ### How to do it...

import datetime
date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30,
    second=19, microsecond=463198)
dt = datetime.datetime(year=2013, month=6, day=7,
    hour=12, minute=30, second=19,
    microsecond=463198)
print(f"date is {date}")


print(f"time is {time}")


print(f"datetime is {dt}")


td = datetime.timedelta(weeks=2, days=5, hours=10,
    minutes=20, seconds=6.73,
    milliseconds=99, microseconds=8)
td


print(f'new date is {date+td}')


print(f'new datetime is {dt+td}')


time + td


pd.Timestamp(year=2012, month=12, day=21, hour=5,
   minute=10, second=8, microsecond=99)


pd.Timestamp('2016/1/10')


pd.Timestamp('2014-5/10')


pd.Timestamp('Jan 3, 2019 20:45.56')


pd.Timestamp('2016-01-05T05:34:43.123456789')


pd.Timestamp(500)


pd.Timestamp(5000, unit='D')


pd.to_datetime('2015-5-13')


pd.to_datetime('2015-13-5', dayfirst=True)


pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm',
    format='Start Date: %b %d, %Y Start Time: %I:%M %p')


pd.to_datetime(100, unit='D', origin='2013-1-1')


s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')


s = pd.Series(['12-5-2015', '14-1-2013',
   '20/12/2017', '40/23/2017'])


pd.to_datetime(s, dayfirst=True, errors='coerce')


pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])


pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')


pd.Timedelta(days=5, minutes=7.34)


pd.Timedelta(100, unit='W')


pd.to_timedelta('67:15:45.454')


s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')


time_strings = ['2 days 24 minutes 89.67 seconds',
    '00:45:23.6']
pd.to_timedelta(time_strings)


pd.Timedelta('12 days 5 hours 3 minutes') * 2


(pd.Timestamp('1/1/2017') + 
   pd.Timedelta('12 days 5 hours 3 minutes') * 2)


td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2


pd.Timedelta('12 days') / pd.Timedelta('3 days')


ts = pd.Timestamp('2016-10-1 4:23:23.9')
ts.ceil('h')


ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second


ts.dayofweek, ts.dayofyear, ts.daysinmonth


ts.to_pydatetime()


td = pd.Timedelta(125.8723, unit='h')
td


td.round('min')


td.components


td.total_seconds()


# ### How it works...

# ### There's more...

date_string_list = ['Sep 30 1984'] * 10000
# %timeit pd.to_datetime(date_string_list, format='%b %d %Y')


# %timeit pd.to_datetime(date_string_list)


# ## Slicing time series intelligently

# ### How to do it...

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


crime = crime.set_index('REPORTED_DATE')
crime


crime.loc['2016-05-12 16:45:00']


crime.loc['2016-05-12']


crime.loc['2016-05'].shape


crime.loc['2016'].shape


crime.loc['2016-05-12 03'].shape


crime.loc['Dec 2015'].sort_index()


crime.loc['2016 Sep, 15'].shape


crime.loc['21st October 2014 05'].shape


crime.loc['2015-3-4':'2016-1-1'].sort_index()


crime.loc['2015-3-4 22':'2016-1-1 11:22:00'].sort_index()


# ### How it works...

mem_cat = crime.memory_usage().sum()
mem_obj = (crime
   .astype({'OFFENSE_TYPE_ID':'object',
            'OFFENSE_CATEGORY_ID':'object',
           'NEIGHBORHOOD_ID':'object'}) 
   .memory_usage(deep=True)
   .sum()
)
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)


crime.index[:2]


# ### There's more...

# %timeit crime.loc['2015-3-4':'2016-1-1']


crime_sort = crime.sort_index()
# %timeit crime_sort.loc['2015-3-4':'2016-1-1']


# ## Filtering columns with time data

# ### How to do it... 

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


(crime
    [crime.REPORTED_DATE == '2016-05-12 16:45:00']
)


(crime
    [crime.REPORTED_DATE == '2016-05-12']
)


(crime
    [crime.REPORTED_DATE.dt.date == '2016-05-12']
)


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12', '2016-05-13')]
)


(crime
    [crime.REPORTED_DATE.between(
         '2016-05', '2016-06')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '2016', '2017')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12 03', '2016-05-12 04')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '2016 Sep, 15', '2016 Sep, 16')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '21st October 2014 05', '21st October 2014 06')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 23:59:59')]
    .shape
)


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 11:22:00')]
    .shape
)


# ### How it works...

lmask = crime.REPORTED_DATE >= '2015-3-4 22'
rmask = crime.REPORTED_DATE <= '2016-1-1 11:22:00'
crime[lmask & rmask].shape


# ### There's more...

ctseries = crime.set_index('REPORTED_DATE')
# %timeit ctseries.loc['2015-3-4':'2016-1-1']


# %timeit crime[crime.REPORTED_DATE.between('2015-3-4','2016-1-1')]


# ## Using methods that only work with a DatetimeIndex

# ### How to do it...

crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE')
)
type(crime.index)


crime.between_time('2:00', '5:00', include_end=False)


crime.at_time('5:47')


crime_sort = crime.sort_index()
crime_sort.first(pd.offsets.MonthBegin(6))


crime_sort.first(pd.offsets.MonthEnd(6))


crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))


crime_sort.loc[:'2012-06']


crime_sort.first('5D') # 5 days


crime_sort.first('5B') # 5 business days


crime_sort.first('7W') # 7 weeks, with weeks ending on Sunday


crime_sort.first('3QS') # 3rd quarter start


crime_sort.first('A') # one year end


# ### How it works...

import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0),
                   include_end=False)


first_date = crime_sort.index[0]
first_date


first_date + pd.offsets.MonthBegin(6)


first_date + pd.offsets.MonthEnd(6)


step4 = crime_sort.first(pd.offsets.MonthEnd(6))
end_dt = crime_sort.index[0] + pd.offsets.MonthEnd(6)
step4_internal = crime_sort[:end_dt]
step4.equals(step4_internal)


# ### There's more...

dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)


do = pd.DateOffset(years=2, months=5, days=3,
    hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do


# ## Counting the number of weekly crimes

# ### How to do it...

crime_sort = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


crime_sort.resample('W')


(crime_sort
    .resample('W')
    .size()
)


len(crime_sort.loc[:'2012-1-8'])


len(crime_sort.loc['2012-1-9':'2012-1-15'])


(crime_sort
    .resample('W-THU')
    .size()
)


weekly_crimes = (crime_sort
    .groupby(pd.Grouper(freq='W')) 
    .size()
)
weekly_crimes


# ### How it works...

r = crime_sort.resample('W')
[attr for attr in dir(r) if attr[0].islower()]


# ### There's more...

crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)


weekly_crimes_gby2 = (crime
    .groupby(pd.Grouper(key='REPORTED_DATE', freq='W'))
    .size()
)
weekly_crimes2.equals(weekly_crimes)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 4))
weekly_crimes.plot(title='All Denver Crimes', ax=ax)
fig.savefig('/tmp/c12-crimes.png', dpi=300)


# ## Aggregating weekly crime and traffic accidents separately

# ### How to do it...

crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


(crime
    .resample('QS')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


(crime
   .loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']]
   .sum()
)


(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .plot(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents')
)
fig.savefig('/tmp/c12-crimes2.png', dpi=300)


# ### How it works...

(crime
    .resample('Q')
    .sum()
)


(crime_sort
    .resample('QS-MAR')
    ['IS_CRIME', 'IS_TRAFFIC'] 
    .sum()
)


# ### There's more...

crime_begin = (crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .iloc[0]
)


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .div(crime_begin)
    .sub(1)
    .round(2)
    .mul(100)
    .plot.bar(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents % Increase')
)


fig.autofmt_xdate()
fig.savefig('/tmp/c12-crimes3.png', dpi=300, bbox_inches='tight')


# ## Measuring crime by weekday and year

# ### How to do it...

crime = pd.read_hdf('data/crime.h5', 'crime')
crime


(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
)


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .reindex(days)
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes4.png', dpi=300, bbox_inches='tight')                 


title = 'Denver Crimes and Traffic Accidents per Year'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.year 
   .value_counts()
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes5.png', dpi=300, bbox_inches='tight')                 


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
)


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
)


criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()


round(272 / 365, 3)


crime_pct = (crime
   ['REPORTED_DATE']
   .dt.dayofyear.le(272) 
   .groupby(crime.REPORTED_DATE.dt.year) 
   .mean()
   .round(3)
)


crime_pct


crime_pct.loc[2012:2016].median()


def update_2017(df_):
    df_.loc[2017] = (df_
        .loc[2017]
        .div(.748) 
        .astype('int')
    )
    return df_
(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
table = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)
sns.heatmap(table, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes6.png', dpi=300, bbox_inches='tight')                 


denver_pop = pd.read_csv('data/denver_pop.csv',
    index_col='Year')
denver_pop


den_100k = denver_pop.div(100_000).squeeze()
normalized = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(normalized, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes7.png', dpi=300, bbox_inches='tight')                 


# ### How it works...

(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .loc[days]
)


(crime
    .assign(year=crime.REPORTED_DATE.dt.year,
            day=crime.REPORTED_DATE.dt.weekday_name)
    .pipe(lambda df_: pd.crosstab(df_.year, df_.day))
)


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
) / den_100k


# ### There's more...

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
crime_type = 'auto-theft'
normalized = (crime
    .query('OFFENSE_CATEGORY_ID == @crime_type')
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# ## Grouping with anonymous functions with a DatetimeIndex

# ### How to do it...

crime = (pd.read_hdf('data/crime.h5', 'crime') 
   .set_index('REPORTED_DATE') 
   .sort_index()
)


common_attrs = (set(dir(crime.index)) & 
    set(dir(pd.Timestamp)))
[attr for attr in common_attrs if attr[0] != '_']


crime.index.weekday_name.value_counts()


(crime
   .groupby(lambda idx: idx.weekday_name) 
   ['IS_CRIME', 'IS_TRAFFIC']
   .sum()    
)


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
)


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
    .style.highlight_max(color='lightgrey')
)


# ### How it works...

# ## Grouping by a Timestamp and another column

# ### How to do it...

employee = pd.read_csv('data/employee.csv',
    parse_dates=['JOB_DATE', 'HIRE_DATE'],
    index_col='HIRE_DATE')
employee


(employee
    .groupby('GENDER')
    ['BASE_SALARY']
    .mean()
    .round(-2)
)


(employee
    .resample('10AS')
    ['BASE_SALARY']
    .mean()
    .round(-2)    
)


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
)


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
   .unstack('GENDER')
)


employee[employee['GENDER'] == 'Male'].index.min()


employee[employee['GENDER'] == 'Female'].index.min()


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
)


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# ### How it works...

# ### There's more...

sal_final = (employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)
years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final


cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values


(employee
    .groupby([cuts, 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .unstack('GENDER')
    .round(-2)
)


