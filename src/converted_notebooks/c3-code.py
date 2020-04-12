# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ### How to do it\...

fname = ['Paul', 'John', 'Richard', 'George']
lname = ['McCartney', 'Lennon', 'Starkey', 'Harrison']
birth = [1942, 1940, 1940, 1943]


people = {'first': fname, 'last': lname, 'birth': birth}


beatles = pd.DataFrame(people)
beatles


# ### How it works\...

beatles.index


pd.DataFrame(people, index=['a', 'b', 'c', 'd'])


# ### There\'s More

pd.DataFrame(
[{"first":"Paul","last":"McCartney", "birth":1942},
 {"first":"John","last":"Lennon", "birth":1940},
 {"first":"Richard","last":"Starkey", "birth":1940},
 {"first":"George","last":"Harrison", "birth":1943}])


[{"first":"Paul","last":"McCartney", "birth":1942},
 {"first":"John","last":"Lennon", "birth":1940},
 {"first":"Richard","last":"Starkey", "birth":1940},
 {"first":"George","last":"Harrison", "birth":1943}],
 columns=['last', 'first', 'birth'])


# ### How to do it\...

beatles


from io import StringIO
fout = StringIO()
beatles.to_csv(fout)  # use a filename instead of fout


print(fout.getvalue())


# ### There\'s More

_ = fout.seek(0)
pd.read_csv(fout)


_ = fout.seek(0)
pd.read_csv(fout, index_col=0)


fout = StringIO()
beatles.to_csv(fout, index=False) 
print(fout.getvalue())


# ### How to do it\...

diamonds = pd.read_csv('data/diamonds.csv', nrows=1000)
diamonds


diamonds.info()


diamonds2 = pd.read_csv('data/diamonds.csv', nrows=1000,
    dtype={'carat': np.float32, 'depth': np.float32,
           'table': np.float32, 'x': np.float32,
           'y': np.float32, 'z': np.float32,
           'price': np.int16})


diamonds2.info()


diamonds.describe()


diamonds2.describe()


diamonds2.cut.value_counts()


diamonds2.color.value_counts()


diamonds2.clarity.value_counts()


diamonds3 = pd.read_csv('data/diamonds.csv', nrows=1000,
    dtype={'carat': np.float32, 'depth': np.float32,
           'table': np.float32, 'x': np.float32,
           'y': np.float32, 'z': np.float32,
           'price': np.int16,
           'cut': 'category', 'color': 'category',
           'clarity': 'category'})


diamonds3.info()


np.iinfo(np.int8)


np.finfo(np.float16)


cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']
diamonds4 = pd.read_csv('data/diamonds.csv', nrows=1000,
    dtype={'carat': np.float32, 'depth': np.float32,
           'table': np.float32, 'price': np.int16,
           'cut': 'category', 'color': 'category',
           'clarity': 'category'},
    usecols=cols)


diamonds4.info()


cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']
diamonds_iter = pd.read_csv('data/diamonds.csv', nrows=1000,
    dtype={'carat': np.float32, 'depth': np.float32,
           'table': np.float32, 'price': np.int16,
           'cut': 'category', 'color': 'category',
           'clarity': 'category'},
    usecols=cols,
    chunksize=200)


def process(df):
    return f'processed {df.size} items'


for chunk in diamonds_iter:
    process(chunk)


# ### How it works\...

# ### There\'s more \...

diamonds.price.memory_usage()


diamonds.price.memory_usage(index=False)


diamonds.cut.memory_usage()


diamonds.cut.memory_usage(deep=True)


diamonds4.to_feather('/tmp/d.arr')
diamonds5 = pd.read_feather('/tmp/d.arr')


diamonds4.to_parquet('/tmp/d.pqt')


# ### How to do it\...

beatles.to_excel('/tmp/beat.xls')


beatles.to_excel('/tmp/beat.xlsx')


beat2 = pd.read_excel('/tmp/beat.xls')
beat2


beat2 = pd.read_excel('/tmp/beat.xls', index_col=0)
beat2


beat2.dtypes


# ### How it works\...

# ### There\'s more\...

xl_writer = pd.ExcelWriter('/tmp/beat.xlsx')
beatles.to_excel(xl_writer, sheet_name='All')
beatles[beatles.birth < 1941].to_excel(xl_writer, sheet_name='1940')
xl_writer.save()


# ### How to do it\...

autos = pd.read_csv('data/vehicles.csv.zip')
autos


autos.modifiedOn.dtype


autos.modifiedOn


pd.to_datetime(autos.modifiedOn)  # doctest: +SKIP


autos = pd.read_csv('data/vehicles.csv.zip',
    parse_dates=['modifiedOn'])  # doctest: +SKIP
autos.modifiedOn


import zipfile


with zipfile.ZipFile('data/kaggle-survey-2018.zip') as z:
    print('\n'.join(z.namelist()))
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    kag_questions = kag.iloc[0]
    survey = kag.iloc[1:]


print(survey.head(2).T)


# ### How it works\...

# ### There\'s more\...

# ### How to do it\...

import sqlite3
con = sqlite3.connect('data/beat.db')
with con:
    cur = con.cursor()
    cur.execute("""DROP TABLE Band""")
    cur.execute("""CREATE TABLE Band(id INTEGER PRIMARY KEY,
        fname TEXT, lname TEXT, birthyear INT)""")
    cur.execute("""INSERT INTO Band VALUES(
        0, 'Paul', 'McCartney', 1942)""")
    cur.execute("""INSERT INTO Band VALUES(
        1, 'John', 'Lennon', 1940)""")
    _ = con.commit()


import sqlalchemy as sa
engine = sa.create_engine(
  'sqlite:///data/beat.db', echo=True)
sa_connection = engine.connect()


beat = pd.read_sql('Band', sa_connection, index_col='id')
beat


sql = '''SELECT fname, birthyear from Band'''
fnames = pd.read_sql(sql, con)
fnames


# ### How it work\'s\...

import json
encoded = json.dumps(people)
encoded


json.loads(encoded)


# ### How to do it\...

beatles = pd.read_json(encoded)
beatles


records = beatles.to_json(orient='records')
records


pd.read_json(records, orient='records')


split = beatles.to_json(orient='split')
split


pd.read_json(split, orient='split')


index = beatles.to_json(orient='index')
index


pd.read_json(index, orient='index')


values = beatles.to_json(orient='values')
values


pd.read_json(values, orient='values')


(pd.read_json(values, orient='values')
   .rename(columns=dict(enumerate(['first', 'last', 'birth'])))
)


table = beatles.to_json(orient='table')
table


pd.read_json(table, orient='table')


# ### How it works\...

# ### There\'s more\...

output = beat.to_dict()
output


output['version'] = '0.4.1'
json.dumps(output)


# ### How to do it\...

url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url)
len(dfs)


dfs[0]


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—')
len(dfs)


dfs[0].columns


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—',
    header=[0,1])
len(dfs)


dfs[0]


dfs[0].columns


df = dfs[0]
df.columns = ['Title', 'Release', 'UK', 'AUS', 'CAN', 'FRA', 'GER',
    'NOR', 'US', 'Certifications']
df


res = (df
  .pipe(lambda df_: df_[~df_.Title.str.startswith('Released')])
  .iloc[:-1]
  .assign(release_date=lambda df_: pd.to_datetime(
             df_.Release.str.extract(r'Released: (.*) Label')
               [0]
               .str.replace(r'\[E\]', '')
          ),
          label=lambda df_:df_.Release.str.extract(r'Label: (.*)')
         )
   .loc[:, ['Title', 'UK', 'AUS', 'CAN', 'FRA', 'GER', 'NOR',
            'US', 'release_date', 'label']]
)
res


# ### How it works\...

# ### There is more\...

url = 'https://github.com/mattharrison/datasets/blob/master/data/anscombes.csv'
dfs = pd.read_html(url, attrs={'class': 'csv-data'})
len(dfs)


dfs[0]


