import pandas as pd




import pymysql
from sqlalchemy import create_engine
from datetime import datetime
engine = create_engine("mysql://root:123456@localhost/rumor")

df = pd.read_csv('datafile/all_data.txt',sep='\t',names=['lable','text'])
df.index.name = 'id'
df['type'] = df['lable'].apply(lambda x: '谣言' if x == 0 else '正常')
df['date'] = datetime.now()
df.to_sql('web_dataset',engine,if_exists='append')
# df1 = pd.read_csv('stopwords.txt',names=['name'])
# df1.index.name = 'id'
# df1.to_sql('web_stopword',engine,if_exists='append')