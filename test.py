import pandas as pd
columns = [['aaa','sss','sss','2.0'],['review', 'house', 'class', '4.0']]
df = pd.DataFrame(columns)
df.columns = ['a','b','c','d']
df['e'] = df['c'] +" "+ df['d']
print(df)