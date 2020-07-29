import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('cs-training.csv',index_col=False)
df = df.fillna(0)
df = df.drop(['Unnamed: 0'], axis =1)
X = df.drop(['SeriousDlqin2yrs'], axis=1)
y = df['SeriousDlqin2yrs']
X_train, X_test2, y_train, y_test2 = train_test_split(X,y, test_size=0.20, random_state=42)
X_train.to_csv('X_train.csv',index=False)
X_test2.to_csv('X_test2.csv',index=False)
y_train.to_csv('y_train.csv',index=False)
y_test2.to_csv('y_test2.csv',index=False)