import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('Data/insurance.csv')
encoder = preprocessing.OneHotEncoder()
scaler = preprocessing.StandardScaler()
encoder.fit(df[['region']])
scaler.fit(df[['age','bmi','children']])
scaledInput = scaler.transform(df[['age','bmi','children']])
one_hot = encoder.transform(df[['region']]).toarray()
df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
df[['age','bmi','children']] = scaledInput
smoker_values = {'no':0,'yes':1}
sex_values = {'male':0,'female':1}
df['smoker'] = df['smoker'].map(smoker_values)
df['sex'] = df['sex'].map(sex_values)
df.to_csv('Data/insurance_preprocessed.csv',index=False)
