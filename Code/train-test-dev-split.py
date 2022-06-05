import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./complete.csv', sep = '\t')

# print(df)
# print(df.shape)

X_train, X_test = train_test_split(df, test_size=0.2)

X_train = X_train.sort_values(by=['Context'])
X_test = X_test.sort_values(by=['Context'])
# print(X_test)
# print(X_train)
# print(X_train.shape)

# print(X_test.shape)

X_train.to_csv('./bidaf-keras-bert/final-dataset/train.csv', index = False, sep = '\t')

X_train, X_test = train_test_split(X_test, test_size=0.25)

X_train = X_train.sort_values(by=['Context'])
X_test = X_test.sort_values(by=['Context'])

X_train.to_csv('./bidaf-keras-bert/final-dataset/dev.csv', index = False, sep = '\t')
X_test.to_csv('./bidaf-keras-bert/final-dataset/test.csv', index = False, sep = '\t')

print("Splitting complete")