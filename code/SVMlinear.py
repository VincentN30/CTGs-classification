import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

train_file_path = 'train.xlsx'
test_file_path = 'test.xlsx'

train_df = pd.read_excel(train_file_path)
test_df = pd.read_excel(test_file_path)


train_df.columns = train_df.iloc[0]
train_df = train_df[1:]
test_df.columns = test_df.iloc[0]
test_df = test_df[1:]


train_df = train_df.iloc[:, 10:]
test_df = test_df.iloc[:, 10:]

train_df = train_df.iloc[:, :-1]
test_df = test_df.iloc[:,:-1]

X_train = train_df.drop('NSP', axis=1) 
X_train = X_train.astype(float)
y_train = train_df['NSP'].astype(int)  

X_test = test_df.drop('NSP', axis=1) 
X_test = X_test.astype(float)
y_test = test_df['NSP'].astype(int)  


scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

SVMl = SVC(kernel='linear', random_state=42)
SVMl.fit(X_train,y_train)

train_y_perd = SVMl.predict(X_train)
report = classification_report(y_train, train_y_perd,digits=4)
macro_auc = roc_auc_score(pd.get_dummies(y_train), pd.get_dummies(train_y_perd), multi_class='ovr', average='macro')

y_pred = SVMl.predict(X_test)
rep = classification_report(y_test,y_pred,digits=4)
auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), multi_class='ovr', average='macro')

print(report)
print(macro_auc)

print(rep)
print(auc)