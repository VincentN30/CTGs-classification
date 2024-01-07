import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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

y_train_one_hot = to_categorical(y_train, num_classes=3)
y_test_one_hot = to_categorical(y_test, num_classes=3)

ANN = Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  
])

custom_optimizer = Adam(learning_rate=0.001) 
# 编译模型
ANN.compile(optimizer=custom_optimizer,  # 优化器
              loss='categorical_crossentropy',  # 损失函数，适用于多类别分类
              metrics=['accuracy'])  # 评估指标

ANN.summary()

#提前停下，防止过拟合
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = ANN.fit(X_train, y_train_one_hot, epochs=100, batch_size=64, validation_split=0.2,callbacks=[early_stopping])

train_y_perd = ANN.predict(X_train)
train_y_perd = tf.argmax(train_y_perd, axis=1)  # 将概率转换为类别
report = classification_report(y_train, train_y_perd,digits=4)
macro_auc = roc_auc_score(pd.get_dummies(y_train), pd.get_dummies(train_y_perd), multi_class='ovr', average='macro')

y_pred = ANN.predict(X_test)
y_pred = tf.argmax(y_pred, axis=1)  # 将概率转换为类别
rep = classification_report(y_test,y_pred,digits=4)
auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), multi_class='ovr', average='macro')

print(report)
print(macro_auc)

print(rep)
print(auc)