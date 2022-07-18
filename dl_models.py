import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D 
from keras.layers import SimpleRNN, GRU, LSTM
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

trainX = np.load("trainX.npy")
testT = np.load("testT.npy")
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy') 

dnn_X_train = np.array(trainX)
dnn_X_test = np.array(testT)


# DNN
dnn_model = Sequential()
dnn_model.add(Dense(1024,input_dim=41,activation='relu'))  
dnn_model.add(Dropout(0.01))
dnn_model.add(Dense(768,activation='relu'))  
dnn_model.add(Dropout(0.01))
dnn_model.add(Dense(512,activation='relu'))  
dnn_model.add(Dropout(0.01))
dnn_model.add(Dense(256,activation='relu'))  
dnn_model.add(Dropout(0.01))
dnn_model.add(Dense(128,activation='relu'))  
dnn_model.add(Dropout(0.01))
dnn_model.add(Dense(5, activation='softmax'))

dnn_model.summary()

dnn_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

dnn_history = dnn_model.fit(dnn_X_train, y_train, validation_data=(dnn_X_test, y_test),batch_size=64, epochs=200, verbose = 0)

dnn_acc = dnn_history.history['accuracy']
dnn_val_acc = dnn_history.history['val_accuracy']
dnn_loss = dnn_history.history['loss']
dnn_val_loss = dnn_history.history['val_loss']
epochs = range(len(dnn_acc))

plt.plot(epochs, dnn_acc, 'r', label='Training accuracy')
plt.plot(epochs, dnn_val_acc, 'b', label='Validation accuracy')
plt.title('DNN Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()


plt.plot(epochs, dnn_loss, 'r', label='Training Loss')
plt.plot(epochs, dnn_val_loss, 'b', label='Validation Loss')
plt.title('DNN Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

dnn_pred = dnn_model.predict(dnn_X_test)
dnn_pred = np.argmax(dnn_pred, axis=1)
dnn_y_test = np.argmax(y_test, axis=1)
print(classification_report(dnn_y_test,dnn_pred, zero_division=0))



# CNN
# reshape input to be [samples, time steps, features]
cnn_X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
cnn_X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

cnn_model = Sequential()
cnn_model.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
cnn_model.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn_model.add(MaxPooling1D(2))
cnn_model.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_model.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_model.add(MaxPooling1D(2))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(5, activation="softmax"))

cnn_model.summary()

cnn_model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

cnn_history = cnn_model.fit(cnn_X_train, y_train, epochs=200, batch_size=64, validation_data=(cnn_X_test, y_test), verbose=0)

cnn_acc = cnn_history.history['accuracy']
cnn_val_acc = cnn_history.history['val_accuracy']
cnn_loss = cnn_history.history['loss']
cnn_val_loss = cnn_history.history['val_loss']
epochs = range(len(cnn_acc))

plt.plot(epochs, cnn_acc, 'r', label='Training accuracy')
plt.plot(epochs, cnn_val_acc, 'b', label='Validation accuracy')
plt.title('CNN Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

plt.plot(epochs, cnn_loss, 'r', label='Training Loss')
plt.plot(epochs, cnn_val_loss, 'b', label='Validation Loss')
plt.title('CNN Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

cnn_pred = cnn_model.predict(cnn_X_test)
cnn_pred = np.argmax(cnn_pred, axis=1)
cnn_y_test = np.argmax(y_test, axis=1)
print(classification_report(cnn_y_test,cnn_pred, zero_division=0))



# RNN
# reshape input to be [samples, time steps, features]
rnn_X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
rnn_X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

rnn_model = Sequential()
rnn_model.add(SimpleRNN(64,input_dim=41, return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(64,return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(64, return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(64, return_sequences=False))
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(5, activation="softmax"))

rnn_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

rnn_history = rnn_model.fit(rnn_X_train, y_train, batch_size=64, epochs=200, validation_data=(rnn_X_test, y_test), verbose = 0)
rnn_acc = rnn_history.history['accuracy']
rnn_val_acc = rnn_history.history['val_accuracy']
rnn_loss = rnn_history.history['loss']
rnn_val_loss = rnn_history.history['val_loss']
epochs = range(len(rnn_acc))

plt.plot(epochs, rnn_acc, 'r', label='Training accuracy')
plt.plot(epochs, rnn_val_acc, 'b', label='Validation accuracy')
plt.title('RNN Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

plt.plot(epochs, rnn_loss, 'r', label='Training Loss')
plt.plot(epochs, rnn_val_loss, 'b', label='Validation Loss')
plt.title('RNN Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

rnn_pred = rnn_model.predict(rnn_X_test)
rnn_pred = np.argmax(rnn_pred, axis=1)
rnn_y_test = np.argmax(y_test, axis=1)
print(classification_report(rnn_y_test,rnn_pred, zero_division=0))



# GRU
# reshape input to be [samples, time steps, features]
gru_X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
gru_X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

gru_model = Sequential()
gru_model.add(GRU(64,input_dim=41, return_sequences=True))  
gru_model.add(Dropout(0.1))
gru_model.add(GRU(64,return_sequences=True))  
gru_model.add(Dropout(0.1))
gru_model.add(GRU(64, return_sequences=True))  
gru_model.add(Dropout(0.1))
gru_model.add(GRU(64, return_sequences=False))  
gru_model.add(Dropout(0.1))
gru_model.add(Dense(5,activation='softmax'))

gru_model.summary()

gru_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

gru_history = gru_model.fit(gru_X_train, y_train, batch_size=64, epochs=200, validation_data=(gru_X_test, y_test), verbose = 0)
gru_acc = gru_history.history['accuracy']
gru_val_acc = gru_history.history['val_accuracy']
gru_loss = gru_history.history['loss']
gru_val_loss = gru_history.history['val_loss']
epochs = range(len(gru_acc))

plt.plot(epochs, gru_acc, 'r', label='Training accuracy')
plt.plot(epochs, gru_val_acc, 'b', label='Validation accuracy')
plt.title('GRU Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

plt.plot(epochs, gru_loss, 'r', label='Training Loss')
plt.plot(epochs, gru_val_loss, 'b', label='Validation Loss')
plt.title('GRU Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

gru_pred = gru_model.predict(gru_X_test)
gru_pred = np.argmax(gru_pred, axis=1)
gru_y_test = np.argmax(y_test, axis=1)
print(classification_report(gru_y_test,gru_pred, zero_division=0))



# LSTM
# reshape input to be [samples, time steps, features]
lstm_X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
lstm_X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(64,input_dim=41, return_sequences=True)) 
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(64,return_sequences=True)) 
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(64, return_sequences=True))  
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(64, return_sequences=False)) 
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(5,activation='softmax'))

lstm_model.summary()

lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

lstm_history = lstm_model.fit(lstm_X_train, y_train, batch_size=64, epochs=200, validation_data=(lstm_X_test, y_test), verbose = 0)
lstm_acc = lstm_history.history['accuracy']
lstm_val_acc = lstm_history.history['val_accuracy']
lstm_loss = lstm_history.history['loss']
lstm_val_loss = lstm_history.history['val_loss']
epochs = range(len(lstm_acc))

plt.plot(epochs, lstm_acc, 'r', label='Training accuracy')
plt.plot(epochs, lstm_val_acc, 'b', label='Validation accuracy')
plt.title('LSTM Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

plt.plot(epochs, lstm_loss, 'r', label='Training Loss')
plt.plot(epochs, lstm_val_loss, 'b', label='Validation Loss')
plt.title('LSTM Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

lstm_pred = lstm_model.predict(lstm_X_test)
lstm_pred = np.argmax(lstm_pred, axis=1)
lstm_y_test = np.argmax(y_test, axis=1)
print(classification_report(lstm_y_test,lstm_pred, zero_division=0))


# CNN-LSTM
# reshape input to be [samples, time steps, features]
cnn_lstm_X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
cnn_lstm_X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

cnn_lstm_model = Sequential()
cnn_lstm_model.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
cnn_lstm_model.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn_lstm_model.add(MaxPooling1D(2))
cnn_lstm_model.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_lstm_model.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn_lstm_model.add(MaxPooling1D(2))
cnn_lstm_model.add(LSTM(70))
cnn_lstm_model.add(Dropout(0.1))
cnn_lstm_model.add(Dense(5, activation="softmax"))

cnn_lstm_model.summary()

cnn_lstm_model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

cnn_lstm_history = cnn_lstm_model.fit(cnn_lstm_X_train, y_train, batch_size=64, epochs=200, validation_data=(cnn_lstm_X_test, y_test), verbose = 0)

cnn_lstm_acc = cnn_lstm_history.history['accuracy']
cnn_lstm_val_acc = cnn_lstm_history.history['val_accuracy']
cnn_lstm_loss = cnn_lstm_history.history['loss']
cnn_lstm_val_loss = cnn_lstm_history.history['val_loss']
epochs = range(len(cnn_lstm_acc))

plt.plot(epochs, cnn_lstm_acc, 'r', label='Training accuracy')
plt.plot(epochs, cnn_lstm_val_acc, 'b', label='Validation accuracy')
plt.title('CNN LSTM Training and Validation accuracy')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

plt.plot(epochs, cnn_lstm_loss, 'r', label='Training Loss')
plt.plot(epochs, cnn_lstm_val_loss, 'b', label='Validation Loss')
plt.title('CNN LSTM Training and Validation loss')
plt.xlabel("Number of Epochs")
plt.legend()
plt.show()

cnn_lstm_pred = cnn_lstm_model.predict(cnn_lstm_X_test)
cnn_lstm_pred = np.argmax(cnn_lstm_pred, axis=1)
cnn_lstm_y_test = np.argmax(y_test, axis=1)
print(classification_report(cnn_lstm_y_test,cnn_lstm_pred, zero_division=0))

