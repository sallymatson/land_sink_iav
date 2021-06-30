from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from matplotlib import pyplot as plt
from core.data.file_processing import open_all_data
from core.data.data_processing import trend
from math import sqrt
import pandas as pd

def data_pipeline():
	df = open_all_data()
	df_no_na = df.dropna()
	df_iav = trend(df_no_na)
	return df_iav


def prep_data(df, dependent_variable):
    
    # DEFINE TARGET USING WINDOW SHIFT
    df['y'] = df[dependent_variable].shift(-1)
    df = df.dropna()
    del df[dependent_variable]
    
    # SPLIT INTO TRAIN, VAL, AND TEST
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    print("There are {} in the training set, {} for val, and {} for testing."
          .format(len(train_df), len(val_df), len(test_df)))
    
    # NORMALIZE
    # The mean and standard deviation should only be computed using 
    # the training data so that the models have no access to the values 
    #     in the validation and test sets
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    # DEFINE TARGET USING WINDOW SHIFT
    train_y = train_df.pop('y')
    val_y = val_df.pop('y')
    test_y = test_df.pop('y')
    
    # RESHAPE
    # LSTM expects format of [samples, timestep, features]
    # Here, the shape is for a single-step model which predicts a single feature's value 
    # one timestep in the future based on current conditions.
    train_X = train_df.values.reshape((train_df.shape[0], 1, train_df.shape[1]))
    val_X = val_df.values.reshape((val_df.shape[0], 1, val_df.shape[1]))
    test_X = test_df.values.reshape((test_df.shape[0], 1, test_df.shape[1]))
    
    return train_X, val_X, test_X, train_y, val_y, test_y


def define_model_and_train(train_X, train_y, val_X, val_y):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='relu'))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=15, validation_data=(val_X, val_y), verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()
    return model


def evaluate_model(model, test_X, test_y):
	# todo: invert scaling for forecast
	# calculate RMSE
	yhat = model.predict(test_X)
	rmse = sqrt(mean_squared_error(test_y, yhat))
	print('Test RMSE: %.3f' % rmse)
	# plot the results
	results = pd.DataFrame(index=test_y.index)
	results['observed'] = test_y
	results['predicted'] = yhat
	results.plot()


def baseline(test_X, test_y):
	yhat = test_y.shift(1) # This returns the prediction as the previous value.
	yhat[0] = yhat[1]
	rmse = sqrt(mean_squared_error(test_y, yhat))
	print('Test RMSE: %.3f' % rmse)
	results = pd.DataFrame(index=test_y.index)
	results['observed'] = test_y
	results['predicted'] = yhat
	results.plot()


def basic_linear_model(train_X, train_y, test_X, test_y):
	reg = linear_model.LinearRegression()
	reg.fit(train_X.reshape(train_X.shape[0],train_X.shape[2]), train_y)
	yhat = reg.predict(test_X.reshape(test_X.shape[0],test_X.shape[2]))
	rmse = sqrt(mean_squared_error(test_y, yhat))
	print('Test RMSE: %.3f' % rmse)
	results = pd.DataFrame(index=test_y.index)
	results['observed'] = test_y
	results['predicted'] = yhat
	results.plot()

