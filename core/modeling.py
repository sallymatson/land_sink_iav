<<<<<<< HEAD
import pandas as pd
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from matplotlib import pyplot as plt
from core.data.file_processing import open_all_data
from core.data.data_processing import trend
from math import sqrt


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


=======
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


>>>>>>> e1e688eaaa39f54df2ac2cb15a2edacbf7a3ef6d
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



class WindowGenerator():
    def __init__(self,
                 input_width,  # num of inputs each; i.e. timestep
                 label_width,  # num to predict
                 shift, # deplay between input & label,
                 train_df, val_df, test_df,
                 sequence_stride=1,
                 batch_size=10,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.sequence_stride = sequence_stride
        self.batch_size = batch_size

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride, # Period between successive output sequences. 
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)
        

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result  


def compile_and_fit(model, window, patience=20, batch_size=1, max_epochs=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True,
                                                      verbose=0)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val, verbose=0,
                        batch_size=batch_size,
                        callbacks=[early_stopping])
    return history


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def lstm_model():
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return lstm_model