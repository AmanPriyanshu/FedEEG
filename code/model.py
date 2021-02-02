from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization

def get_model():
	model = Sequential([
		LSTM(16,input_shape=(11,11), return_sequences=True),
		LSTM(16),
		BatchNormalization(),
		Dense(3, activation='softmax')
		])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model