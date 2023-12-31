# # https://finance.yahoo.com/
# # https://www.youtube.com/watch?v=s3CnE2tqQdo
# # go to cmd of the desried file and then type streamlit run app.py

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas_datareader as data
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# from sklearn.preprocessing import MinMaxScaler 

# start = '2010-01-01'
# end = '2023-08-01'

# st.title('Stock Trend Prediction')

# user_input = st.text_input('Enter Stock Ticker', 'AAPL')
# df = yf.download(user_input ,start=start, end=end)

# # DESCRIBING DATA
# st.subheader('Data from 2010-2019')
# st.write(df.describe())

# # Visualization
# st.subheader('Closing Price vs Time Chart')
# fig = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'])
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA')

# ma100 = df['Close'].rolling(window=100).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, label='100 Moving Average')
# plt.plot(df['Close'], label='Closing Price')
# plt.legend()
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100 & 200MA')
# ma200 = df['Close'].rolling(window=200).mean()
# ma100 = df['Close'].rolling(window=100).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, label='100 Moving Average')
# plt.plot(ma200, label='200 Moving Average')
# plt.plot(df['Close'], label='Closing Price')
# st.pyplot(fig)

# # Splitting data into training and testing
# data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
# data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

# scaler = MinMaxScaler(feature_range=(0,1 ))
# data_training_array = scaler.fit_transform(data_training)

# # x_train = []
# # y_train = []
# # #we are predicting value by seeing stock price of last 100 days
# # for i in range(100, data_training_array.shape[0]):
# #     x_train.append(data_training_array[i-100:i])
# #     y_train.append(data_training_array[i,0])
                   
# # x_train,y_train =np.array(x_train),np.array(y_train)

# model = load_model('stock_trend_model.keras')  # Load the Keras model
# #define 2 empty list

# past_100_days = data_training.tail(100)

# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# input_data = scaler.fit_transform(final_df)

# x_test =[]
# y_test =[]

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i,0])

# x_test, y_test = np.array(x_test),np.array(y_test)
# y_predicted = model.predict(x_test)

# scale_factor = 1/scaler.scale_[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor

# # Display predictions vs. original
# st.subheader('Predictions vs Original')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_predicted, 'r', label='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler 

start = '2010-01-01'
end = '2023-08-01'

st.title('Financial Chatbot')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input ,start=start, end=end)

# DESCRIBING DATA
st.subheader('Data from 2010-2023')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')

ma100 = df['Close'].rolling(window=100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100 Moving Average')
plt.plot(df['Close'], label='Closing Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200MA')
ma200 = df['Close'].rolling(window=200).mean()
ma100 = df['Close'].rolling(window=100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100 Moving Average')
plt.plot(ma200, label='200 Moving Average')
plt.plot(df['Close'], label='Closing Price')
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('stock_trend_model.keras')  # Load the Keras model

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Display predictions vs. original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.markdown('---')  # Adding a separator for visual clarity

# User input for questions
st.subheader('Ask a Question')
user_question = st.text_input('Your Question:', '')

if st.button('Ask'):
    if user_question.lower() == 'what is the current stock price?':
        if not df.empty:
            current_price = df['Close'].iloc[-1]
            st.write(f"The current stock price of {user_input} is ${current_price:.2f}")
        else:
            st.write("No data available.")

    elif user_question.lower() == 'what was the highest stock price in the given period?':
        if not df.empty:
            highest_price = df['High'].max()
            st.write(f"The highest stock price of {user_input} in the given period was ${highest_price:.2f}")
        else:
            st.write("No data available.")

    else:
        st.write("I'm sorry, I couldn't understand your question.")
