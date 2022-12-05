import streamlit as st
import time
from tqdm.notebook import tqdm
from tensorflow import keras
import datetime as dt
from datetime import date
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
import  plotly.express as px
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import unicodedata
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS


START = "2014-01-01"
TODAY = dt.datetime.now().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ["Select the Stock", "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "GME", "TWTR", "AMD"]


# Loading Data ---------------------

#@st.cache(suppress_st_warning=True)
def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)
    return data


#For Stock Financials ----------------------

def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    marketCap = df_ticker.info['marketCap']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
    fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    ftWeekChange = df_ticker.info['52WeekChange']
    website = df_ticker.info['website']

    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Market Cap -', marketCap)
    st.write('Previous Close -', prevClose)
    st.write('52 Week Change -', ftWeekChange)
    st.write('52 Week High -', fiftyTwoWeekHigh)
    st.write('52 Week Low -', fiftyTwoWeekLow)
    st.write('200 Day Average -', twoHunDayAvg)


#Plotting Raw Data ---------------------------------------

def plot_raw_data(stock, data_1):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName']
    #data1 = df_ticker.history()
    data_1.reset_index()
    #st.write(data_1)
    numeric_df = data_1.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to **_Plot_**')
    features_selected = st.multiselect("", numeric_cols)
    if st.button("Generate Plot"):
        cust_data = data_1[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data_1['Date'], y=features_selected,
                                title= Name + ' ' + '<i>timeline</i>')
        plotly_figure.update_layout(title = {'y':0.9,'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
        plotly_figure.update_xaxes(title_text='Date')
        plotly_figure.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, title="Price"), width=800, height=550)
        st.plotly_chart(plotly_figure)

#Website from where news to be scraped
finviz_url = 'https://finviz.com/quote.ashx?t='

#Getting News Data --------------------------------------

def get_news_df(stock_name,finviz_url):
    
    news_tables = {}
    url = finviz_url + stock_name

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[stock_name] = news_table
    
    parsed_data = []

    for stock_name, news_table in news_tables.items():

        for row in news_table.findAll('tr'):

            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([stock_name, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['stock_name', 'date', 'time', 'title'])

    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date
    df["Negative"] = ''
    df["Neutral"] = ''
    df["Positive"] = ''
    
    sentiment_i_a = SentimentIntensityAnalyzer()
    for indexx, row in df.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', df.loc[indexx, 'title'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            df['Negative'].iloc[indexx] = sentence_sentiment['neg']
            df['Neutral'].iloc[indexx] = sentence_sentiment['neu']
            df['Positive'].iloc[indexx] = sentence_sentiment['pos']
        except TypeError:
            print (df.loc[indexx, 'title'])
            print (indexx)
    
    
    return df

#For Plotting Pie ----------------------------------------------

def plot_cgraph(df):
    posi=0
    nega=0
    neu=0
    for i in range (0,len(df)):
        get_val=df.compound[i]
        if(float(get_val)<(0)):
            nega=nega+1
        if(float(get_val>(0))):
            posi=posi+1
        if(float(get_val==(0))):
            neu=neu+1    
    posper=(posi/(len(df)))*100
    negper=(nega/(len(df)))*100
    neuper=(neu/(len(df)))*100

    fig,ax = plt.subplots()
    arr=np.asarray([posper,negper,neuper], dtype=int)
    ax.pie(arr,labels=['positive','negative','neutral'])
    ax.plot()
    return fig,posper,negper,neuper  

#For Plotting Word Cloud ------------------------------------------

def wordCloudFunction(df,column,numWords):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=numWords,
                          width=1000,height=1000,
                         ).generate(word_string)
    
    fig , ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis('off')
    return fig

#For Plotting Word Bar Graph ----------------------------------------------
    
def wordBarGraphFunction(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    fig , ax = plt.subplots()
    ax.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    return fig   

#For LSTM MOdel ------------------------------

def create_train_test_LSTM(df, epoch, b_s, ticker_name):

    df_filtered = df.filter(['Close'])
    dataset = df_filtered.values

    #Training Data
    training_data_len = math.ceil(len(dataset) * .7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i-60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    #Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]

    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_data, y_train_data, batch_size=int(b_s), epochs=int(epoch))
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted vs Actual Results for LSTM")
    st.write("Stock Prediction on Test Data for - ",ticker_name)

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = df_filtered[:training_data_len]
    valid = df_filtered[training_data_len:]
    valid['Predictions'] = predictions

    new_valid = valid.reset_index()
    new_valid.drop('index', inplace=True, axis=1)
    st.dataframe(new_valid)
    st.markdown('')
    st.write("Plotting Actual vs Predicted ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(14, 8))
    plt.title('Actual Close prices vs Predicted Using LSTM Model', fontsize=20)
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size":20})
    st.pyplot()



#Creating Training and Testing Data for other Models ----------------------

def create_train_test_data(df1):

    data = df1.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df1)), columns=['Date', 'High', 'Low', 'Open', 'Volume', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['High'][i] = data['High'][i]
        new_data['Low'][i] = data['Low'][i]
        new_data['Open'][i] = data['Open'][i]
        new_data['Volume'][i] = data['Volume'][i]
        new_data['Close'][i] = data['Close'][i]

    #Removing the hour, minute and second
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date

    train_data_len = math.ceil(len(new_data) * .8)

    train_data = new_data[:train_data_len]
    test_data = new_data[train_data_len:]

    return train_data, test_data


#Finding Moving Average ---------------------------------------

def find_moving_avg(ma_button, df):
    days = ma_button

    data1 = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data1)):
        new_data['Date'][i] = data1['Date'][i]
        new_data['Close'][i] = data1['Close'][i]

    new_data['SMA_'+str(days)] = new_data['Close'].rolling(min_periods=1, window=days).mean()

    #new_data.dropna(inplace=True)
    new_data.isna().sum()

    #st.write(new_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['SMA_'+str(days)], mode='lines', name='SMA_'+str(days)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)


#Finding Linear Regression ----------------------------

def Linear_Regression_model(train_data, test_data):

    x_train = train_data.drop(columns=['Date', 'Close'], axis=1)
    x_test = test_data.drop(columns=['Date', 'Close'], axis=1)
    y_train = train_data['Close']
    y_test = test_data['Close']

    #First Create the LinearRegression object and then fit it into the model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, y_train)

    #Making the Predictions
    prediction = model.predict(x_test)

    return prediction


#Plotting the Predictions -------------------------


def prediction_plot(pred_data, test_data, models, ticker_name):

    test_data['Predicted'] = 0
    test_data['Predicted'] = pred_data

    #Resetting the index
    test_data.reset_index(inplace=True, drop=True)
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted Price vs Actual Close Price Results for - " ,models)
    st.write("Stock Prediction on Test Data for - ", ticker_name)
    st.write(test_data[['Date', 'Close', 'Predicted']])
    st.write("Plotting Close Price vs Predicted Price for - ", models)

    #Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)



# Sidebar Menu -----------------------

menu=["Stock Exploration and Feature Extraction", "Train Model"]
st.sidebar.title("Settings")
st.sidebar.subheader("Timeseries Settings")
choices = st.sidebar.selectbox("Select the Activity", menu,index=0)



if choices == 'Stock Exploration and Feature Extraction':
    st.subheader("Extract Data")
    #user_input = ''
    st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
    #user_input=st.selectbox("", stocks)
    user_input = st.text_input("", '')

    if not user_input:
        pass
    else:
        data = load_data(user_input)
        st.markdown('Select from the options below to Explore Stocks')

        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration',
                                                     'Extract Features for Stock Price Forecasting','Sentiment Analysis from News'], index=0)
        if selected_explore == 'Stock Financials Exploration':
            st.markdown('')
            st.markdown('**_Stock_ _Financial_** Information ------')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_raw_data(user_input, data)
            st.markdown('')
            shw_SMA = st.checkbox('Show Moving Average')

            if shw_SMA:
                st.write('Stock Data based on Moving Average')
                st.write('A Moving Average(MA) is a stock indicator that is commonly used in technical analysis')
                st.write(
                    'The reason for calculating moving average of a stock is to help smooth out the price of data over '
                    'a specified period of time by creating a constanly updated average price')
                st.write(
                    'A Simple Moving Average (SMA) is a calculation that takes the arithmatic mean of a given set of '
                    'prices over the specified number of days in the past, for example: over the previous 15, 30, 50, '
                    '100, or 200 days.')

                ma_button = st.number_input("Select Number of Days Moving Average", 5, 200)

                if ma_button:
                    st.write('You entered the Moving Average for ', ma_button, 'days')
                    find_moving_avg(ma_button, data)

        elif selected_explore == 'Extract Features for Stock Price Forecasting':
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input("", date(2014, 1, 1))
            st.write('You Selected Data From - ', start_date)
            submit_button = st.button("Extract Features")

            start_row = 0
            if submit_button:
                st.write('Extracted Features Dataframe for ', user_input)
                for i in range(0, len(data)):
                    if start_date <= pd.to_datetime(data['Date'][i]):
                        start_row = i
                        break
                # data = data.set_index(pd.DatetimeIndex(data['Date'].values))
                st.write(data.iloc[start_row:, :])
        
        elif selected_explore == 'Sentiment Analysis from News':
            #Getting news headlines
            ndf = get_news_df(user_input, finviz_url)
            
            st.subheader(' Sentiment Analysis of News ')
            
            
            st.markdown("**Sentiment Views of users**")
            cgs,pstr,nstr,nlstr = plot_cgraph(ndf)
            
            col1, col2= st.columns(2)
            with col1:
                st.pyplot(cgs)
                st.set_option('deprecation.showPyplotGlobalUse', True)   

            with col2:
                st.caption("Positive views= "+str(pstr)+"%")
                st.caption("Negative views= "+str(nstr)+"%")
                st.caption("Neutral views= "+str(nlstr)+"%")
            
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most used words in headlines**")
                wcf = wordCloudFunction(ndf,'title',10000)
                st.pyplot(wcf)

            with col2:
                st.markdown("**Word Count**")
                wbg = wordBarGraphFunction(ndf,'title',"Popular Words in News")
                st.pyplot(wbg)
            
            
            if pstr >=40 and nlstr >=25:
                st.success('Overall Positive : According to Sentiment Analysis of the data, A rise in price trend to be expected.') 
            elif nstr >=40 and nlstr >=25:
                st.error('Negative : According to Sentiment Analysis of the data, A fall in price trend to be expected.')
            else:
                st.info('Neutral : According to Sentiment Analysis of the data, Stalement in price to be expected.')


elif choices == 'Train Model':
    st.subheader("Train Machine Learning Models for Stock Prediction")
    st.markdown('')
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    stock_select = st.selectbox("", stocks, index=0)
    df1 = load_data(stock_select)
    df1 = df1.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    options = ['Select your option', 'Moving Average', 'Linear Regression', 'Random Forest', 'XGBoost', 'LSTM']
    st.markdown('')
    st.markdown('**_Select_ _Machine_ _Learning_ _Algorithms_ to Train**')
    models = st.selectbox("", options)
    submit = st.button('Train Model')

    if models == 'LSTM':
        st.markdown('')
        st.markdown('')
        st.markdown("**Select the _Number_ _of_ _epochs_ and _batch_ _size_ for _training_ form the following**")
        st.markdown('')
        epoch = st.slider("Epochs", 0, 300, step=1)
        b_s = st.slider("Batch Size", 32, 1024, step=1)
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            create_train_test_LSTM(df1, epoch, b_s, stock_select)


    elif models == 'Linear Regression':
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            train_data, test_data = create_train_test_data(df1)
            pred_data = Linear_Regression_model(train_data, test_data)
            prediction_plot(pred_data, test_data, models, stock_select)


    elif models == 'Moving Average':
        ma_button = st.slider('Select Number of Days Moving Average', 0, 200, step=1)
        submit_1 = st.button('Generate')
        if submit_1:
            st.write('Stock Data based on Moving Average')
            st.write('A Moving Average(MA) is a stock indicator that is commonly used in technical analysis')
            st.write('The reason for calculating moving average of a stock is to help smooth out the price of data over '
                  'a specified period of time by creating a constanly updated average price')
            st.write('A Simple Moving Average (SMA) is a calculation that takes the arithmatic mean of a given set of '
                 'prices over the specified number of days in the past, for example: over the previous 15, 30, 50, '
                 '100, or 200 days.')