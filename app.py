import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


st.title('Machine Failure Prediction: ')
st.write('The application using Random forest Classifier model which trained from the data retrived from the Hana data base.')

with open("rfc.pkl", 'rb') as file:
    rfc = pickle.load(file)

Air_temp = st.number_input('Air Temperature [K]: ',step =1, min_value = 293, max_value = 306)
Pross_temp = st.number_input('Process Temperature [K]: ',step =1, min_value = 303, max_value = 315)
rpm = st.number_input('Rotational Speed: ',step =1, min_value = 1165, max_value = 2890)
Torque = st.number_input('Torque: ',step =1, min_value = 3, max_value = 76)
Tool_wear = st.number_input('Tool_wear: ',step =1, min_value = 0, max_value = 255)

if st.button('Predict'):
    dic = {'Air temperature [K]':Air_temp,'Process temperature [K]':Pross_temp,'Rotational speed [rpm]':rpm,'Torque [Nm]':Torque,'Tool wear [min]':Tool_wear}
    df = pd.DataFrame(dic, index=[0])
    minmax = MinMaxScaler()
    df_1 = minmax.fit_transform(df)
    df_1 = pd.DataFrame(df_1, columns=df.columns)
    pred = rfc.predict(df_1)
    df['Prediction'] = pred
    st.write(df)
    if(pred[0] == 1):
        st.subheader(':red[WARNING: the chance of machine to failure is High!!!]')
    else:
        st.subheader(':green[Chill!! Your Machine is good to use.]')

if st.button('Auto fill and Predict'):
    df = pd.read_csv('test.csv')
    minmax = MinMaxScaler()
    df_1 = minmax.fit_transform(df)
    df_1 = pd.DataFrame(df_1, columns=df.columns)
    pred = rfc.predict(df_1)
    df['Prediction'] = pred
    df_2 = df.sample()
    st.write(df_2)
    if(int(df_2['Prediction']) == 1):
        st.subheader(':red[WARNING: the chance of machine to failure is High!!!]')
    else:
        st.subheader(':green[Chill!! Your Machine is good to use.]')
