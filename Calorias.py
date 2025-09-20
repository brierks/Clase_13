import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calorías quemadas ''')
st.image("calories-burn.jpg", caption="Cantidad de calorías quemadas.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Gender = st.number_input('Género (Mujer = 0, Hombre = 1):',  min_value=0, max_value=1, value = 0, step = 1)
  Age = st.number_input('Edad:', min_value=0, max_value=100, value = 0, step = 1)
  Height = st.number_input('Altura (en cm):', min_value=0, max_value=230, value = 0, step = 1)
  Weight = st.number_input('Peso (en kg):', min_value=0, max_value=140, value = 0, step = 1)
  Duration = st.number_input('Duración (en minutos por sesión):', min_value=0, max_value=30, value = 0, step = 1)
  Heart_Rate = st.number_input('Frecuencia cardíaca:', min_value=0, max_value=130, value = 0, step = 1)
  Body_Temp = st.number_input('Temperatura corporal:', min_value=0.0, max_value=42.0, value = 0.0, step = 0.1)

  user_input_data = {'Gender': Gender,
                     'Age': Age,
                     'Height': Height,
                     'Weight': Weight,
                     'Duration': Duration,
                     'Heart_Rate': Heart_Rate,
                     'Body_Temp': Body_Temp}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

calories =  pd.read_csv('Calories_df.csv', encoding='latin-1')
X = calories.drop(columns='Calories')
y = calories['Calories']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df.Gender + b1[1]*df.Age + b1[2]*df.Height + b1[3]*df.Weight + b1[4]*df.Duration + b1[5]*df.Heart_Rate + b1[6]*df.Body_Temp

st.subheader('Cálculo de calorías')
st.write('La cantidad de calorías quemadas fue', prediccion)
