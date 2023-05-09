import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor


"""
### Prediction of ROP (Rate of Penetration) of the rock during drilling of wells
"""

TARGET = 'Rate Of Penetration'
#LABELS = {
#    1: 'Low',
#    2: 'Medium',
#    3: 'High',
#    4: 'Very High'
#}

@st.cache_data
def download_data(path: str) -> pd.DataFrame:
    """Download data from local csv and return DataFrame"""
    df = pd.read_csv(path)
    return df

@st.cache_data
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses dataset"""
    df = df_raw.copy()
    
    X = df.drop(['Rate Of Penetration'], axis=1)
    
    scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler.fit(X)
    df_scaled=scaler.transform(X)

    df_scaled=pd.DataFrame(df_scaled, columns=['Hole Depth', 'Hook Load', 'Rotary RPM', 'Rotary Torque', 'Weight on Bit', 'Differential Pressure', 'Gamma at Bit'])
    
    y = df[['Rate Of Penetration']]
    X = df_scaled
    
    return y, X, scaler

@st.cache_data
def download_model(X, y):
    """Download ML model, retrain if doesnt exist"""
    try:
        model = joblib.load('model.joblib')
        return model
    except:
        seed=1000
        np.random.seed(seed)
        SVM = SVR(kernel='rbf', gamma=1.5,C=5)
        model = SVM.fit(X, np.ravel(y))
        joblib.dump(model, 'model.joblib')
        return model

df_raw = download_data('data/ROP_DataSet.csv')
y, X, scaler = preprocess_data(df_raw)

#df, scaler = preprocess_data(df_raw)
#y = df[['Rate Of Penetration']]
#X = df.drop(['Rate Of Penetration'], axis=1)


#mcal1, mcol2, mcol3 = st.columns(3)
#mcal1.metric("Rows", df.shape[0])
#mcol2.metric("Features", df.shape[1] - 1)
#mcol3.metric("Target = Yes", f"{round(df[TARGET].value_counts(normalize=True)[0] * 100, 1)} %")

tab1, tab2 = st.tabs(["Input drilling data", "Prediction of ROP"])

with tab1:
    # Questionnaire
    spinner = st.empty()
    form = st.form("pulse")
    #form = st.form(key='submit_form')
    
#    Hole_Depth = form.select_slider(
#        'How satisfied are you with the job environment?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4],
#    )

    Hole_Depth = st.number_input("Measured Depth in ft")
    Hook_Load = st.number_input("Hook Load in Klbs")
    Rotary_RPM = st.number_input("Rotary RPM")
    Rotary_Torque = st.number_input("Rotary Torque in Klbs-ft")
    Weight_on_Bit = st.number_input("Weight on Bit in Klbs")
    Differential_Pressure = st.number_input("Differential Pressure in psi")
    Gamma_at_Bit = st.number_input("Gamma at Bit in gAPI")

#    Hook_Load = form.select_slider(
#        'How would you describe your level of job involvement?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4],
#    )
#
#    Rotary_RPM = form.select_slider(
#        'How satisfied are you with the job?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4]
#    )
#
#    Rotary_Torque = form.select_slider(
#        'How satisfied are you with the relationships at work with colleagues?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4]
#    )
#
#    Weight_on_Bit = form.select_slider(
#        'How would you describe your level of work-life balance?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4]
#    )
#
#    Differential_Pressure = form.select_slider(
#        'How would you describe your level of work-life balance?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4]
#    )
#
#    Gamma_at_Bit = form.select_slider(
#        'How would you describe your level of work-life balance?',
#        format_func = lambda x: LABELS.get(x),
#        options=[1, 2, 3, 4]
#    )

    is_submitted = form.form_submit_button("Submit")

    if is_submitted:
        with st.spinner('Uploading your answers...'):
            time.sleep(3)
        st.success('Prediction is made!')

    answers = {
        'Hole Depth': Hole_Depth,
        'Hook Load': Hook_Load,
        'Rotary RPM': Rotary_RPM,
        'Rotary Torque': Rotary_Torque,
        'Weight on Bit': Weight_on_Bit,
        'Differential Pressure': Differential_Pressure,
        'Gamma at Bit': Gamma_at_Bit,
    }


with tab2:

    answers_to_predict = pd.DataFrame(answers, index=[0])
    
    scaled_answers_to_predict=scaler.transform(answers_to_predict)
    
    scaled_answers_to_predict=pd.DataFrame(scaled_answers_to_predict, columns=['Hole Depth', 'Hook Load', 'Rotary RPM', 'Rotary Torque','Weight on Bit', 'Differential Pressure', 'Gamma at Bit'])

    model = download_model(X,y)

#    proba = model.predict(answers_to_predict)[:,1][0]

    proba = model.predict(scaled_answers_to_predict)[0]
    
    score = proba
    if is_submitted:
        st.success("Your result is ready and presented below")
        st.balloons()
        st.metric('The rate of penetration for provided drilling data is: ', f'{round(score,2)} ft/hr')
    else:
        st.error("Submit questionnaire")