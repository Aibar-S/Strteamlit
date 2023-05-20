import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor

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

def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")
    
    st.sidebar.title("Tabs")
    tabs = ["About", "Picture", "Prediction"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
    df_raw = download_data('data/ROP_DataSet.csv')
    y, X, scaler = preprocess_data(df_raw)
        
    if selected_tab == "Picture":
        st.title("Picture")
        st.write("Here is the picture:")
        
        picture_url = 'drilling_rig.JPG'  # Replace with the URL of your picture
        st.image(picture_url, use_column_width=False)
        
        st.write("Click on the buttons to view the description of each part.")
        
        # Define the parts of the picture and their descriptions
        parts = {
            "Part A": "Description of Part A.",
            "Part B": "Description of Part B.",
            "Part C": "Description of Part C.",
        }
        
        # Display the buttons and descriptions
        for part_name, description in parts.items():
            if st.button(part_name):
                st.write(description)
    
    elif selected_tab == "Prediction":
        st.title("Enter the required data and press 'Submit' button")
        Hole_Depth = st.number_input("Measured Depth in ft")
        Hook_Load = st.number_input("Hook Load in Klbs")
        Rotary_RPM = st.number_input("Rotary RPM")
        Rotary_Torque = st.number_input("Rotary Torque in Klbs-ft")
        Weight_on_Bit = st.number_input("Weight on Bit in Klbs")
        Differential_Pressure = st.number_input("Differential Pressure in psi")
        Gamma_at_Bit = st.number_input("Gamma at Bit in gAPI")
    
        spinner = st.empty()
        form = st.form("pulse")
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

        answers_to_predict = pd.DataFrame(answers, index=[0])
        
        scaled_answers_to_predict=scaler.transform(answers_to_predict)
        
        scaled_answers_to_predict=pd.DataFrame(scaled_answers_to_predict, columns=['Hole Depth', 'Hook Load', 'Rotary RPM', 'Rotary Torque','Weight on Bit', 'Differential Pressure', 'Gamma at Bit'])

        model = download_model(X,y)

        proba = model.predict(scaled_answers_to_predict)[0]
        
        score = proba
        if is_submitted:
            st.success("The predicted rate of penetration for above provided data is below:")
            st.metric('', f'{round(score,2)} ft/hr')

        
    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")
    
if __name__ == "__main__":
    main()
