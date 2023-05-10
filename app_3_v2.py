import streamlit as st
import pandas as pd
import joblib
import time

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
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

tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Images" , "Input drilling data", "Prediction of ROP"])


with tab1:
    st.markdown(
        """
        - What is the Rate of Penetration (**:blue[ROP]**)?
          The Rate of Penetration is defined as the speed at which a drilling bit breaks a formation to deepen the borehole.
          It is normally measured in feet per hour.
          
        - Why **:blue[ROP]** is important?
          The **:blue[ROP]** is an important parameter of drilling activities because an increase in **:blue[ROP]** will certainly improve the economics of the drilling operation.
          
        - How to achieve high **:blue[ROP]**?
          The improvement of the **:blue[ROP]** efficiency is achieved by optimizing the drilling parameters of the formations being penetrated as well as borehole conditions.
          Therefore, well-site engineers must make rational decisions based on a combination of offset-well analysis and real-time live well feed to ascertain the highest feasible **:blue[ROP]**.
        - Why to predict **:blue[ROP]**?
          **:blue[ROP]** prediction can assist precise planning of drilling operations and can reduce drilling costs by decreasing drilling time.

        - Which parameters affect **:blue[ROP]**
            - Measured Depth - measured depth along the well trajectory.
            - Hook Load - The total force pulling down on the hook. This total force includes the weight of the drillstring in air and any other equipment, reduced by any force that tends to reduce that weight (friction along the wellbore wall, buoyant forces on the drillstring caused by its immersion in drilling fluid.
            - Rotary RPM - revolutions per minute. A rotary table is a mechanical device on a drilling rig that provides clockwise rotational force to the drill string. Rotary speed is the number of times the rotary table makes one full revolution in one minute (rpm).
            - Rotary Torque - The force required to rotate the entire drill string and the drilling bit at the bottom of the hole to overcome the rotational friction against the wellbore, the viscous force between the drill string and drilling fluid as well as the drilling bit torque.
            - Weight on Bit - The amount of downward force exerted on the drilling bit by drill collars to break a rock.
            - Differential Pressure - The difference in pressure between the hydrostatic head of the drilling fluid in the fluid column, and the pressure exerted from the formation at any given depth in the hole. May be positive, zero, or negative with respect to the hydrostatic head.
            - Gamma at Bit - The natural radioactivity in a formation used to classify lithologies and correlate zones.


    """
    )

with tab2:

    st.markdown(
        """
        - A drilling bit is attached to the end of a long string of jointed, hollow drill pipe.
        - The whole assembly is rotated by a motorized turntable at the surface, the rotary table.
        - The rotating bit cuts or crushes the rock.
        - Drilling mud, consisting of water or an oil-water mixture, solids, and various additives, is circulated down through the drill pipe and out through nozzles in the drilling bit.
        - The mud returns to the surface up the annulus, the space outside of the drill pipe.
        - The mud lubricates the bit, prevents it from getting too hot because of friction, and lifts the drilled rock cuttings up the hole.
        - It should be dense enough to overbalance any high-pressure formations encountered while drilling.
        - If it fails in this last action, the fluid in the formation will displace the mud up the hole and hydrocarbons could exit at the surface and a blowout results.

    """

    image = Image.open('drilling_rig.JPG')
    st.image(image, caption='Drilling rig')

with tab3:
    # Questionnaire

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


with tab4:

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
        

