import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    
    st.sidebar.title("Pages")
    tabs = ["Project description", "Drilling process", "Rate of penetration", "Prediction", "Cross Plot"]
    selected_tab = st.sidebar.radio("Select a page below", tabs)
    
    df_raw = download_data('data/ROP_DataSet.csv')
    y, X, scaler = preprocess_data(df_raw)
        
    if selected_tab == "Drilling process":
        st.title("Drilling process explained")
        st.write("""
        - **:blue[A drilling bit]** is attached to the end of a long string of jointed, hollow **:blue[drill pipe]**.
        - The whole assembly is rotated by a motorized turntable at the surface, the **:blue[rotary table]**.
        - The rotating **:blue[bit]** cuts or crushes the **:blue[rock]**.
        - **:blue[Drilling mud]**, consisting of water or an oil-water mixture, solids, and various additives, is circulated down through the **:blue[drill pipe]** and out through nozzles in the **:blue[drilling bit]**.
        - The **:blue[mud]** returns to the surface up the annulus, the space outside of the **:blue[drill pipe]**.
        - The **:blue[mud]** lubricates the **:blue[bit]**, prevents it from getting too hot because of friction, and lifts the drilled **:blue[rock]** cuttings up the hole.
        - It should be dense enough to overbalance any high-pressure formations encountered while drilling.
        - If it fails in this last action, the fluid in the formation will displace the **:blue[mud]** up the hole and hydrocarbons could exit at the surface and a blowout results. """)
        
        picture_url = 'Pictures/Drilling_pic_1.JPG'  # Replace with the URL of your picture
        st.image(picture_url, use_column_width=False)
        
#        st.write("Click on the buttons to view the description of each part.")
#        
#        # Define the parts of the picture and their descriptions
#        parts = {
#            "Part A": "Description of Part A.",
#            "Part B": "Description of Part B.",
#            "Part C": "Description of Part C.",
#        }
#        
#        # Display the buttons and descriptions
#        for part_name, description in parts.items():
#            if st.button(part_name):
#                st.write(description)
    elif selected_tab == "Rate of penetration":
        st.title("Rate of penetration (ROP) explained")
        st.write("Which parameters affect the **:blue[ROP]**?")
        
        picture_url = 'Pictures/Drilling_pic_3.JPG'  # Replace with the URL of your picture
        st.image(picture_url, use_column_width=False)
        
        st.write("Click on the buttons to view the description of each parameter.")
        
        # Define the parts of the picture and their descriptions
        parts = {
            "**:blue[1-Measured Depth]**": "Measured depth along the well trajectory",
            "**:blue[2-Hook Load]**": "The total force pulling down on the hook. This total force includes the weight of the drillstring in air and any other equipment, reduced by any force that tends to reduce that weight (friction along the wellbore wall, buoyant forces on the drillstring caused by its immersion in drilling fluid.",
            "**:blue[3-Rotary RPM]**": "Revolutions per minute. A rotary table is a mechanical device on a drilling rig that provides clockwise rotational force to the drill string. Rotary speed is the number of times the rotary table makes one full revolution in one minute (rpm).",
            "**:blue[4-Rotary Torque]**": "The force required to rotate the entire drill string and the drilling bit at the bottom of the hole to overcome the rotational friction against the wellbore, the viscous force between the drill string and drilling fluid as well as the drilling bit torque.",
            "**:blue[5-Weight on Bit]**": "The amount of downward force exerted on the drilling bit by drill pipes to break a rock.",
            "**:blue[6-Differential Pressure]**": "The difference in pressure between the hydrostatic head of the drilling fluid in the fluid column, and the pressure exerted from the formation at any given depth in the hole. May be positive, zero, or negative with respect to the hydrostatic head.",
            "**:blue[7-Gamma at Bit]**": "The natural radioactivity in a formation used to classify lithologies and correlate zones. It shows the potential lithologies which can contain oil and gas",
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
    elif selected_tab == "Cross Plot":
        st.title("Rate of penetration vs Measured depth")
        st.write("""
        ### **:blue[The cross plot below shows the real ROP vs predicted ROP]** """)

        # Generate random data for demonstration
        x1 = df_raw["Rate Of Penetration"]
        y1 = df_raw["Hole Depth"]

        model = download_model(X,y)
        x2 = model.predict(X)
        
        #x2 = df_raw["Rate Of Penetration"]
        
        # Create the cross plot graph using Matplotlib
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(x1, y1, c ="blue", label='Actual data')
        ax.scatter(x2, y1, c ="green", label='Predicted data')
        ax.set_xlabel("Rate of penetration, ft/hr")
        ax.set_ylabel("Measured Depth, ft")
        ax.set_title("Predicted vs Actual ROP")
        ax.legend()
        st.pyplot(fig)

        
    else:
        st.title("Project description")
        st.write("""
        ###  **:blue[The current application predicts which rate of penetration could be expected by drilling team]**
          
        - What is the Rate of Penetration (**:blue[ROP]**)?
        
          The Rate of Penetration is defined as the speed at which a drilling bit breaks a formation to deepen the borehole.
          
        - Why **:blue[ROP]** is important?
          
          The **:blue[ROP]** is an important parameter of drilling activities because an increase in **:blue[ROP]** will improve the economics of the drilling operation.
          
        - How to achieve high **:blue[ROP]**?
          
          The improvement of the **:blue[ROP]** efficiency is achieved by optimizing the drilling parameters of the formations being penetrated as well as borehole conditions.
          Therefore, well-site engineers must make rational decisions based on a combination of offset-well analysis and real-time live well feed to ascertain the highest feasible **:blue[ROP]**.

        - Why to predict **:blue[ROP]**?
          
          **:blue[ROP]** prediction can assist precise planning of drilling operations and can reduce drilling costs by decreasing drilling time.""")
    
if __name__ == "__main__":
    main()
