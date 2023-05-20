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


def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")
    
    st.sidebar.title("Tabs")
    tabs = ["Picture", "Prediction", "About"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
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
        st.title("predictions made here")
    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")
    
if __name__ == "__main__":
    main()
