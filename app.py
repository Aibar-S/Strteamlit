#Aibar

import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier



"""
### IBM HR Analytics Employee Attrition & Performance
"""

TARGET = 'Attrition'
LABELS = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}

@st.cache_data
def download_data(path: str) -> pd.DataFrame:
    """Download data from local csv and return DataFrame"""
    df = pd.read_csv(path)
    return df

@st.cache_data
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses dataset"""
    df = df_raw.copy()

    bad_predictors = [
        'EmployeeCount', 'EmployeeNumber', 'Gender', 'MaritalStatus', 'Over18', 'StandardHours',
    ]

    df = df.drop(columns=bad_predictors)

    df[TARGET] = df[TARGET] == 'Yes'
    df = df.replace(["Manager", "Research Director", "Manufacturing Director"], "People Manager")
    df['IsPeopleManager'] = df.JobRole == 'People Manager'
    df['OverTime'] = df['OverTime'] == 'Yes'
    df = df.replace({"BusinessTravel": {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}})
    df['AgeGroup'] = pd.cut(df.Age, bins=[18, 30, 40, 50, 60, 70], include_lowest=True, labels=[1,2,3,4,5])

    unselected_features = [
        'Age', 'TrainingTimesLastYear', 'JobRole', 
        'EducationField', 'Department', 'YearsWithCurrManager',
        'MonthlyIncome', 'MonthlyRate', 'HourlyRate', 'DailyRate', 'PercentSalaryHike',
        'NumCompaniesWorked', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'DistanceFromHome', 'StockOptionLevel', 'BusinessTravel',
    ]

    df = df.drop(columns=unselected_features)

    return df

@st.cache_data
def download_model(X, y):
    """Download ML model, retrain if doesnt exist"""
    try:
        model = joblib.load('model.joblib')
        return model
    except:
        model = RandomForestClassifier().fit(X, y)
        joblib.dump(model, 'model.joblib')
        return model

df_raw = download_data('streamlit/data/HR_employee_attrition.csv')
df = preprocess_data(df_raw)

final_features = [
    'Education', 'IsPeopleManager', 'AgeGroup', 'JobLevel', 
    'EnvironmentSatisfaction','JobInvolvement', 'JobSatisfaction',
    'PerformanceRating', 'RelationshipSatisfaction','WorkLifeBalance','OverTime',
]

X, y = df[final_features], df[TARGET]

mcal1, mcol2, mcol3 = st.columns(3)
mcal1.metric("Rows", df.shape[0])
mcol2.metric("Features", df.shape[1] - 1)
mcol3.metric("Target = Yes", f"{round(df[TARGET].value_counts(normalize=True)[0] * 100, 1)} %")

tab1, tab2 = st.tabs(["Pulse", "Prediction"])

with tab1:
    # Questionnaire
    spinner = st.empty()
    form = st.form("pulse")
    env_satisfaction = form.select_slider(
        'How satisfied are you with the job environment?',
        format_func = lambda x: LABELS.get(x),
        options=[1, 2, 3, 4],
    )

    job_involvement = form.select_slider(
        'How would you describe your level of job involvement?',
        format_func = lambda x: LABELS.get(x),
        options=[1, 2, 3, 4],
    )

    job_satisfaction = form.select_slider(
        'How satisfied are you with the job?',
        format_func = lambda x: LABELS.get(x),
        options=[1, 2, 3, 4]
    )

    relationship_satisfaction = form.select_slider(
        'How satisfied are you with the relationships at work with colleagues?',
        format_func = lambda x: LABELS.get(x),
        options=[1, 2, 3, 4]
    )

    work_life_balance = form.select_slider(
        'How would you describe your level of work-life balance?',
        format_func = lambda x: LABELS.get(x),
        options=[1, 2, 3, 4]
    )

    is_overtime = form.checkbox('Do you work over-time?')

    is_submitted = form.form_submit_button("Submit")

    if is_submitted:
        with st.spinner('Uploading your answers...'):
            time.sleep(3)
        st.success('Prediction is made!')

    answers = {
        'Education': 4,
        'IsPeopleManager': False,
        'AgeGroup': 2,
        'JobLevel': 2,
        'EnvironmentSatisfaction': env_satisfaction,
        'JobInvolvement': job_involvement,
        'JobSatisfaction': job_satisfaction,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': relationship_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': is_overtime,
    }


with tab2:

    answers_to_predict = pd.DataFrame(answers, index=[0])

    model = download_model(X,y)

    proba = model.predict_proba(answers_to_predict)[:,1][0]
    score = round(proba * 100)
    if is_submitted:
        if proba >= 0.5:
            st.error("ATTENTION!")
            st.snow()
            st.metric('Probability of leaving: ', f'{score} %')
        else:
            st.success("GREAT")
            st.balloons()
            st.metric('Probability of leaving: ', f'{score} %')
    else:
        st.error("Submit questionnaire")