import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle, joblib
import urllib.request
from sklearn.preprocessing import StandardScaler


# Set style of page
st.set_page_config(page_title="AZUBI TMP - TERM DEPOSIT PREDICTION APP", page_icon="GH", initial_sidebar_state="expanded")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
}

# Define functions to calculate values
def calculate_campaign_diff(campaign, previous):
    return campaign - previous

# Set up home page
def home_page():
    st.title("CLIENTS' TERM DEPOSIT SUBSCRIPTION APP")
    exp_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/images.jpeg"
    st.image(exp_url, caption='PEOPLE NATIONAL BANK TERM DEPOSIT PREDICTION APP', use_container_width=True)
    st.write("""<h2>Welcome to the People's National Bank Client Term Deposit Prediction App!</h2>""", unsafe_allow_html=True)
    st.write("This App is for an African commercial bank, The People's National Bank. The bank provides all kinds of financial assistance to clients.")
    st.write("The objective of this project is to develop a machine learning model to predict the likelihood of each client subscribing to a term depoist given certain defined features.")
    st.write("This will enable the bank in streamlining it's marketing campaigns to keep existing term deposit clients and also gain new subscribers.")
    st.write(f"""
    <p>The following method will help you to use the app:</p>
    <ul>
        <li>Input Features: Input values for customer features.</li>
        <li>Click Predict: This will give allow the model to predict the lieklihood of a client subscribing to a new term deposit using the entered client features.</li>
        <li>Result: The prediction will show a YES or NO output, as to whether the client will subscribe a for new term deposit.</li>
        <li>Accuracy Score : This score signifies the accuracy of the model in making a prediction.</li>
        <li>The Term Deposit Propbability score: This score signifies the likelihood of a client subscribing to a new term deposit.</li>
        <li>The Feature Importance Plot : This plot shows in descending order, the importance of each client feature.</li>
        <li>Recommendations : Using the importance of client features, the marketing department is given recommendations to retain and gain clients.</li>
    </ul>
    """, unsafe_allow_html=True)

# Set up prediction page
def prediction_page():    
    
    model_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/dev/gb_model_tuned.pkl"
    response = requests.get(model_url)
    with open("gb_model_tuned.pkl", "wb") as f:
        f.write(response.content)    
    model = pickle.load(open("gb_model_tuned.pkl", "rb"))
    

    # Title of the page
    st.title('TERM DEPOSIT SUBSCRIPTION PREDICTION')

    # Add the image using st.image
    image_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/deposit.jpeg"
    st.image(image_url, caption='Term Deposit Prediction App', use_container_width=True)

    
    # Create categorical features
    job_type = ['housemaid', 'services', 'administration', 'blue-collar', 'technician',
       'retired', 'management', 'unemployed', 'self-employed', 'unknown',
       'entrepreneur', 'student']

    marital_status = ['married', 'single', 'divorced', 'unknown']

    education_level = ["lower basic", "high school", "mid basic", "upper basic", 'illiterate', 'unknown',
            "professional course", "university degree"]

    loan_default = ['no', 'unknown', 'yes']

    housing_loan = ['no', 'yes', 'unknown']

    personal_loan = ['no', 'yes', 'unknown']
    
    contact_form = ['telephone', 'cellular']

    month_m = ["june", "july", "august", "october", "november", "december", "march", "may", "april", "september"]

    day = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    outcome = ["nonexistent", "failure", "success"]

    # Input form
    age = st.number_input('Age: Age of client')
    job = st.selectbox('Job: Type of Job', job_type)
    marital = st.selectbox('Marital: Marital status of client', marital_status)
    education = st.selectbox('Education: Education level of client', education_level)
    default = st.selectbox('Credit Default: Has client defaulted on credit?', loan_default)
    housing = st.selectbox('Housing: Does Client have a house loan?', housing_loan)
    loan = st.selectbox('Personal Loan: Does the client have a personal loan', personal_loan)
    contact = st.selectbox('Contact: Contact communication of client', contact_form)
    month = st.selectbox('Month: Last contact month of the year', month_m)
    day_of_week = st.selectbox('Day of Week: Last contact day of the year', day)
    duration = st.number_input('Duration: Last contact duration of the year, in seconds')
    previous = st.number_input('Previous: Number of contacts performed before this campaign and for this client')
    poutcome = st.selectbox('Previous Outcome: Outcome of the previous marketing campaign', outcome)
    emp_var_rate = st.number_input('Employment Variation Rate: Client Employment variation rate') 
    cons_price_idx = st.number_input('Consumer Price Index: Current Consumer Price Index')
    cons_conf_idx =  st.number_input('Consumer Confidence Index: Current Consumer Confidence Index')
    euribor3m =  st.number_input('Euro Interbank Offered Rate: Current 3 months EURIBO rate')
    nr_employed = st.number_input('Number of Employees: Number of Bank Employees')
    pdays = st.number_input('Pdays: Number of days that passed by after the client was last contacted from a previous campaign')
    campaign = st.number_input('Campaign: Number of contacts performed during this campaign and for this client')

    # Calculate values
    campaign_diff = calculate_campaign_diff(campaign, previous)
    
    # Display calculated values
    st.number_input("Campaign Difference", campaign_diff)
    

    # Create feature for dataframe 
    if st.button('Predict'):
        features = {
            "age" : age,
            "job" : job, 
            "marital" : marital, 
            "education" : education, 
            "default" : default, 
            "housing" : housing, 
            "loan" : loan, 
            "contact" : contact, 
            "month" : month, 
            "day_of_week" : day_of_week, 
            "duration" : duration, 
            "campaign" : campaign, 
            "pdays" : pdays,
            "previous" : previous, 
            "poutcome" : poutcome,  
            "emp_var_rate" : emp_var_rate,
            "cons_price_idx" : cons_price_idx,
            "cons_conf_idx" : cons_conf_idx,
            "euribor3m" : euribor3m,
            "nr_employed" : nr_employed,
            "campaign_diff" : campaign_diff
        }
        
        # Display client feature input as a dataframe in streamlit app
        st.dataframe([features])

        # convert client input as a dataframe to be used by model
        input_features = pd.DataFrame([features])

        # Scale data with standard scaler to scale numeric data
        scaler = StandardScaler()
        num_cols = ["age", "duration", "campaign", "pdays", "previous", "campaign_diff"]
        input_features[num_cols] = scaler.fit_transform(input_features[num_cols])

        #import encoder to encode categorical data
        encoder = LabelEncoder()
        input_features["job"] = encoder.fit_transform(input_features["job"])
        input_features["marital"] = encoder.fit_transform(input_features["marital"])
        input_features["education"] = encoder.fit_transform(input_features["education"])
        input_features["default"] = encoder.fit_transform(input_features["default"])
        input_features["housing"] = encoder.fit_transform(input_features["housing"])
        input_features["loan"] = encoder.fit_transform(input_features["loan"])
        input_features["contact"] = encoder.fit_transform(input_features["contact"])
        input_features["month"] = encoder.fit_transform(input_features["month"])
        input_features["day_of_week"] = encoder.fit_transform(input_features["day_of_week"])
        input_features["poutcome"] = encoder.fit_transform(input_features["poutcome"])

    
        # Use model for prediction
        prediction = model.predict(input_features)
        
        # Handle instance where prediction is "YES"
        if prediction[0] == "yes":
            st.image("https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/suscribe.png")
            st.write('Prediction: YES, Client is likely to subscribe to new term deposit')
            
            # Display churn probability score
            prediction_probability = model.predict_proba(input_features)[:, 1] 
            st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.89  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {round(accuracy * 100)}%')            
            
            # Plot feature importance 
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10,5))
            feature_importances = model.feature_importances_

            # Get feature names
            feature_names = input_features.columns 

            # Sort feature importances in descending order
            sorted_idx = np.argsort(feature_importances)[::-1]
                
            # Plot bar chart
            plt.bar(feature_names[sorted_idx], feature_importances[sorted_idx])
            plt.xlabel("Features")
            plt.ylabel("Feature Importance")
            plt.title("Feature Importances")
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
                        
            st.write(f"""
            <p>RECOMMENDATIONS FOR CLIENTS WHO ARE LIKELY FOR SUBSCRIBE:</p>
            <ul>
                <li>The marketing department should have long conversations with this clients and explain to them the benefits of having term deposits</li>
                <li>Multiple people from the marketing department should contact this client, although one private banking counsultant should be assinged to him/her.</li>
                <li>Dedicated follow-up calls should be made to this client even after subscribing for a new term deposit.</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            # Handle the case where the prediction is "NO"
            st.image("https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/unsuscribe.jpeg")
            st.write('Prediction: NO, Client is not likely to subscribe to new term deposit')
            
            # Display churn probability score
            prediction_probability = model.predict_proba(input_features)[:, 1] 
            st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.89  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {round(accuracy * 100)}%')            
            
            # Plot feature importance to show most important client feature
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10,5))
            feature_importances = model.feature_importances_
            # Get feature names
            feature_names = input_features.columns 
            # Sort feature importances in descending order
            sorted_idx = np.argsort(feature_importances)[::-1]   
            # Plot bar chart
            plt.bar(feature_names[sorted_idx], feature_importances[sorted_idx])
            plt.xlabel("Features")
            plt.ylabel("Feature Importance")
            plt.title("Feature Importances")
            plt.xticks(rotation=90)
            st.pyplot(plt)

            st.write(f"""
            <p>RECOMMENDATIONS FOR CLIENTS WHO ARE NOT LIKELY TO SUBSCRIBE FOR NEW TERM DEPOSIT:</p>
            <ul>
                <li>The marketing department should have frequently contact these clients and explain to them the benefits of having term deposits</li>
                <li>These clients should be targeted at the point of the year where consumer confiedence is at its peak. These clients will likely subscribe to new deposit when they are prepared to spend more.</li>
                <li>The marketing department should hire more staff and train them to frequently call and handle this client to persuade him/her to try a new subscription.</li>
            </ul>
            """, unsafe_allow_html=True)
# Create developer page of streamlit app
def developers_page():
    st.title('THE APP DEVELOPER')
    dev_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/developer.png"
    st.image(dev_url, caption='Term Deposit Subscription App', use_container_width=True)
    st.write(f"""
    <p>This term deposit subscription App was solely built by Elvis Darko for the People's National Bank</p>
    <p>Elvis Darko is a budding Azubi Africa trained Data Scientist who aspires to be a fully fledged Artificial Intelligence Engineer</p>
    """, unsafe_allow_html=True)
 
# Set up option menu (side bar)
with st.sidebar:
    cust_url =  "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/images.jpeg"
    st.image(cust_url, use_container_width=True)
    selected = option_menu(
        menu_title=None,
        options=["HOME", "PREDICTION", "DEVELOPER"],
        icons=["house", "droplet", "people"],
        styles=css_style
   )
    

if selected == "HOME":
    home_page()

elif selected == "PREDICTION":
    prediction_page()

elif selected == "DEVELOPER":
    developers_page()
