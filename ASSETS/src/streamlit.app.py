import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import pandas as pd



# Set style of page
st.set_page_config(page_title="PEOPLE  NATIONAL BANK TERM DEPOSIT PREDICTION APP", page_icon="GH", initial_sidebar_state="expanded")

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
    st.title('CLIENT TERM SUBSCRIPTION APP')
    exp_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/images.jpeg"
    st.image(exp_url, caption='PEOPLE NATIONAL BANK TERM DEPOSIT PREDICTION APP', use_container_width=True)
    st.write("""<h2>Welcome to the People's National Bank Client Term Deposit Prediction App!</h2>""", unsafe_allow_html=True)
    st.write("This App is for an African commercial bank, The People's National Bank. The bank provides all kinds of financial assistance to clients.")
    st.write("The objective of this project is to develop a machine learning model to predict the likelihood of each client subscribing to a term depoist given certain conditions.")
    st.write("This will enable the bank in streamlining it's marketing campaigns to keep existing term deposit clients and also gan new subscribers.")
    st.write(f"""
    <p>The following method will help you to use the app:</p>
    <ul>
        <li>Input Features: Imput values for customer features.</li>
        <li>Click 'Predict': Get term deposit prediction."</li>
        <li>Result: See if it's 'yes' or 'no'.</li>
        <li>Recommendations (no): Explore subscription suggestions.</li>
        <li>Accuracy Score: Check prediction performance."</li>
        <li>Feedback (no): Provide input for improvements.</li>
    </ul>
    """, unsafe_allow_html=True)
#     st.write('The following are the features of clients')
    
# table = pd.DataFrame([
#     {"FEATURE": "Age", "DESCRIPTION": "Age of client", "DATA TYPE": "Numerical"},
#     {"FEATURE": "Age", "DESCRIPTION": "Age of client", "DATA TYPE": "Numerical"}])
    
#st.table(table)    
   

# Set up prediction page
def prediction_page():
    global gb_model_tuned    
    
    # Raw GitHub URL of your model
    model_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/src/gb_model_tuned.joblib"

    # Download the model file from the URL and save it locally
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("gb_model_tuned.joblib", "wb") as f:
            f.write(response.content)
        gb_model_tuned = joblib.load("gb_model_tuned.joblib")
    else:
        st.error("Failed to load the model from GitHub.")


    # Title of the page
    st.title('CLIENT TERM SUBSCRIPTION PREDICTION')

    # Add the image using st.image
    image_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/deposit.jpeg"
    st.image(image_url, caption='Term Deposit Prediction App', use_container_width=True)

    # Input form
    age = st.number_input('Age: Age of client')
    job = st.text_input('Job: Type of Job')
    marital = st.text_input('Marital: Marital status of client')
    education = st.text_input('Education: Education level of client')
    default = st.text_input('Credit Default: Has client defaulted on credit?')
    housing = st.text_input('Housing: Does Client have a house loan?')
    loan = st.text_input('Personal Loan: Does the client have a personal loan')
    contact = st.text_input('Contact: Contact communication of client')
    month = st.text_input('Month: Last contact month of the year')
    day_of_week = st.text_input('Day of Week: Last contact day of the year')
    duration = st.number_input('Duration: Last contact duration of the year, in seconds')
    previous = st.number_input('Previous: Number of contacts performed before this campaign and for this client')
    poutcome = st.text_input('Previous Outcome: Outcome of the previous marketing campaign')
    pdays = st.number_input('Pdays: Number of days that passed by after the client was last contacted from a previous campaign')
    campaign = st.number_input('Campaign: Number of contacts performed during this campaign and for this client')

    # Calculate values
    campaign_diff = calculate_campaign_diff(campaign, previous)
    
    # Display calculated values
    st.text_input("Campaign Difference", campaign_diff)
    

    # Make prediction
    if st.button('Predict'):
        input_features = np.array([[age, job, marital, education, default, housing, loan, contact, 
                                    month, day_of_week, duration, previous, poutcome, pdays, campaign, 
                                    campaign_diff]])
        prediction = gb_model_tuned.predict(input_features)
        prediction_probability = gb_model_tuned.predict_proba(input_features)[:, 1]  # Probability of churn

        if prediction[0] == 0:
            st.image("https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/suscribe.png", use_container_width=True)
            st.write('Prediction: Client likely to subscribe to new term deposit')
            
            # Display churn probability score
            st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.80  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {accuracy:.2f}')
            
            # Display feature importance as a bar chart
            feature_importance = gb_model_tuned.feature_importances_
            feature_names = [age, job, marital, education, default, housing, loan, contact, month, 
                             day_of_week, duration, previous, poutcome, pdays, campaign, campaign_diff]
            
            # Create a bar chart
            plt.barh(feature_names, feature_importance)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance Scores')
            
            # Display the chart using Streamlit
            st.pyplot(plt)
            
            # Display recommendations for customers who did not subscribe to new term deposit
            st.write("Recommendations for Term Deposit by clients:")
            st.write("1. The marketing team should .")
            st.write("2. Explore our new product offerings for additional benefits")
            st.write("3. Unlock personalized recommendations and tailored experiences as a loyalty program member. We'll cater  for your preferences and needs like never before.")
            st.write("4. Get an exclusive sneak peek at upcoming features or products. You can even participate in beta testing and help shape our future offerings.")
            st.write("5. Accumulate rewards points with every purchase, which you can redeem for exciting prizes, discounts, or even free products.")
            
        else:
            # Handle the case where the prediction is churn
            unsuscribe_pic = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/unsuscribe.jpeg"
            st.image(unsuscribe_pic, use_container_width=True) 
            st.write('Prediction: Customer is likely not to subscribe to new term deposit')
            
            # Display churn probability score
            st.write(f'Churn Probability Score: {round(prediction_probability[0] * 100, 2)}%')
            
            # Add a message to clients who churn
            # Display recommendations for customers who did not subscribe to new term deposit
            st.write("Recommendations for Term Deposit by clients:")
            st.write("1. The marketing team should .")
            st.write("2. Explore our new product offerings for additional benefits")
            st.write("3. Unlock personalized recommendations and tailored experiences as a loyalty program member. We'll cater  for your preferences and needs like never before.")
            st.write("4. Get an exclusive sneak peek at upcoming features or products. You can even participate in beta testing and help shape our future offerings.")
            st.write("5. Accumulate rewards points with every purchase, which you can redeem for exciting prizes, discounts, or even free products.")
            

def developers_page():
     st.title('THE APP DEVELOPERS')
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
        options=["Home", "Prediction", "Developers"],
        icons=["house", "droplet", "people"],
        styles=css_style
   )
    

if selected == "Home":
    home_page()

elif selected == "Prediction":
    prediction_page()

elif selected == "Developers":
    developers_page()
