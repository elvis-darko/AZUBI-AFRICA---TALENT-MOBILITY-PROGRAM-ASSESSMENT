import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image

# Set style of page
st.set_page_config(page_title="EXPRESSO CUSTOMER CHURN PREDICTION APP", page_icon="GH", initial_sidebar_state="expanded")

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
    exp_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/blob/main/ASSETS/images/images.jpeg"
    st.image(exp_url, caption='PEOPLE NATIONAL BANK TERM DEPOSIT PREDICTION App', use_column_width=True)
    st.write("""<h2>Welcome to the People's National Bank Client Term Dposit Probability prediction App!</h2>""", unsafe_allow_html=True)
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
    st.write(f"""
    <p>The following are the features of clients:</p>
    <ul>
        <li>AGE : The Age of client (numeric)<li>
        <li>JOB : Type of job (categorical: "administration","unknown","unemployed","management","housemaid","entrepreneur","student"blue-collar","self-employed","retired","technician","services")<li> 
        <li>MARITAL : Marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)<li>
        <li>EDUCATION : Education level of client (categorical: "unknown","secondary","primary","tertiary")<li>
        <li>DEFAULT: Does the client have a credit in default? (binary: "yes","no")<li>
        <li>BALANCE: average yearly balance, in euros (numeric)<li>
        <li>HOUSING: Has the client taken housing loan? (binary: "yes","no")<li>
        <li>LOAN: Has the client taken personal loan? (binary: "yes","no")<li>
        <li>CONTACT: Contact communication type (categorical: "unknown","telephone","cellular")<li> 
        <li>DAY: Last contact day of the month (numeric)<li>
        <li>MONTH: Last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")<li>
        <li>DURATION: Last contact duration, in seconds (numeric)<li>
        <li>CAMPAIGN: Number of contacts performed during this campaign and for this client (numeric, includes last contact)<li>
        <li>PDAYS: Number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)<li>
        <li>PREVIOUS: Number of contacts performed before this campaign and for this client (numeric)<li>
        <li>POUTCOME: Outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")<li>
        <li>TERM_DEPOSIT : Has the client subscribed a term deposit? (binary: "yes","no")<li>
        
    </ul>
    """, unsafe_allow_html=True)

# Set up prediction page
def prediction_page():

    # Title of the page
    st.title('CLIENT TERM SUBSCRIPTION PROBABILITY')

    # Add the image using st.image
    image_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/blob/main/ASSETS/images/images.jpeg"
    st.image(image_url, caption='Term Deposit Prediction App', use_column_width=True)

    # Raw GitHub URL of your model
    model_url = "https://github.com/Preencez/Team_Zurich_Capstone_Project/raw/main/Assets/src/tuned_gb_model.joblib"

    # Download the model file from the URL and save it locally
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("tuned_gb_model.joblib", "wb") as f:
            f.write(response.content)
        tuned_gb_model = joblib.load("tuned_gb_model.joblib")
    else:
        st.error("Failed to load the model from GitHub.")


    # Input form
    age = st.number_input('Age: Age of client', 1, 12, 7)
    job = st.text_input('Job: Type of Job', value=0.0)
    marital = st.text_input('Marital: Marital status of client', value=0.0)
    education = st.text_input('Education: Education level of client', value=0.0)
    default = st.text_input('Credit Default: Has client defaulted on credit?', value=0.0)
    housing = st.text_input('Housing: Does Client have a house loan?', value=0.0)
    loan = st.text_input('Personal Loan: Does the client have a personal loan', value=0.0)
    contact = st.text_input('Contact: Contact communication of client', value=0.0)
    month = st.number_input('Month: Last contact month of the year', value=0.0)
    day_of_week = st.number_input('Day of Week: Last contact day of the year', value=0.0)
    duration = st.number_input('Duration: Last contact duration of the year, in seconds', value=0.0)
    previous = st.number_input('Previous: Number of contacts performed before this campaign and for this client', value=0.0)
    poutcome = st.number_input('Previous Outcome: Outcome of the previous marketing campaign', value=0.0)
    pdays = st.slider('Pdays: Number of days that passed by after the client was last contacted from a previous campaign', 1, 61, 30)
    campaign = st.number_input('Campaign: Number of contacts performed during this campaign and for this client', value=0.0)

    # Calculate values
    campaign_diff = calculate_campaign_diff(campaign, previous)
    
    # Display calculated values
    st.text_input("Campaign Difference", campaign_diff)
    

    # Make prediction
    if st.button('Predict'):
        input_features = np.array([[age, job, marital, education, default, 
                                    housing, loan, contact, month, day_of_week, 
                                    duration, previous, poutcome, pdays, campaign, 
                                    campaign_diff]])
        
        prediction = tuned_gb_model.predict(input_features)
        prediction_probability = tuned_gb_model.predict_proba(input_features)[:, 1]  # Probability of churn

        if prediction[0] == 0:
            st.image("https://creazilla-store.fra1.digitaloceanspaces.com/cliparts/65532/happy-emoji-clipart-md.png", use_column_width=True)
            st.write('Prediction: Subscribed for Term Deposit')
            
            # Display churn probability score
            st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.80  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {accuracy:.2f}')
            
            # Display feature importance as a bar chart
            feature_importance = tuned_gb_model.feature_importances_
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
            churn_pic = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/churn_pic.jpg"
            st.image(churn_pic, use_column_width=True) 
            st.write('Prediction: Churn')
            
            # Display churn probability score
            st.write(f'Churn Probability Score: {round(prediction_probability[0] * 100, 2)}%')
            
            # Add a message to clients who churn
            st.write("We're sorry to see you go. If you have any feedback or concerns, please don't hesitate to reach out to us. We value your input and are always looking to improve our services.")


def developers_page():
     st.title('THE APP DEVELOPERS')
     dev_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/developer.png"
     st.image(dev_url, caption='Term Deposit Subscription App', use_column_width=True)
     st.write(f"""
    <p>This term deposit subscription App was solely built by Elvis Darko for the People's National Bank</p>
    <p>Elvis Darko is a budding Azubi Africa trained Data Scientist who aspires to be a fully fledged Artificial Intelligence Engineer</p>
    """, unsafe_allow_html=True)
 
# Set up option menu (side bar)
with st.sidebar:
    cust_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/expresso.jpg"
    st.image(cust_url, use_column_width=True)
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
