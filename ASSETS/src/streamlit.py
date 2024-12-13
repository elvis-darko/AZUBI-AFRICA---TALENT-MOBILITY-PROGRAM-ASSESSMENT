import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
# import requests
import numpy 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import urllib.request

--hiddenimport=numpy.distutils
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
    #exp_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/images.jpeg"
    #st.image(exp_url, caption='PEOPLE NATIONAL BANK TERM DEPOSIT PREDICTION APP', use_container_width=True)
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
    
    # Raw GitHub URL of your model
    #gb_model_tund = joblib.load(r"C:\\Users\\ICUMS\\Documents\\GitHub\\AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT\\ASSETS\\src\\gb_model_tuned.joblib")
    
    # Get the raw URL of your model from GitHub
    
    model_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/dev/gb_model_tuned.pkl"

    # response = requests.get(model_url)
    # with open("gb_model_tuned.pkl", "wb") as f:
    #     f.write(response.content)
        
    # model = pickle.load(open("gb_model_tuned.joblib", "rb"))
    # st.cache_data 
    # st.cache_resource

    def load_model():

        with urllib.request.urlopen(model_url) as url:

            model_data = pickle.load(url)

        return model_data



    model = load_model()
    


    # Title of the page
    st.title('CLIENT TERM SUBSCRIPTION PREDICTION')

    # Add the image using st.image
    #image_url = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/deposit.jpeg"
    #st.image(image_url, caption='Term Deposit Prediction App', use_container_width=True)

    
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
    pdays = st.number_input('Pdays: Number of days that passed by after the client was last contacted from a previous campaign')
    campaign = st.number_input('Campaign: Number of contacts performed during this campaign and for this client')

    # Calculate values
    campaign_diff = calculate_campaign_diff(campaign, previous)
    
    # Display calculated values
    st.number_input("Campaign Difference", campaign_diff)
    

    # Make prediction, 
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
            "campaign_diff" : campaign_diff
        }
        
        st.dataframe([features])
        #mlit input_features = np.array([[age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, previous, poutcome, pdays, campaign, campaign_diff]])
        input_features = pd.DataFrame([features])


        #input_features["campaign_diff"] = input_features["campaign"] - input_features["previous"]
        input_features = input_features.astype(str)
        input_features = input_features.values.reshape(-1, 1)

        #input_features = pd.DataFrame([features])
        encoder = LabelEncoder()
        #encoder.transformstre(input_features[["job", "marital",  "education", "default", "housing", "loan", "contact", "month",  "day_of_week", "poutcome"]])
        input_features = encoder.fit_transform(input_features)
        

        #st.dataframe([input_features])
        #gb_model_tuned = joblib.load(r"C:\\Users\\ICUMS\\Documents\\GitHub\\AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT\\ASSETS\\src\\gb_model_tuned.joblib")

        prediction = model.predict([input_features])
        #prediction_probability = gb_model_tuned.predict_proba(input_features)[:, 1]  # Probability of churn

        if prediction[0] == "yes":
            #st.image("https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/suscribe.png", use_container_width=True)
            st.write('Prediction: Client is likely to subscribe to new term deposit')
            
            # Display churn probability score
            # prediction_probability = gb_model_tuned.predict_proba(input_features)[:, 1] 
            # st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.80  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {accuracy:.2f}')
            
            # # Display feature importance as a bar chart
            # feature_importance = gb_model_tuned.feature_importances_
            # feature_names = [age, job, marital, education, default, housing, loan, contact, month, 
            #                   day_of_week, duration, previous, poutcome, pdays, campaign, campaign_diff]
            
            # # Create a bar chart
            # plt.barh(feature_names, feature_importance)
            # plt.xlabel('Feature Importance')
            # plt.ylabel('Features')
            # plt.title('Feature Importance Scores')
            
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
            #unsuscribe_pic = "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/unsuscribe.jpeg"
            #st.image(unsuscribe_pic, use_container_width=True) 
            st.write('Prediction: Customer is likely not to subscribe to new term deposit')
            # Display accuracy score
            accuracy = 0.80  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {accuracy:.2f}')
            
            # Display churn probability score
            #st.write(f'Churn Probability Score: {round(prediction_probability[0] * 100, 2)}%')
            
            # Add a message to clients who churn
            # Display recommendations for customers who did not subscribe to new term deposit
            st.write("Recommendations for Term Deposit by clients:")
            st.write("1. The marketing team should .")
            st.write("2. Explore our new product offerings for additional benefits")
            st.write("3. Unlock personalized recommendations and tailored experiences as a loyalty program member. We'll cater  for your preferences and needs like never before.")
            st.write("4. Get an exclusive sneak peek at upcoming features or products. You can even participate in beta testing and help shape our future offerings.")
            st.write("5. Accumulate rewards points with every purchase, which you can redeem for exciting prizes, discounts, or even free products.")
            

def developers_page():
     st.title('THE APP DEVELOPER')
     #dev_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/developer.png"
     #st.image(dev_url, caption='Term Deposit Subscription App', use_container_width=True)
     st.write(f"""
    <p>This term deposit subscription App was solely built by Elvis Darko for the People's National Bank</p>
    <p>Elvis Darko is a budding Azubi Africa trained Data Scientist who aspires to be a fully fledged Artificial Intelligence Engineer</p>
    """, unsafe_allow_html=True)
 
# Set up option menu (side bar)
with st.sidebar:
    #cust_url =  "https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT/raw/main/ASSETS/images/images.jpeg"
    #st.image(cust_url, use_container_width=True)
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

elif selected == "Developer":
    developers_page()
