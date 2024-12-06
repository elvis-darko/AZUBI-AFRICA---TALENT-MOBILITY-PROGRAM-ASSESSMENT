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
def calculate_total_recharge(montant, frequence_rech):
    return montant * frequence_rech

def calculate_avg_revenue_montant(revenue, montant):
    return (revenue + montant) / 2

def calculate_frequence_squared(frequence):
    return frequence ** 2

def calculate_on_net_reg_ratio(on_net, regularity):
    return on_net / regularity

# Set up home page
def home_page():
    st.title('EXPRESSO CUSTOMER CHURN PREDICTION APP')
    exp_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/expresso.png"
    st.image(exp_url, caption='Team Zurich Churn Prediction App', use_column_width=True)
    st.write("""<h2>Welcome to the People's National Bank Client Term Dposit Probability prediction App!</h2>""", unsafe_allow_html=True)
    st.write("This App is for an African commercial bank, The People's National Bank. The bank provides all kinds of financial assistance to clients.")
    st.write("The objective of this project is to develop a machine learning model to predict the likelihood of each client subscribing to a term depoist given certain conditions.")
    st.write("This will the bank to streamlining it's marketing campaigns to keep existing term deposit clients and also gan new subscribers.")
    st.write(f"""
    <p>The following method will help you to use the app:</p>
    <ul>
        <li>Input Features: Adjust values for customer features.</li>
        <li>Click 'Predict': Get term deposit prediction."</li>
        <li>Result: See if it's 'yes' or no.</li>
        <li>Recommendations (no): Explore subscription suggestions.</li>
        <li>Accuracy Score: Check prediction performance."</li>
        <li>Feedback (no): Provide input for improvements.</li>
    </ul>
    """, unsafe_allow_html=True)

# Set up prediction page
def prediction_page():

    # Title of the page
    st.title('CLIENT TERM SUBSCRIPTION PROBABILITY')

    # Add the image using st.image
    image_url = "https://github.com/elvis-darko/Team_Zurich_Capstone_Project/raw/main/Assets/images/cust_churn.jpg"
    st.image(image_url, caption='Team Zurich Churn Prediction App', use_column_width=True)

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
    tenure = st.slider('Tenure: Duration in the network', 1, 12, 7)
    montant = st.number_input('Montant: Top-up amount', value=0.0)
    frequence_rech = st.number_input('Frequence Recharge: Number of times the customer refilled', value=0.0)
    revenue = st.number_input('Revenue: Monthly income of each client', value=0.0)
    arpu_segment = st.number_input('ARPU Segment: Income over 90 days / 3', value=0.0)
    frequence = st.number_input('Frequency: Number of times the customer refilled', value=0.0)
    data_volume = st.number_input('Data Volume: Number of connections', value=0.0)
    on_net = st.number_input('On Net: Inter-expresso call', value=0.0)
    orange = st.number_input('Orange: Call to Orange', value=0.0)
    tigo = st.number_input('Tigo: Call to Tigo', value=0.0)
    zone1 = st.number_input('Zone1: Call to Zone1', value=0.0)
    zone2 = st.number_input('Zone2: Call to Zone2', value=0.0)
    regularity = st.slider('Regularity: Number of times the client is active for 90 days', 1, 61, 30)
    freq_top_pack = st.number_input('Frequency Top Pack: Number of times the client has activated the top pack packages', value=0.0)

    # Calculate values
    total_recharge = calculate_total_recharge(montant, frequence_rech)
    avg_revenue_montant = calculate_avg_revenue_montant(revenue, montant)
    frequence_squared = calculate_frequence_squared(frequence)
    on_net_reg_ratio = calculate_on_net_reg_ratio(on_net, regularity)

    # Display calculated values
    st.text_input("Total Recharge", total_recharge)
    st.text_input("Average Revenue Montant", avg_revenue_montant)
    st.text_input("Frequence Squared", frequence_squared)
    st.text_input("On Net Reg Ratio", on_net_reg_ratio)

    # Make prediction
    if st.button('Predict'):
        input_features = np.array([[tenure, montant, frequence_rech, revenue, arpu_segment, 
                                    frequence, data_volume, on_net, orange, tigo, 
                                    zone1, zone2, regularity, freq_top_pack, 
                                    total_recharge, avg_revenue_montant, 
                                    frequence_squared, on_net_reg_ratio]])
        
        prediction = tuned_gb_model.predict(input_features)
        prediction_probability = tuned_gb_model.predict_proba(input_features)[:, 1]  # Probability of churn

        if prediction[0] == 0:
            st.image("https://creazilla-store.fra1.digitaloceanspaces.com/cliparts/65532/happy-emoji-clipart-md.png", use_column_width=True)
            st.write('Prediction: Not Churn')
            
            # Display churn probability score
            st.write(f'Term Deposit Probability Score: {round(prediction_probability[0] * 100)}%')
            
            # Display accuracy score
            accuracy = 0.80  # Replace with your actual accuracy score
            st.write(f'Accuracy Score: {accuracy:.2f}')
            
            # Display feature importance as a bar chart
            feature_importance = tuned_gb_model.feature_importances_
            feature_names = ["tenure", "montant", "frequence_rech", "revenue", "arpu_segment",
                            "frequence", "data_volume", "on_net", "orange", "tigo",
                            "zone1", "zone2", "regularity", "freq_top_pack", "total_recharge",
                            "avg_revenue_montant", "frequence_squared", "on_net_reg_ratio"]
            
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
