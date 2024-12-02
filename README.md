# AZUBI AFRICA TALENT MOBILITY PROGRAM ASSESSMENT

## ABOUT THE PROJECT
As a member of the data analytics team, my role involves creating tools that use the bank's operational data to help the business achieve its goals and projections.</br>
For this project, I have been tasked to predict whether a client will subscribe to a term deposit (indicated by the variable "y" as "yes" or "n" as "no"). My task involves analyzing the dataset to assess trends and inisghts. Also, I am tasked to build a predictive model that determines the likelihood of a client subscribing to a term deposit based on the features provided in the dataset.

Sample of my tasks are as follows;
1. Conduct Exploratory Data Analysis (EDA)</br>
I identify patterns, correlations, and any necessary data preprocessing steps, such as handling missing values, outliers, and data normalization. 


2. Feature Engineering</br>
I evaluate which features might be most relevant to predicting client subscription and consider creating new features if applicable. 


3. Build a Predictive Model</br> 
I use a machine learning algorithm of choice to build a model predicting the subscription outcome. 


4. Evaluate Model Performance</br> 
I use appropriate metrics such as accuracy, precision, recall, and F1 score to assess model effectiveness. Also, I Consider any imbalanced classes and adjust accordingly, possibly using techniques like oversampling, undersampling, or adjusting the class weights. 


5. Explain the Findings and Insights</br>
I summarize key findings from the EDA and insights from the model, such as which features were most impactful, common characteristics of clients likely to subscribe, and actionable recommendations for the marketing team. 


## THE DATA  
The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

There are four datasets:  

- `bank-additional-full.csv` with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed. 
- `bank-additional.csv with` 10% of the examples (4119), randomly selected from 1), and 20 inputs. 
- `bank-full.csv` with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).  
- `bank.csv` with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).

The data has 17 attributes which are listed in the table.
|VARAIABLE|DEFINITION|DATA TYPE|
|---------|----------|---------|
|`AGE`| Age of client|Numeric|
|`JOB` |Type of job|Categorical|
|`MARITAL` |Marital status of client|Categorical|
|`EDUCATION` |Education level of client|Categorical|
|`DEFAULT`| Has client's credit defaluted|Binary|
|`HOUSING`|Does client have a house loan|Binary|
|`BALANCE`|Client average yearly balance|Numeric|
|`LOAN`| Does the client have a personal loan|Binary|
|`CONTACT`| Type of communication with bank|Categorical|
|`DAY`|Last contact day of the month|Numeric|
|`MONTH`|Last contact month of the year |Categorical|
|`DURATION`|Last contact duration in seconds|Numeric|
|`CAMPAIGN`|Number of contacts performed during this campaign and for this client|Numeric, includes last contact|
|`PDAYS`|Number of days that passed by after the client was last contacted from a previous campaign|Numeric, -1 means client was not previously contacted|
|`PREVIOUS`|Number of contacts performed before this campaign and for this client|Numerical|
|`POUTCOME`|Outcome of the previous marketing campaign|Categorical|
|`Y`|Has the client subscribed a term deposit?|Binary|

## SETUP
It is recommended to have Virtual Studio Code or any other standard code editor on your local machine.<br />Install the required packages locally to your computer.

It is recommended that you run a python version 3.0 and above. 
You can download the required python version from [here](https://www.python.org/downloads/).

Use these recommended steps to set up your local machine for this project:

1. **Clone the repo :** To clone this repo, copy the url and paste it in your GitHub desktop or code editor on your local machine.
        
        https://github.com/elvis-darko/AZUBI-AFRICA---TALENT-MOBILITY-PROGRAM-ASSESSMENT.git

1. **Create the Python's virtual environment :** <br />This will isolate the required libraries of the project to avoid conflicts.<br />Choose any of the line of code that will work on your local machine.

            python3 -m venv venv
            python -m venv venv


2. **Activate the Python's virtual environment :**<br />This will ensure that the Python kernel & libraries will be those of the created isolated environment.

            - for windows : 
                         venv\Scripts\activate

            - for Linux & MacOS :
                         source venv/bin/activate


3. **Upgrade Pip :**<br />Pip is the installed libraries/packages manager. Upgrading Pip will give an to up-to-date version that will work correctly.

            python -m pip install --upgrade pip


4. **Install the required libraries/packages :**<br />There are libraries and packages that are required for this project. These libraries and packages are listed in the `requirements.txt` file.<br />The text file will allow you to import these libraries and packages into the python's scripts and notebooks without any issue.

            python -m pip install -r requirements.txt 


## MACHINE LEARNING MODEL DEPLOYMENT
### Run Streamlit App
A streamlit app was added for further exploration of the model. The streamlit app provides a simple Graphic User Interface where predicitons can be made from inputs.

- Run the demo app (being at the root of the repository):
        
        Streamlit run streamlit.app.py


## EVALUATION

The model would predict the likelihood of a client subscribing to a term limit given certain parameters.

The final work would look like this:

            client 1d                                   TERM DEPOSIT
            00001                                            0
            000055                                           1
            000081                                           1

- `0 Stands for NO, meaning the client is not likely to suscribe to a term limit`
- `1 Stands for YES, meaning the client is likely to suscribe to a term limit`



## RESOURCES
Here are some ressources you would read to have a good understanding of tools, packages and concepts used in the project:
- [How to improve machine learning models](https://neptune.ai/blog/improving-ml-model-performance)
- [Machine Learning tutorial - A step by step guide](https://github.com/eaedk/Machine-Learning-Tutorials/blob/main/ML_Step_By_Step_Guide.ipynb)
- [Create user interfaces for machine learning models](https://www.youtube.com/watch?v=RiCQzBluTxU)
- [Getting started with Streamlit](https://docs.streamlit.io/library/get-started)


## CONTRIBUTORS
| NAME  |   COUNTRY |   E-MAIL  |
|:------|:----------|:----------|
|ELVIS DARKO|GHANA|elvis_darko@outlook.com|

