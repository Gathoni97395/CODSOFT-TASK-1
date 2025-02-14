# Titanic Survival Prediction
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The model utilizes logistic regression to analyze passenger data and predict the likelihood of survival.

## Overview
The Titanic dataset provides information about passengers such as their age, class, sex, and whether or not they survived the tragic sinking of the RMS Titanic. The goal of this project is to create a logistic regression model that can predict survival based on these features.


Data Description
The dataset used for this project includes the following columns:

PassengerId: The unique ID assigned to each passenger,Pclass: The passenger's class (1 = First Class, 2 = Second Class, 3 = Third Class), Name: The name of the passenger, Sex: Gender of the passenger (male or female), Age: The age of the passenger,
SibSp: The number of siblings/spouses aboard, Parch: The number of parents/children aboard, Ticket: The ticket number, Fare: The amount of money the passenger paid for the ticket
Cabin: The cabin where the passenger stayed, Embarked: The port where the passenger boarded the Titanic (C = Cherbourg; Q = Queenstown; S = Southampton), Survived: Whether the passenger survived (0 = No, 1 = Yes) — This is the target variable for prediction, Model Implementation
### Data cleaning
Data cleaning was done bu handling missing values, and ensuting consistencty of the data.
Encoding of categorical variables was also done.
### model selection
Logistic regression was chosen due to its simplicity and effectiveness for binary classification tasks (survived or not).
### model training and evaluation
Data Preprocessing: Clean the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
Train-Test Split: Splitting the data into training and testing sets to evaluate model performance.
Train the logistic regression model on the training data.
### Model Evaluation
Evaluating the model on the test data using accuracy, precision, recall, and F1 score.
Prediction: Use the trained model to make survival predictions for new passengers.
