House Price Prediction System

This project is a Machine Learning web application developed using Streamlit and KNN Regression. The application predicts house prices based on various housing features such as median income, house age, average rooms, population, and location details.

The main goal of this project is to demonstrate how Machine Learning can be used in real-world price prediction problems through an interactive and user-friendly web interface.

Project Overview

The House Price Prediction System allows users to enter housing-related information and receive a predicted house price instantly using a trained KNN Regression model.

The project also includes multiple visualizations and performance analysis sections to better understand the dataset and model behavior.

What This Project Does
Predicts house prices using Machine Learning
Takes user input through a Streamlit web interface
Uses KNN Regression for prediction
Displays data visualizations and graphs
Shows model performance metrics
Provides dataset analysis and insights
Machine Learning Workflow

The application follows the complete Machine Learning workflow:

1. Data Collection

The California Housing dataset is loaded using Scikit-Learn.

2. Data Preprocessing
Features and target values are separated
Data is split into training and testing sets
Feature scaling is applied using StandardScaler
3. Model Training

A KNN Regression model is trained using the scaled training data.

4. Prediction

The trained model predicts house prices based on user input values.

5. Evaluation

The model is evaluated using:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R2 Score
Application Pages
Prediction Page

Users can enter house details such as:

Median Income
House Age
Average Rooms
Population
Latitude and Longitude

The application predicts the estimated house price using the trained KNN model.

Visualization Page

This page contains:

Feature distribution plots
Correlation heatmap
Scatter plots for feature comparison

These visualizations help understand relationships between different housing features.

Model Performance Page

Displays model evaluation metrics including:

MSE
MAE
R2 Score
Actual vs Predicted graph
Dataset Page

Shows:

Dataset preview
Number of rows and columns
Statistical summary of the dataset
Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Algorithm Used
KNN Regression

KNN (K-Nearest Neighbors) Regression predicts values based on the average of the nearest data points.

Why KNN Regression?

Simple and effective
Works well for prediction problems
Easy to understand and implement
Good for learning regression concepts
Features of the Application
Interactive web interface
Professional dashboard design
Real-time predictions
Multiple visualizations
Fast performance using caching
Easy navigation using sidebar
Future Improvements
Add multiple regression algorithms
Allow CSV file uploads
Improve UI with dark mode
Add downloadable prediction reports
Deploy using Streamlit Cloud
Add advanced analytics and charts
Developed By

Sneha Reddy Cheeruka
