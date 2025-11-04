# MOFs: Machine learning for discovery of MOFs for gas separation applications

In this project, I am building a machine learning model that can predict the gas uptake (carbon dioxide and methane) of metal-organic frameworks (MOFs), which are crystalline materials consisting of inorganic metal nodes linked by organic linkers. The discovery of MOFs for carbon capture is needed for emission reduction technologies, as these materials can efficiently adsorb and store greenhouse gases like CO2.
Machine learning accelerates this discovery process by enabling the prediction of gas uptake properties from structural and chemical descriptors, reducing the need for time-consuming and costly experiments or simulations.
<p align="center">
  <img src="Images/photo_2025-11-04_11-38-42.jpg" alt="Experimental setup" width="90%">

# Main Goals:
- Exploratory data analysis (EDA) to understand the meaningful features and descriptors of MOFs for predicting the CO2 uptake at low/high pressure.
- Predicting the CO2 uptake at low pressure is the main challenge

# Data:
- The dataset used in this project originates from the publication "Understanding the diversity of the metal-organic framework ecosystem", which explores the application of machine learning for predicting gas adsorption properties in MOFs. This dataset contains geometric and chemical descriptors, as well as simulated gas uptake values, enabling the development and evaluation of regression models for materials discovery.
- Number of data points: ~ 5,000
- Number of features: ~ 300

# Model of Choice:
XGBoost
Why?
- I traditionally used XGBoost for any tabular data. So it was an arbitrary choice
- Very good for chemical tabular data
-   It has high accuracy, but is limited such that it overfits easily.
-       usually due to tree depth... so we should be worry about this

## Step 1
Initial train.csv → 80/20 split
Baseline val score (R2) - untuned model, all features = 0.67 to 0.68

## Pipeline
