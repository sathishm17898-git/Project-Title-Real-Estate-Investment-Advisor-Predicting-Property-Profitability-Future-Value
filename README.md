# Project-Title-Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value
This project builds a machine learning pipeline to predict future property prices and classify investment type (Good Investment vs. Not Good Investment). 
**Files**
raw data - Raw dataset used for preprocessing and exploratory analysis
Data - Cleaned dataset with y_price (future predicted variable used for price prediction) and growth rate
Used data - Dataset used for model training. (It has y_price (for regression) and y_class (for classification))
Real Estate Investment Advisor - Documentation explains all steps taken in this project.
Project_2.ipynb - Python script file
streamlit_app - Dashboard script file
**Preprocessing**
    •	Converted Year_Built → datetime, ID → string.
    •	Removed redundant variables: ID, Amenities, Year_Built, Price_per_SqFt (correlated –0.61 with Size_in_SqFt), Price_p_s, and Growth Rate.
    •	Outliers in Price_per_SqFt (20,020 rows) detected and dropped due to correlation with property size.
    •	Applied Min–Max scaling to Nearby_Schools and Nearby_Hospitals; dropped density variables.
    •	Standardized all numeric features and the target (y_price) for uniformity.
    •	Dummy encoding with baseline categories: State: Andhra Pradesh, City: Ahmedabad, Locality: Locality_1, Property Type: Apartment, Furnished Status: Unfurnished,       Public Transport: Low, Parking: No, Security: No, Availability: Under Construction, Facing: North, Owner Type: Broker.
**Exploratory Data Analysis**
    •	No duplicate rows.
    •	Property prices and sizes broadly similar across states, cities, and property types.
    •	No linear relationship between BHK, Floor_No, Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals with price.
    •	Property size and price per square foot are inversely correlated.
    •	Qualitative features (furnishing, facing, owner type, amenities) showed limited impact on price.
**Feature Engineering**
    •	Future Price Feature: Constructed using Multiple Correspondence Analysis (MCA) on City, Property Type, and Built Year → reduced to Growth Rate, scaled between –      15 and 25.
    •	Investment Type Classification:
    •	Good Investment: Current price <= median & future price > current price.
    •	Not Good Investment: All other cases.
**Dataset Splitting**
    •	Training: 80%
    •	Validation: 10%
    •	Test: 10%
    •	Fixed random state for reproducibility.
**Model Selection**
    •	Used MLflow to track several algorithms and compare the results to choose the best fitted model.
    •	Regression: XGBoost Regressor chosen (Max. R² and lowest RMSE).
    •	Classification: XGBoost Regressor chosen (high accuracy, precision, recall, F1 > 96%; minimal misclassifications).
**Deployment**
    •	Deployed in Streamlit.
    •	Dashboard shows:
    o	Predicted future property value (bar chart).
    o	Investment type classification (colored badge).
    •	Inputs standardized using training set parameters and the predicted output transformed inversely to original scale for interpretability.


