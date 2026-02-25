#Loading the required libraries
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

#Streamlit dashboard creation
import streamlit as st
import plotly.express as px


#Loading the trained model
##Load the registered model for the Streamlit dashboard
@st.cache_resource
def load_rmodel():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #Loading the Decision tree for regression
    run_id = "bdee01bedada4dba95d5b1897490fecc"
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

xgbl = load_rmodel()

#Load the artifacts for this model
@st.cache_resource
def load_rartifact():
    artifact_path = mlflow.artifacts.download_artifacts(run_id="bdee01bedada4dba95d5b1897490fecc")
    #Load the Scaler for the Regression model
    x_scaler = joblib.load(f"{artifact_path}/x.std_scaler.pkl")
    x_minmax = joblib.load(f"{artifact_path}/x.minmax.pkl")
    y_scaler = joblib.load(f"{artifact_path}/y.scaler.pkl")
    return x_scaler, x_minmax, y_scaler

x_scaler,x_minmax,y_scaler = load_rartifact()


#Loading the trained model
##Load the registered model for the Streamlit dashboard
@st.cache_resource
def load_nrmodel():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #Loading the Decision tree for regression
    run_id = "bce057d0839840ddab11309f5ccf1db4"
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

xgbnl = load_nrmodel()

#Load the artifacts for this model
@st.cache_resource
def load_nrartifact():
    artifact_path = mlflow.artifacts.download_artifacts(run_id="bce057d0839840ddab11309f5ccf1db4")
    #Load the Scaler for the Regression model
    xc_scaler = joblib.load(f"{artifact_path}/xc.std_scaler.pkl")
    xc_minmax = joblib.load(f"{artifact_path}/xc.minmax.pkl")
    return xc_scaler, xc_minmax

xc_scaler,xc_minmax = load_nrartifact()

#Loading the Dataset 
data = pd.read_csv("D:/Sathish/AIML/Real Estate Investment Advisor/india_housing_prices.csv")
#Preprocessing the dataset as similar as training dataset
data['Amenities'] = data['Amenities'].apply(lambda x: [i.strip() for i in x.split(',')])
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
amenities = mlb.fit_transform(data['Amenities'])
amen = pd.DataFrame(amenities,columns=mlb.classes_)
data_1 = pd.concat([data,amen],axis=1)
data_1.drop(data_1[['ID','Price_per_SqFt','Year_Built','Amenities']],axis=1,inplace=True)
data_1 = pd.get_dummies(data_1,columns=['State','City','Locality','Property_Type','Furnished_Status','Public_Transport_Accessibility','Facing','Owner_Type'])
data_1.drop(data_1[['State_Andhra Pradesh','City_Ahmedabad','Locality_Locality_1','Property_Type_Apartment','Furnished_Status_Unfurnished','Public_Transport_Accessibility_Low','Facing_North','Owner_Type_Broker']],axis=1,inplace=True)

input = pd.DataFrame(columns=data_1.columns)
input.loc[0] = 0


st.set_page_config(layout='wide')
st.title("Real Estate Advisor")
st.sidebar.title("Page Navigation")
pages = st.sidebar.radio(
    "Go to",
    ['Home','Real time prediction']
)

if pages == "Home":
    st.write("projectwefwdcx")
elif pages == "Real time price prediction":
    s,col1,a,col2,col3,col4,col5,col6 = st.columns(8)
    with s:
        state_filter = st.selectbox(
            "Select State",
            options = sorted(data['State'].unique())
        )
    with col1:
        city_filter = st.selectbox(
        "Select City",
        options = sorted(data['City'].unique())
        )
    with a:
        locality_filter = st.selectbox(
            "Select Locality",
            options = sorted(data['Locality'].unique())
        )
    with col2:
        prop_type = st.selectbox(
        "Select Property type",
        options = sorted(data['Property_Type'].unique())
        )
    with col3:
        avail_status = st.selectbox(
        "Availability Status",
        options = sorted(data['Availability_Status'].unique())
        )
    with col4:
        furn_status = st.selectbox(
            "Furnished Status",
            options=sorted(data['Furnished_Status'].unique())
        )
    with col5:
        owner_type = st.selectbox(
            "Select Owner type",
            options = sorted(data['Owner_Type'].unique())
        )
    with col6:
        facing_dir = st.selectbox(
            "Facing direction",
            options = sorted(data['Facing'].unique())
        )
    st.markdown("**Amenities**")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        clubhouse = st.radio('Clubhouse',['Yes','No'])
    with col2:
        garden = st.radio('Garden',['Yes','No'])
    with col3:
        gym = st.radio("Gym",['Yes','No'])
    with col4:
        playground = st.radio('Playground',['Yes','No'])
    with col5:
        pool = st.radio('Pool',['Yes','No'])
    price = st.number_input(
        "Enter Current Property price in lakhs",
        min_value = 0.0,
        step = 0.1,
        format = "%.2f"
    )
    with st.sidebar:
        st.markdown("**Filters**")
        BHK_filter = st.selectbox(
            "Select House BHK",
            options = sorted(data['BHK'].unique())
        )
        sqft_size = st.selectbox(
            "Select sqft size",
            options = sorted(data['Size_in_SqFt'].unique())
        )
        prop_age = st.selectbox(
            "Select Property age",
            options = sorted(data['Age_of_Property'].unique())
        )
        floor_no = st.selectbox(
            "Select Floor no",
            options = sorted(data['Floor_No'].unique())
        )
        total_floors = st.selectbox(
            "Select total Floors",
            options = sorted(data['Total_Floors'].unique())
        )
        no_schl = st.selectbox(
            "No.of.Schools",
            options = sorted(data['Nearby_Schools'].unique())
        )
        no_hosp = st.selectbox(
            "No.of.Hosp",
            options = sorted(data['Nearby_Hospitals'].unique())
        )
        park_space = st.selectbox(
            "Parking available",
            options = sorted(data['Parking_Space'].unique())
        )
        security_avail = st.selectbox(
            "Security available",
            options = sorted(data['Security'].unique())
        )
        transport_fac = st.selectbox(
            "Transport Facility",
            options = sorted(data['Public_Transport_Accessibility'].unique()) 
        )

    #Map Numeric values to filter
    yes_no = {'No':0, 'Yes':1}
    Parking_space = yes_no.get(park_space,0)
    sec_sp = {'No':0, 'Yes':1}
    Security = sec_sp.get(security_avail,0)
    avail_sp = {'Under_Construction':0, 'Ready_to_Move':1}
    Availability_Status = avail_sp.get(avail_status,0)
    Clubhouse = yes_no.get(clubhouse,0)
    Garden = yes_no.get(garden,0)
    Gym = yes_no.get(gym,0)
    Playground = yes_no.get(playground,0)
    Pool = yes_no.get(pool,0)

    #Assign numeric filters
    input.loc[0,'BHK'] = BHK_filter
    input.loc[0,'Size_in_SqFt'] = sqft_size
    input.loc[0,'Floor_No'] = floor_no
    input.loc[0,'Total_Floors'] = total_floors
    input.loc[0,'Age_of_Property'] = prop_age
    input.loc[0,'Nearby_Schools'] = no_schl
    input.loc[0,'Nearby_Hospitals'] = no_hosp
    input.loc[0,'Parking_Space'] = Parking_space
    input.loc[0,'Security'] = Security
    input.loc[0,'Availability_Status'] = Availability_Status
    input.loc[0,'Clubhouse'] = Clubhouse
    input.loc[0,'Garden'] = Garden
    input.loc[0,'Gym'] = Gym
    input.loc[0,'Playground'] = Playground
    input.loc[0,'Pool'] = Pool
    input.loc[0,'Price_in_Lakhs'] = price

    #Assign dummy variable values
    state = f"State_{state_filter}"
    if state in input.columns:
        input.loc[0,state] = 1

    city = f"City_{city_filter}"
    if city in input.columns:
        input.loc[0,city] = 1

    locality = f"Locality_{locality_filter}"
    if locality in input.columns:
        input.loc[0,locality] = 1

    property = f"Property_Type_{prop_type}"
    if property in input.columns:
        input.loc[0,property] = 1

    furnished = f"Furnished_Status_{furn_status}"
    if furnished in input.columns:
        input.loc[0,furnished] = 1

    owner = f"Owner_Type_{owner_type}"
    if owner in input.columns:
        input.loc[0,owner] = 1

    facing = f"Facing_{facing_dir}"
    if facing in input.columns:
        input.loc[0,facing] = 1

    #Transform the inputs features
    inputr = input
    inputr[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']] = x_scaler.transform(inputr[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']])
    inputr[['Nearby_Schools','Nearby_Hospitals']] = x_minmax.transform(inputr[['Nearby_Schools','Nearby_Hospitals']])

    y_pred_scaled = xgbl.predict(inputr)
    st.write("Predicted value:",y_pred_scaled)
    y_pred_real = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
    st.write("Predicted Ruppees in lakhs:",y_pred_real)
    #Storing the predicted value
    st.metric("Prediction",f"{y_pred_real[0][0]}")
    df_reg = pd.DataFrame({"Prediction":['Price'],"Value":[y_pred_real[0][0]]})
    fig = px.bar(df_reg,x='Prediction',y="Value",title="Predicted Property Price")
    st.plotly_chart(fig,use_container_width=True)

    #Transform the input features
    inputc = input
    inputc[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']] = xc_scaler.transform(inputc[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']])
    inputc[['Nearby_Schools','Nearby_Hospitals']] = xc_minmax.transform(inputc[['Nearby_Schools','Nearby_Hospitals']])

    y_cpred_scaled = xgbnl.predict(inputc)[0]
    st.write("Is a good Investment?:",'yes' if y_cpred_scaled == 1 else 'no')



