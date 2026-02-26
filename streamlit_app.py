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
mlflow.set_tracking_uri("http://127.0.0.1:5000")
@st.cache_resource
def load_rmodel():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #Loading the Decision tree for regression
    run_id = "af8acf2682264888af941fd059f41346"
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

xgbl = load_rmodel()

#Load the artifacts for this model
@st.cache_resource
def load_rartifact():
    artifact_path = mlflow.artifacts.download_artifacts(run_id="af8acf2682264888af941fd059f41346")
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
    run_id = "9721835849bb42d586685ff0b4ff7340"
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

xgbnl = load_nrmodel()

#Load the artifacts for this model
@st.cache_resource
def load_nrartifact():
    artifact_path = mlflow.artifacts.download_artifacts(run_id="9721835849bb42d586685ff0b4ff7340")
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
df = pd.concat([data,amen],axis=1)
data_1 = pd.concat([data,amen],axis=1)
data_1.drop(data_1[['ID','Price_per_SqFt','Year_Built','Amenities']],axis=1,inplace=True)
data_1 = pd.get_dummies(data_1,columns=['State','City','Locality','Property_Type','Furnished_Status','Public_Transport_Accessibility','Facing','Owner_Type'])
data_1.drop(data_1[['State_Andhra Pradesh','City_Ahmedabad','Locality_Locality_1','Property_Type_Apartment','Furnished_Status_Unfurnished','Public_Transport_Accessibility_Low','Facing_North','Owner_Type_Broker']],axis=1,inplace=True)

input = pd.DataFrame(columns=data_1.columns)
input.loc[0] = 0


st.set_page_config(layout='wide')
st.title("Real Estate Investment Advisor: Predicting Property Profitability & Future Value")
st.sidebar.title("Page Navigation")
pages = st.sidebar.radio(
    "Go to",
    ['Home','Data Exploration','Real time prediction']
)

if "city_selected" not in st.session_state:
    st.session_state.city_selected = ['Chennai']

if "prop_selected" not in st.session_state:
    st.session_state.prop_selected = ['Apartment']

if pages == "Home":
    st.write(""" 
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
    This project builds a machine learning pipeline to predict future property prices and classify investment type (Good Investment vs. Not Good Investment). 
        - Regression: XGBoost Regressor chosen (lowest R² and RMSE).
        - Classification: XGBoost Regressor chosen (accuracy, precision, recall, F1 > 96%; minimal misclassifications)
""")
elif pages == "Data Exploration":
    st.sidebar.write("Filters")
    excity_filter = st.sidebar.multiselect(
        "Select City",
        options = sorted(df['City'].unique()),
        default = st.session_state.city_selected
    )
    if st.sidebar.button("Select all City"):
        st.session_state.city_selected = list(df['City'].unique())
    if st.sidebar.button("Deselect all City"):
        st.session_state.city_selected = []
    exprop_type = st.sidebar.multiselect(
        "Select Property type",
        options = sorted(df['Property_Type'].unique()),
        default = st.session_state.prop_selected
        )
    if st.sidebar.button("Select all Property"):
        st.session_state.prop_selected = list(df['Property_Type'].unique())
    if st.sidebar.button("Deselect all Property"):
        st.session_state.prop_selected = []
    exprop_age = st.sidebar.number_input(
        "Enter Property Age",
        min_value = 0,
        step = 1,
        max_value = int(df['Age_of_Property'].max()),
        format = "%d"
    )

    col2,col3,col4,col5 = st.columns(4)
    with col2:
        exBHK_filter = st.number_input(
            "Enter BHK",
            min_value = df['BHK'].min(),
            step = 1,
            max_value = int(df['BHK'].max()),
            format = "%d"
        )
    with col3:
        expark_space = st.selectbox(
            "Parking available",
            options = sorted(df['Parking_Space'].unique())
        )
    with col4:
        exsecurity_avail = st.selectbox(
            "Security available",
            options = sorted(df['Security'].unique())
        )
    with col5:
        extransport_fac = st.selectbox(
            "Transport Facility",
            options = sorted(df['Public_Transport_Accessibility'].unique()) 
        )
    filter_df = df[(df['City'].isin(excity_filter)) & (df['Age_of_Property']==exprop_age) & (df['Property_Type'].isin(exprop_type))  \
                   & (df['BHK'] == int(exBHK_filter)) & (df['Parking_Space'] == expark_space) & (df['Security'] == exsecurity_avail) \
                            & (df['Public_Transport_Accessibility'] == extransport_fac)]
    st.write("Total rows:", len(filter_df))
    #Main dashboard layout
    with st.container():
        col1, col2 = st.columns(2)

    with col1:
        #--Bar Chart--#
        st.markdown("**Prices by City**")
        fig1 = px.bar(filter_df,x="City",y="Price_in_Lakhs", color = "City",color_discrete_sequence=["#FF5733", "#33C1FF"])
        fig1.update_layout(height = 400)
        st.plotly_chart(fig1,use_container_width=True)
    with col2:
        st.markdown("**Prices by Property**")
        fig1 = px.bar(filter_df,x="Property_Type",y="Price_in_Lakhs", color = "Property_Type",color_discrete_sequence=["#FF5733", "#33C1FF"])
        fig1.update_layout(height = 400)
        st.plotly_chart(fig1,use_container_width=True)


elif pages == "Real time prediction":
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
        BHK_filter = st.number_input(
            "Enter BHK",
            min_value = int(data['BHK'].min()),
            step =1,
            max_value = int(data['BHK'].max()),
            format = "%d"
        )   
        sqft_size = st.number_input(
            "Enter sqft size",
            min_value = int(data['Size_in_SqFt'].min()),
            step =1,
            max_value = int(data['Size_in_SqFt'].max()),
            format = "%d"
        )
        prop_age = st.number_input(
            "Enter Property age",
            min_value = int(data['Age_of_Property'].min()),
            step =1,
            max_value = int(data['Age_of_Property'].max()),
            format = "%d"
        )
        floor_no = st.number_input(
            "Enter Floor No",
            min_value = int(data['Floor_No'].min()),
            step =1,
            max_value = int(data['Floor_No'].max()),
            format = "%d"
        )
        total_floors = st.number_input(
            "Enter total Floors",
            min_value = int(data['Total_Floors'].min()),
            step =1,
            max_value = int(data['Total_Floors'].max()),
            format = "%d"
        )
        no_schl = st.number_input(
            "Enter nearby school",
            min_value = int(data['Nearby_Schools'].min()),
            step =1,
            max_value = int(data['Nearby_Schools'].max()),
            format = "%d"
        )
        no_hosp = st.number_input(
            "Enter nearby hospitals",
            min_value = int(data['Nearby_Hospitals'].min()),
            step =1,
            max_value = int(data['Nearby_Hospitals'].max()),
            format = "%d"
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

    #Transform the inputs features regressioin
    inputr = input.copy()
    inputr[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']] = x_scaler.transform(inputr[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']])
    inputr[['Nearby_Schools','Nearby_Hospitals']] = x_minmax.transform(inputr[['Nearby_Schools','Nearby_Hospitals']])
    #Transform the input features classification
    inputc = input.copy()
    inputc[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']] = xc_scaler.transform(inputc[['BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors','Age_of_Property']])
    inputc[['Nearby_Schools','Nearby_Hospitals']] = xc_minmax.transform(inputc[['Nearby_Schools','Nearby_Hospitals']])

    st.write("Input Dataset to the Price Prediction Model")
    st.dataframe(inputr)
    st.markdown("---")
    st.write("Input Dataset to the Investment Classification Model")
    st.dataframe(inputc)

    y_cpred_scaled = xgbnl.predict(inputc)[0]
    if y_cpred_scaled == 1:
        st.success("Investment Type: Good Investment")
    else:
        st.error("Investment Type: Not Good Investment")
    
    y_pred_scaled = xgbl.predict(inputr)
    y_pred_real = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
    st.metric("Prediction in Lakhs",f"{y_pred_real[0][0]:.4f}")
    #Storing the predicted value
    df_reg = pd.DataFrame({"Prediction":['Current Price','Future Price'],"Value":[price,y_pred_real[0][0]]})
    fig = px.bar(df_reg,x='Prediction',y="Value",title="Predicted Property Price",color_discrete_sequence=["#FF5733", "#33C1FF"])
    fig.update_layout(bargap=0.8) 
    st.plotly_chart(fig)

else:
    st.write("End")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#