import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(page_title='Van Price Prediction', page_icon=':car:', layout='wide')

df = pd.read_csv('Datasets/df_final.csv')

# define the sidebar
with st.sidebar:
    selection = st.radio('', [':house: Homepage', ':chart: Visualizations', ':heavy_dollar_sign: Price Predictor'])

if selection == ':house: Homepage':
    st.title('Homepage')
    st.write("Do you want to know how much is the van of your dreams? Or at least the one you can afford...")
    st.image('https://images.stockcake.com/public/1/4/7/147f0712-e989-4512-8558-bd57bbdd91b2_large/van-life-adventure-stockcake.jpg')

elif selection == ':chart: Visualizations':
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header('Price per brand')
        price_per_brand = df.groupby(["brand"]).agg({'price': 'mean'}).sort_values('price', ascending=False).astype({'price': int}) # mean price per brand
        fig, ax = plt.subplots(figsize=(5,5))
        sns.barplot(data=price_per_brand, y='brand', x='price')
        st.pyplot(fig)

        st.header('Scatterplots')
        st.write('How do the different features affect the price?')
        columnes = ['age', 'km', 'power_cv']
        for i in columnes:
            fig, ax = plt.subplots(figsize=(5,5))
            sns.scatterplot(data=df, x=i, y='price')
            st.pyplot(fig)

    with col2:
        st.header('Price per model')
        price_per_model = df.pivot_table(values="price", index="model", aggfunc='mean').sort_values(['price'], ascending=False).astype({'price': int}) # mean price per model
        fig, ax = plt.subplots(figsize=(10,20))
        sns.barplot(data=price_per_model, y='model', x='price')
        st.pyplot(fig)

else:
    st.title('Van Price Prediction')
    st.write("How much money do you need to save? Let's find out!")
    st.image('https://images.theconversation.com/files/500826/original/file-20221213-22773-agowbw.jpg?ixlib=rb-4.1.0&rect=0%2C502%2C4000%2C2000&q=45&auto=format&w=668&h=324&fit=crop')
    st.subheader('Features :mag:')

    brand_expensive = ['Mercedes-Benz']
    brand_medium = ['Hyundai', 'Volkswagen', 'Toyota', 'Ford', 'Opel', 'Peugeot']
    brand_cheap = ['Nissan', 'Citroën', 'Renault', 'Fiat', 'Dacia']
    brand = st.selectbox(':car: Brand', [None, 'Citroën', 'Dacia', 'Fiat', 'Ford', 'Hyundai', 'Mercedes-Benz',
        'Nissan', 'Opel', 'Peugeot', 'Renault', 'Toyota', 'Volkswagen'])
    if brand is None:
        st.write("Please select a brand.")
    else: # show the rest of the features once the user has selected a brand
        if brand in brand_expensive:
            brand_price_value = 2
        elif brand in brand_medium:
            brand_price_value = 1
        else:
            brand_price_value = 0
        
        model_expensive = ['Grand California', 'Marco Polo', 'V', 'T7', 'Multivan', 'Staria', 'T6']
        model_medium = ['Zafira', 'Traveller', 'Sprinter', 'SpaceTourer', 'Vito', 'T', 'Proace',
        'Crafter', 'Townstar', 'Custom', 'Primastar', 'Transit', 'Caddy',
        'Rifter', 'Trafic', 'NV300', 'Viano', 'Talento', 'H-1', 'Movano',
        'Citan', 'Vivaro', 'T5', 'Connect', 'Ducato', 'Master', 'Combo',
        'Boxer', 'Jumpy', 'NV400', 'Express', 'Jumper']
        model_cheap = ['Berlingo', 'NV250', 'H350', 'Scudo', 'Dokker', 'NV200', 'Kangoo', 'Partner', 'Doblo', 'T4',
        'LT', 'Fiorino', 'Hiace', 'Serena', 'Qubo', 'Bipper', 'Kubistar', 'Nemo']

        # define the different models depending on the brand selected
        if brand == 'Mercedes-Benz':
            model = st.selectbox(':oncoming_automobile: Model', ['Citan', 'Marco Polo', 'Sprinter', 'T', 'V', 'Viano', 'Vito'])
        elif brand == 'Hyundai':
            model = st.selectbox(':oncoming_automobile: Model', ['H-1', 'H350', 'Staria'])
        elif brand == 'Volkswagen':
            model = st.selectbox(':oncoming_automobile: Model', ['Caddy', 'Crafter', 'Grand California', 'LT', 'Multivan', 'T4', 'T5', 'T6', 'T7'])
        elif brand == 'Toyota':
            model = st.selectbox(':oncoming_automobile: Model', ['Hiace', 'Proace'])
        elif brand == 'Ford':
            model = st.selectbox(':oncoming_automobile: Model', ['Connect', 'Custom', 'Transit'])
        elif brand == 'Opel':
            model = st.selectbox(':oncoming_automobile: Model', ['Combo', 'Movano', 'Vivaro', 'Zafira'] )
        elif brand == 'Peugeot':
            model = st.selectbox(':oncoming_automobile: Model', ['Bipper', 'Boxer', 'Partner', 'Rifter', 'Traveller'])
        elif brand == 'Citroën':
            model = st.selectbox(':oncoming_automobile: Model', ['Berlingo', 'Jumper', 'Jumpy', 'Nemo', 'SpaceTourer'])
        elif brand == 'Renault':
            model = st.selectbox(':oncoming_automobile: Model', ['Express', 'Kangoo', 'Master', 'Trafic'])
        elif brand == 'Fiat':
            model = st.selectbox(':oncoming_automobile: Model', ['Doblo', 'Ducato', 'Fiorino', 'Qubo', 'Scudo', 'Talento'])
        elif brand == 'Dacia':
            model = st.selectbox(':oncoming_automobile: Model', ['Dokker'])

        if model in model_expensive:
            model_price_value = 2
        elif model in model_medium:
            model_price_value = 1
        else:
            model_price_value = 0

        # rest of the features
        age = st.number_input(':calendar: Age of the van', min_value=1, max_value=30, step=1)
        km = st.number_input(':straight_ruler: Mileage in km:', min_value=0, max_value=1000000, step=1)
        
        fuel_input = st.radio(':fuelpump: Fuel:', ['Diesel', 'Gasoline'])
        fuel_dict = {'Diesel': 0, 'Gasoline': 1}
        fuel_value = fuel_dict[fuel_input]

        power_cv = st.number_input(':horse: Horsepower in cv:', step=1)
        consumption = st.number_input(':heavy_dollar_sign: Consumption in L/100km:', step=0.1, format="%.1f")

        owners_input = st.radio(':key: Previous owners:', ['One', 'More than one'])
        owners_dict = {'One': 0, 'More than one': 1}
        owners_value = owners_dict[owners_input]

        doors_input = st.radio(':door: Does it have both rear doors:', ['Yes', 'No, only the right one'])
        doors_dict = {'Yes': 1, 'No, only the right one': 0}
        rear_doors = doors_dict[doors_input]

        cargo_input = st.radio(':package: Is it a cargo van?', ['Yes', 'No'])
        cargo_dict = {'Yes': 1, 'No': 0}
        cargo_value = cargo_dict[cargo_input]

        size_big = ['Grand California', 'Sprinter', 'Crafter', 'Transit', 'Movano', 'Ducato', 'Master', 'Boxer', 'NV400', 'Jumper', 'H350', 'LT']
        size_medium = ['Marco Polo', 'V', 'T7', 'Multivan', 'Staria', 'T6', 'Zafira', 'Traveller', 'SpaceTourer', 'Vito', 'T', 'Proace', 'Custom', 'Primastar', 'Trafic', 'NV300', 'Viano',
                    'Talento', 'H-1', 'Vivaro', 'T5', 'Jumpy', 'Scudo', 'T4', 'Hiace', 'Serena']
        size_small = ['Townstar', 'Caddy', 'Rifter', 'Citan', 'Connect', 'Combo', 'Express', 'Berlingo', 'NV250', 'Dokker', 'NV200', 'Kangoo', 'Partner', 'Doblo', 'Fiorino', 'Qubo', 'Bipper', 'Kubistar', 'Nemo']

        if model in size_big:
            van_size = 2
        elif model in size_medium:
            van_size = 1
        else:
            van_size = 0

        # Machine Learning model
        df = pd.read_csv('Datasets/df_final.csv')
        numericals = df.select_dtypes(np.number)  # Select numerical variables
        y = numericals['price']
        X = numericals.drop(columns=['price'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        # Make the prediction with the trained KNN model
        knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
        knn_reg.fit(scaled_X_train, y_train)

        input_features = [age, km, power_cv, consumption, fuel_value, owners_value, rear_doors, cargo_value, brand_price_value, model_price_value, van_size]

        # Scale the input (same scaling transformation as for training data)
        input_data = scaler.transform([input_features])
        prediction = knn_reg.predict(input_data)

        st.header(f'This van would cost:')

        if st.button('Calculate price'):
            # print the predicted price
            st.header(f'This van would cost {prediction[0].astype(int)} €')

        # let's put a funny image depending on the price
            if prediction > 50000:
                st.image('https://i2.cdn.turner.com/money/dam/assets/130926155851-breaking-bad-cash-640x360.jpg')
            elif prediction < 15000:
                st.image('https://i.pinimg.com/originals/d4/77/d6/d477d6acbd276d21705048626822c009.gif')
            else: 
                st.image('https://miro.medium.com/v2/resize:fit:1000/0*UVhb_mFUjRECuaWm.gif')




