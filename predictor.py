import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    selection = st.radio('', [':heavy_dollar_sign: Price Predictor', ':chart: Visualizations'])

if selection == ':chart: Visualizations':
    st.title("Statistics about the models")
    def main():
        html_temp="<div class='tableauPlaceholder' id='viz1742485025753' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B6&#47;B67KWHP49&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;B67KWHP49' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B6&#47;B67KWHP49&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='es-ES' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1742485025753');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1400px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1400px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='2577px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
        components.html(html_temp, width=1420, height=890)

    if __name__ == '__main__':
        main()

else:
    st.title('Van Price Prediction')
    st.write("Do you want to know how much is the van of your dreams? Or at least the one you can afford...")
    st.image('https://images.stockcake.com/public/1/4/7/147f0712-e989-4512-8558-bd57bbdd91b2_large/van-life-adventure-stockcake.jpg')
    st.subheader('Features :mag:')

    brand_expensive = ['Mercedes-Benz']
    brand_medium = ['Hyundai', 'Volkswagen', 'Toyota', 'Ford', 'Opel', 'Peugeot']
    brand_cheap = ['Nissan', 'Citroën', 'Renault', 'Fiat', 'Dacia']
    brand = st.selectbox(':red_car: **Brand**', [None, 'Citroën', 'Dacia', 'Fiat', 'Ford', 'Hyundai', 'Mercedes-Benz',
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
            'Boxer', 'Jumpy', 'NV400', 'Jumper']
        model_cheap = ['Berlingo', 'Scudo', 'Dokker', 'NV200', 'Kangoo', 'Partner', 'Doblo', 'T4',
            'Fiorino', 'Qubo']

        # define the different models depending on the brand selected
        if brand == 'Mercedes-Benz':
            st.image('images/logos/Mercedes-Benz.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Citan', 'Marco Polo', 'Sprinter', 'T', 'V', 'Viano', 'Vito'])
            if model is None:
                pass
            elif model =='Citan':
                st.image('images/van_models/Mercedes-Benz/Citan.jpg')
            elif model =='Marco Polo':
                st.image('images/van_models/Mercedes-Benz/Marco Polo.jpg')
            elif model =='Sprinter':
                st.image('images/van_models/Mercedes-Benz/Sprinter.jpg')
            elif model =='T':
                st.image('images/van_models/Mercedes-Benz/T.jpg')
            elif model =='V':
                st.image('images/van_models/Mercedes-Benz/V.jpg')
            elif model =='Viano':
                st.image('images/van_models/Mercedes-Benz/Viano.jpg')
            else:
                st.image('images/van_models/Mercedes-Benz/Vito.jpg')

        elif brand == 'Hyundai':
            st.image('images/logos/Hyundai.jpg')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'H-1', 'Staria'])
            if model is None:
                pass
            elif model =='H-1':
                st.image('images/van_models/Hyundai/H-1.jpg')
            else:
                st.image('images/van_models/Hyundai/Staria.jpg')

        elif brand == 'Nissan':
            st.image('images/logos/Nissan.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'NV200', 'NV300', 'NV400', 'Townstar', 'Primastar'])
            if model is None:
                pass
            elif model =='NV200':
                st.image('images/van_models/Nissan/NV200.png')
            elif model =='NV300':
                st.image('images/van_models/Nissan/NV300.jpg')
            elif model =='NV400':
                st.image('images/van_models/Nissan/NV400.png')
            elif model =='Townstar':
                st.image('images/van_models/Nissan/Townstar.jpg')
            else:
                st.image('images/van_models/Nissan/Primastar.jpg')

        elif brand == 'Volkswagen':
            st.image('images/logos/Volkswagen.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Caddy', 'Crafter', 'Grand California', 'Multivan', 'T4', 'T5', 'T6', 'T7'])
            if model is None:
                pass
            elif model =='Caddy':
                st.image('images/van_models/Volkswagen/Caddy.jpg')
            elif model =='Crafter':
                st.image('images/van_models/Volkswagen/Crafter.jpg')
            elif model =='Grand California':
                st.image('images/van_models/Volkswagen/Grand California.jpg')
            elif model =='Multivan':
                st.image('images/van_models/Volkswagen/Multivan.jpg')
            elif model =='T4':
                st.image('images/van_models/Volkswagen/T4.jpg')
            elif model =='T5':
                st.image('images/van_models/Volkswagen/T5.jpeg')
            elif model =='T6':
                st.image('images/van_models/Volkswagen/T6.jpg')
            else:
                st.image('images/van_models/Volkswagen/T7.jpg')

        elif brand == 'Toyota':
            st.image('images/logos/Toyota.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Proace'])
            if model is None:
                pass
            else:
                st.image('images/van_models/Toyota/Proace.jpg')

        elif brand == 'Ford':
            st.image('images/logos/Ford.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Connect', 'Custom', 'Transit'])
            if model is None:
                pass
            elif model =='Connect':
                st.image('images/van_models/Ford/Connect.jpg')
            elif model =='Custom':
                st.image('images/van_models/Ford/Custom.jpg')
            else:
                st.image('images/van_models/Ford/Transit.jpg') 

        elif brand == 'Opel':
            st.image('images/logos/Opel.jpg')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Combo', 'Movano', 'Vivaro', 'Zafira'] )
            if model is None:
                pass
            elif model =='Combo':
                st.image('images/van_models/Opel/Combo.jpg')
            elif model =='Movano':
                st.image('images/van_models/Opel/Movano.jpg')
            elif model =='Vivaro':
                st.image('images/van_models/Opel/Vivaro.jpg')
            else:
                st.image('images/van_models/Opel/Zafira.jpg')

        elif brand == 'Peugeot':
            st.image('images/logos/Peugeot.jpg')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Boxer', 'Partner', 'Rifter', 'Traveller'])
            if model is None:
                pass
            elif model =='Boxer':
                st.image('images/van_models/Peugeot/Boxer.jpg')
            elif model =='Partner':
                st.image('images/van_models/Peugeot/Partner.png')
            elif model =='Rifter':
                st.image('images/van_models/Peugeot/Rifter.jpg')
            else:
                st.image('images/van_models/Peugeot/Traveller.jpg')

        elif brand == 'Citroën':
            st.image('images/logos/Citroen.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Berlingo', 'Jumper', 'Jumpy', 'SpaceTourer'])
            if model is None:
                pass
            elif model =='Berlingo':
                st.image('images/van_models/Citroën/Berlingo.png')
            elif model =='Jumper':
                st.image('images/van_models/Citroën/Jumper.png')
            elif model =='Jumpy':
                st.image('images/van_models/Citroën/Jumpy.jpg')
            else:
                st.image('images/van_models/Citroën/SpaceTourer.png')

        elif brand == 'Renault':
            st.image('images/logos/Renault.jpg')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Kangoo', 'Master', 'Trafic'])
            if model is None:
                pass
            elif model =='Kangoo':
                st.image('images/van_models/Renault/Kangoo.jpg')
            elif model =='Master':
                st.image('images/van_models/Renault/Master.jpg')
            else:
                st.image('images/van_models/Renault/Trafic.jpg')

        elif brand == 'Fiat':
            st.image('images/logos/Fiat.jpg')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Doblo', 'Ducato', 'Fiorino', 'Qubo', 'Scudo', 'Talento'])
            if model is None:
                pass
            elif model =='Doblo':
                st.image('images/van_models/Fiat/Doblo.jpg')
            elif model =='Ducato':
                st.image('images/van_models/Fiat/Ducato.jpg')
            elif model =='Fiorino':
                st.image('images/van_models/Fiat/Fiorino.jpg')
            elif model =='Qubo':
                st.image('images/van_models/Fiat/Qubo.png')
            elif model =='Scudo':
                st.image('images/van_models/Fiat/Scudo.jpg')
            else:
                st.image('images/van_models/Fiat/Talento.jpg')

        elif brand == 'Dacia':
            st.image('images/logos/Dacia.png')
            model = st.radio(':oncoming_automobile: **Model**', [None, 'Dokker'])
            if model is None:
                pass
            else:
                st.image('images/van_models/Dacia/Dokker.jpg')

        if model in model_expensive:
            model_price_value = 2
        elif model in model_medium:
            model_price_value = 1
        else:
            model_price_value = 0

        # rest of the features
        if model is None:
            st.write("Please select a model.")
        else: # show the rest of the features once the user has selected a brand
            age = st.number_input(':calendar: Age of the van', min_value=1, max_value=30, step=1)
            km = st.number_input(':straight_ruler: Mileage in km:', min_value=0, max_value=1000000, step=1)
            
            fuel_input = st.radio(':fuelpump: Fuel:', ['Diesel', 'Gasoline'])
            fuel_dict = {'Diesel': 0, 'Gasoline': 1}
            fuel_value = fuel_dict[fuel_input]

            power_cv = st.number_input(':horse: Horsepower in cv:', min_value=50, step=1)
            consumption = st.number_input(':heavy_dollar_sign: Consumption in L/100km:', min_value=4.0, step=0.1, format="%.1f")

            owners_input = st.radio(':key: Previous owners:', ['One', 'More than one'])
            owners_dict = {'One': 0, 'More than one': 1}
            owners_value = owners_dict[owners_input]

            doors_input = st.radio(':door: Does it have both rear doors:', ['Yes', 'No, only the right one'])
            doors_dict = {'Yes': 1, 'No, only the right one': 0}
            rear_doors = doors_dict[doors_input]

            cargo_input = st.radio(':package: Is it a cargo van?', ['Yes', 'No'])
            cargo_dict = {'Yes': 1, 'No': 0}
            cargo_value = cargo_dict[cargo_input]

            size_big = ['Grand California', 'Sprinter', 'Crafter', 'Transit', 'Movano', 'Ducato', 'Master', 'Boxer', 'NV400', 'Jumper']
            size_medium = ['Marco Polo', 'V', 'T7', 'Multivan', 'Staria', 'T6', 'Zafira', 'Traveller', 'SpaceTourer', 'Vito', 'Proace', 'Custom', 'Primastar', 'Trafic', 'NV300', 'Viano',
                        'Talento', 'H-1', 'Vivaro', 'T5', 'Jumpy', 'Scudo', 'T4']
            size_small = ['T', 'Townstar', 'Caddy', 'Rifter', 'Citan', 'Connect', 'Combo', 'Berlingo', 'Dokker', 'NV200', 'Kangoo', 'Partner', 'Doblo', 'Fiorino', 'Qubo']

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




