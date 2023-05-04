import pickle
import cv2
import numpy as np
import requests
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from google_images_search import GoogleImagesSearch

features = pickle.load(open('features.pkl','rb'))
label = pickle.load(open('label.pkl','rb'))
predictor1 = pickle.load(open('dtree.pkl','rb'))
predictor2 = pickle.load(open('rf.pkl','rb'))
Brand = pd.read_csv('brand.csv')

def predict(input_data):
    predicted_price = np.round(predictor2.predict(input_data)[0],2)
    st.write('Predicted Price is: $ {}'.format(predicted_price))

# Define a function to fetch the image of the smartwatch from Google search
def fetch_smartwatch_image(brand, component, number = 11, model = ''):
    query = f'{brand} {model} {component}'
    gis = GoogleImagesSearch('AIzaSyCmRvJKwNjCYBTpGEtx7MOT-V3UHgg4JqY','d07e24259591a4e69')
    gis.search({'q': query,'num': number})
    images_url = []
    for result in gis.results():
        images_url.append(result.url)
    return images_url

def picture_adjuster(images_url, i):
    # Download image from URL
    response = requests.get(images_url[i])
    img_data = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if type(img) == type(None) :
        image = picture_adjuster(images_url, i + 5)
    else:
        # Resize image to 300x300
        img_resized = cv2.resize(img, (300, 300))
        # Add 10-pixel white border
        img_with_border = cv2.copyMakeBorder(img_resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[160, 160, 160])
        # Save image with border
        cv2.imwrite('image_with_border.jpg', img_with_border)
        image = Image.open('image_with_border.jpg')
    return image

def plot_chart(brand_name):
    for i in list(Brand.index):
            if Brand["Brand"][i] == brand:
                Brand.loc[i, "No. of Searches"] = Brand.loc[i, "No. of Searches"] + 1
    
    fig, ax = plt.subplots()
    sorted_brand = Brand.sort_values(by = "No. of Searches", ascending= False).head(6)
    ax.bar(sorted_brand["Brand"], sorted_brand["No. of Searches"] , color ='blue',width = 0.5)
    plt.xlabel("Brand Name")
    plt.ylabel("No. of Searches")
    st.pyplot(fig)
    Brand.to_csv('brand.csv')

st.title('SMART WATCH PRICE PREDICTION')
col1, col2, col3 = st.columns(3)
col1 = col1.container()
col2 = col2.container()
col3 = col3.container()
with col1:
    brand = st.selectbox(
        'Brand',
        features['Brand'].unique()
    )
    st.write('')
    operating_system = st.selectbox(
        'Operating System',
        features['Operating System'].unique()
    )
    st.write('')
    display_type = st.selectbox(
        'Display Type',
        features['Display Type'].unique()
    )

with col2:
    display_size = st.slider(
        'Display Size (inches)',
        min_value=0.5,
        max_value=4.0, 
        step=0.01
    )
    resolution = st.selectbox(
        'Resolution',
        features['Resolution'].unique()
    )
    st.write('')
    water_resistance = st.selectbox(
        'Water Resistance (meters)',
        features['Water Resistance (meters)'].unique()
    )

with col3:
    battery_life = st.selectbox(
        'Battery Life (days)',
        features['Battery Life (days)'].unique()
    )
    st.write('Connectivity')
    gps = st.checkbox('GPS')
    nfc = st.checkbox('NFC')
    wifi = st.checkbox('Wi-Fi')
    cellular = st.checkbox('Cellular')

input_data = pd.DataFrame({
        'Brand': [brand],
        'Operating System': [operating_system],
        'Display Type': [display_type],
        'Display Size (inches)': [display_size],
        'Resolution': [resolution],
        'Water Resistance (meters)': [water_resistance],
        'Battery Life (days)': [battery_life],
        'GPS': [gps],
        'NFC': [nfc],
        'Wi-Fi': [wifi],
        'Cellular': [cellular]
})

correction = {True : 'Yes', False : 'No'}
connectivity_list = ["GPS","NFC","Wi-Fi","Cellular"]
for i in connectivity_list:
    input_data[i][0] = correction[input_data[i][0]]

model = []
for i in features.index:
    if features["Brand"][i] == brand and features["Operating System"][i] == operating_system and features["Display Type"][i] == display_type and features["Display Size (inches)"][i] == display_size and features["Resolution"][i] == resolution and features["Water Resistance (meters)"][i] == water_resistance and features["Battery Life (days)"][i] == battery_life and features["GPS"][i] == input_data["GPS"][0] and features["NFC"][i] == input_data["NFC"][0] and features["Wi-Fi"][i] == input_data["Wi-Fi"][0] and features["Cellular"][i] == input_data["Cellular"][0]:
        model.append(features["Model"][i])

for i in input_data.select_dtypes(include = ['object']).columns:
    input_data[i][0] = label[input_data[i][0]]

if st.button('PREDICT'):
    predict(input_data)
    st.subheader('Suggestion For Watches Matching Above Features :-')
    if len(model) > 0:
        for i in model:
            st.write(i)
        images_url = fetch_smartwatch_image(brand,'smartwatch', model = i)
        col1, col2, col3 = st.columns(3)
        with col1 :
            st.image(picture_adjuster(images_url, 0), use_column_width=True)
            st.image(picture_adjuster(images_url, 3), use_column_width=True)
        with col2 :
            st.image(picture_adjuster(images_url, 1), use_column_width=True)
            st.image(picture_adjuster(images_url, 4), use_column_width=True)
        with col3 :
            st.image(picture_adjuster(images_url, 2), use_column_width=True)
            st.image(picture_adjuster(images_url, 5), use_column_width=True)
    else:
        images_url = fetch_smartwatch_image(brand, 'logo', 1)
        col1, col2, col3 = st.columns(3)
        with col2 :
            st.write(" ")
            st.image(images_url, use_column_width=True)
        st.write("We have no smartwatch details in our database that match the defined features for the {} brand.".format(brand))

    st.subheader('Search Trends :-')
    plot_chart(brand_name= brand)
