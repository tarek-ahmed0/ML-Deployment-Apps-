import streamlit as st
import numpy as np 
import pickle


def load_data() :
    with open(r"D:\Machine Learning Models\AI Engineer Salary Predicion ML App\saved_steps.pkl", 'rb') as file :
       data = pickle.load(file)
    return data

data = load_data()

regressor = data['model']
le_country = data["country_encode"]
le_education = data["education_encod"]


def prediction_page() :
    st.title("AI Engineer Salary Prediction :)")
    st.write("""#### Please Provide Us With Some Information""")
    
    countries = (
    "United States",
    "India",
    "United Kingdom",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden"
    )


    education = (
    "Less than a Bechelors",
    "Bachelor’s degree",
    "Master’s degree",
    "Post grad"
    )

    country_sb = st.selectbox("Country", countries)
    education_sb = st.selectbox("Education Level", education)
    exp_years = st.slider("Years of Experience", 0, 50, 3)
    calc_button = st.button("Calculate Estimated Salary")
    if calc_button:
        x = np.array([[country_sb, education_sb, exp_years]])
        x[:, 0] = le_country.fit_transform(x[:, 0])
        x[:, 1] = le_education.fit_transform(x[:, 1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The Estimated Salary Is ${salary[0]:.2f}")
