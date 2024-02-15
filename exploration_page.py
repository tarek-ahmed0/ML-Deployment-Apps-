import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def shorten_categories(categories, cutoff) :
    categorical_dict = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff :
            categorical_dict[categories.index[i]] = categories.index[i]
        else:
            categorical_dict[categories.index[i]] = 'Other'
    return categorical_dict


def clean_exp(x) :
    if x == 'Less than 1 year' :
        return 0.5
    if x == 'More than 50 years' :
        return 50
    
    return float(x)


def clean_edu(x) :
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x:
        return 'Professional degree'
    return 'Less than a Bachelors'

@st.cache
def load_data() :
    df = pd.read_csv(r"D:\Machine Learning Models\AI Engineer Salary Predicion ML App\Dataset\survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df.rename({"ConvertedComp" : "Salary"}, axis = 1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis = 1)
    country_map = shorten_categories(df["Country"].value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]
    df = df[df["Country"] != 'Other']
    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_exp)
    df["EdLevel"] = df["EdLevel"].apply(clean_edu)
    return df

df = load_data()


def show_exploration_page():
    st.title("Explore AI Engineers Salaries")
    st.write(""" ### Stack Over Flow Survey 2020 """)
    st.subheader("Salaries Over Countries")
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending = True)
    st.bar_chart(data)