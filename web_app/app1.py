# import the necessary packages
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import datasist as ds
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# setting the title name either by st.title or st.markdown
# st.title ('First Streamlit UI Web Application', ) # title-one 
# st.title ('Analysis and Prediction of the Likelihood of Relapse for Selected Addictive Substances Following Rehabilitation') # changed title-two
col1, col2 = st.columns([5, 1])
col1.markdown("<h2 style='text-align: center;'>Predicting Colorectal Cancer Immunotherapy Outcomes Using Machine Learning Based on MSI and Gene Mutation Signatures Web App</h2>", unsafe_allow_html=True)
image = Image.open (r'jhu_logo.png')

# encoded_info = Image.open(r'pics/encoded_infos.png')
# relapse_image = Image.open(r'pics/relapse.png')

intro_image = Image.open(r'pics/intro.png')
study_image = Image.open(r'pics/this_study.png')
objective_image = Image.open(r'pics/objective.png')

col2.image(image, width=125)

@st.cache_data
def load_data_2022():
    data_2022 = pd.read_csv( 'combined_clinical_mutation_gene_data_new.csv')
    return data_2022

# data_path = load_data()
# load_data_2023 = load_data_2023 ()
load_data_2022 = load_data_2022 ()

# making tabs
tab1, tab2 = st.tabs(["INTRODUCTION","PREDICTION"])

# first tb
with tab1:
    st.header(" Introduction")
    st.subheader("Project Overview")
    
    col1, col2, col3 = st.columns(3)

    col1.image(intro_image, use_container_width=True)
    col2.image(study_image, use_container_width=True)
    col3.image(objective_image, use_container_width=True)

with tab2:


    # --- Patient id selection ---
    # st.title("Colorectal Cancer PFS Prediction")
    col1, col2, col3 = st.columns(3)

    patient_ids = load_data_2022["PATIENT_ID"].unique()
    selected_patient_id = col2.selectbox("## Select Patient ID", patient_ids)

    # filter the selected patient's data
    patient_data = load_data_2022[load_data_2022["PATIENT_ID"] == selected_patient_id].iloc[0]


    # --- numerical feature Inputs ---
    # get feature names
    numerical_features = [
        'AGE', 'TMB', 'PDCD1', 'CTLA4', 'CD274',
        'KRAS_MUTATED', 'BRAF_MUTATED', 'NRAS_MUTATED', 'MSI_BINARY','PFS_MONTHS'
    ]
    categorical_features = [
        'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 
        'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE'
    ]
    
    st.subheader("🩺 Clinical Information")

    clinical_features = ['AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE']
    clinical_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(clinical_features):
        col = cols[i % 3]
        with col:
            if feature in load_data_2022.columns:
                if load_data_2022[feature].dtype == object:
                    options = sorted(load_data_2022[feature].dropna().unique())
                    val = patient_data[feature]
                    clinical_inputs[feature] = st.selectbox(f"{feature}", options, index=options.index(val))
                else:
                    val = float(patient_data[feature])
                    clinical_inputs[feature] = st.number_input(f"{feature}", value=val)

    # mutation feature grouping
    st.subheader(" Mutation Information")

    mutation_features = ['KRAS_MUTATED', 'BRAF_MUTATED', 'NRAS_MUTATED', 'MSI_BINARY', 'TMB']
    mutation_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(mutation_features):
        col = cols[i % 3]
        with col:
            val = float(patient_data[feature])
            mutation_inputs[feature] = st.number_input(f"{feature}", value=val)

    # gene expresion feature grouping
    st.subheader(" Gene Expression Information")

    expression_features = ['PDCD1', 'CTLA4', 'CD274']
    expression_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(expression_features):
        col = cols[i % 3]
        with col:
            val = float(patient_data[feature])
            expression_inputs[feature] = st.number_input(f"{feature}", value=val)

    # PFS month input
    st.subheader(" Survival Time Input")

    with st.columns(1)[0]:
        pfs_months = st.number_input("PFS_MONTHS", value=float(patient_data.get('PFS_MONTHS', 0.0)))

    # combining all the input into the dictionary
    user_inputs = {
        **clinical_inputs,
        **mutation_inputs,
        **expression_inputs,
        'PFS_MONTHS': pfs_months
    }

    st.markdown("---")


        # load the preprocessor pkl ( encoder and scaling weight saved during trianing )
    preprocessor = joblib.load("preprocessor_pipeline.pkl")
    # conver to df
    user_df = pd.DataFrame([user_inputs])

    # apply the preprocessor ( transform the data)
    X_user_transformed = preprocessor.transform(user_df)
    # get one-hot encoded column names from the preprocessor
    cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_columns = numerical_features + list(cat_columns)

    # convert back to DataFrame with column names
    X_user_df = pd.DataFrame(X_user_transformed, columns=final_columns)
    # st.dataframe(X_user_df)
    # load the model
    model = joblib.load("extra_trees_classifier.pkl")
    if st.button("Predict"):
        prediction = model.predict(X_user_transformed)[0]
        prediction_proba = model.predict_proba(X_user_transformed)[0]

        # display prediction text
        st.subheader(" Prediction Result")
        # st.write(f"**Predicted  Status:** {'High Risk Relapser' if prediction == 1 else 'Low Risk Relapser'}")
        st.write(f"**Predicted Progression Status:** {'Progression' if prediction == 1 else 'Censored'}")

        # donut chart for showing the percentage
        labels = ['Censored', 'Progression']
        sizes = [prediction_proba[0], prediction_proba[1]]
        colors = ['#4CAF50', '#F44336']  # Green for censored, red for progression

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.6)  # this creates the "donut"
        )

        ax.axis('equal')  # Equal aspect ratio ensures a circular chart
        plt.setp(autotexts, size=15, weight='bold')
        col1, col2, col3 = st.columns(3)
        col2.pyplot(fig)
