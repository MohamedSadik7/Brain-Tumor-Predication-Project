
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import joblib
import numpy as np



# Home Page
st.set_page_config(page_title="Brain Tumor Data Dashboard", layout="wide", page_icon="🧠")
st.title("Brain Tumor Data Dashboard")


# Load data
df = pd.read_csv("cleaned_brain_tumor_data.csv")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 About the project","📄 Dataset Overview", "📊 Analysis", "🔢 Prediction"])

with tab1:
    
    with st.expander("Intro", expanded=True):
        st.markdown("<h1 style='color:#2E86C1; font-weight:700;'>Brain Tumor Type Prediction Project</h1>", unsafe_allow_html=True)
        st.markdown(
        """
        <p style='font-size:16px; line-height:1.6; text-align: justify;'>
        This project demonstrates the application of advanced machine learning techniques to predict brain tumor types (Malignant vs. Benign) using a synthetic dataset containing clinical, demographic, and diagnostic features. The purpose of this work is to showcase the concept and methodology rather than to provide a model for real-world medical use. 
        It is not intended for clinical decision-making or actual patient diagnosis.
        </p>
        """, unsafe_allow_html=True)
        st.image("brain.jpg", use_container_width=True)
        
        
       

with tab2:
    
    
    # KPIs with styled boxes
    st.subheader("📌 Data Parameters")
    k1, k2, k3, k4 = st.columns(4)


    total_patients = len(df)
    malignant_rate = round((df['Tumor_Type'] == 'Malignant').mean() * 100, 2)
    avg_age = round(df['Age'].mean(), 2)
    avg_survival_rate = round(df['Survival_Rate'].mean(), 2)

    with k1:
        st.markdown(
            f"""<div style="background-color:#1f77b4;padding:15px;border-radius:10px;text-align:center">
            <h5 style="color:white;">👥 Total_patients</h5>
            <h3 style="color:white;">{total_patients:,}</h3>
            </div>""",
            unsafe_allow_html=True)

    with k2:
        st.markdown(
            f"""<div style="background-color:#ff7f0e;padding:15px;border-radius:10px;text-align:center">
            <h5 style="color:white;">🧠 Malignant_rate</h5>
            <h3 style="color:white;">{malignant_rate} %</h3>
            </div>""",
            unsafe_allow_html=True)

    with k3:
        st.markdown(
            f"""<div style="background-color:#2ca02c;padding:15px;border-radius:10px;text-align:center">
            <h5 style="color:white;">📅 Avg_age</h5>
            <h3 style="color:white;">{avg_age:,.0f} years</h3>
            </div>""",
            unsafe_allow_html=True)

    with k4:
        st.markdown(
            f"""<div style="background-color:#d62728;padding:15px;border-radius:10px;text-align:center">
            <h5 style="color:white;">💓 Avg_survival_rate</h5>
            <h3 style="color:white;">{avg_survival_rate:,.0f} %</h3>
            </div>""",
            unsafe_allow_html=True)

    
    # Key insights
    st.subheader("💡 Key Insights")
    insight1 = st.columns(1)
    with insight1[0]:
        total_patients = len(df)
        malignant_rate = round((df['Tumor_Type'] == 'Malignant').mean() * 100, 2)
        avg_age = round(df['Age'].mean(), 2)
        avg_survival_rate = round(df['Survival_Rate'].mean(), 2)
        
    st.markdown(f"""
    <div style='background-color:#31333F;padding:15px;border-radius:10px'>
    <ul style="color:white;font-size:15px">
    <li>👥 Dataset contains <b>{total_patients:,} patients</b>.</li>
    <li>🧠 <b>Malignant tumors</b> account for <b>{malignant_rate}%</b> of cases, indicating a balanced distribution with benign cases.</li>
    <li>📅 Average patient age is <b>{avg_age} years</b>.</li>
    <li>💓 Average survival rate is <b>{avg_survival_rate}%</b>, suggesting generally favorable outcomes.</li>
    </ul>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    
# Sidebar checkbox to show full data
    with st.sidebar:
        show_full_data = st.checkbox("📂 Show Full Dataset")

    st.subheader(" Dataset Highlights")
    if show_full_data:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("✅ Check the box in the sidebar to view the dataset.")

    st.markdown("---")


with tab3:
    with st.sidebar:
        selected = option_menu(
            menu_title="Select Analysis Type",
            options=["Univariate &Bivariate Analysis", "Multivariate Analysis"],
            icons=["bar-chart","graph-up"],
            default_index=0,)

    if selected == "Univariate &Bivariate Analysis":

        st.subheader("📈 Univariate Analysis")
    
        # Tumor Types
        st.markdown("### Distribution of Tumor Types")
        fig = px.histogram(df,  x="Tumor_Type",  color="Tumor_Type", title="Distribution of Tumor Types",
         text_auto=True, width=1000,  height=700, color_discrete_map={'Malignant': '#E74C3C', 'Benign': '#27AE60'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight**: The dataset contains an almost equal split between malignant (10,030 cases) and benign (9,971 cases) tumors, with a slight predominance of malignant cases. This balance supports the development of unbiased predictive models.
        """)
        st.markdown("---")
    
        # Tumor Size Distribution Across Tumor Types
        st.markdown("### Tumor Size Distribution Across Tumor Types")
        fig = px.box(df, x="Tumor_Type", y="Tumor_Size", color="Tumor_Type", hover_data=['Age', 'Stage'], title="Tumor Size Distribution Across Tumor Types", color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        fig.update_layout( xaxis_title="Tumor Type",yaxis_title="Tumor Size (cm)", width=1000, height=600, showlegend=False)
        fig.update_xaxes(tickangle=45)

        # Add custom hover template
        fig.update_traces( hovertemplate="<b>%{x}</b><br>" + "Tumor Size: %{y}<br>" + "<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** **Insight:** Malignant and benign tumors have similar median sizes (\~5.27 cm), but benign tumors show greater size variability, with both types reaching sizes close to 10 cm.
        """)
        st.markdown("---")

        
         # Age Distribution by Tumor Type
        st.markdown("### Age Distribution by Tumor Type")
        fig = px.violin(df, x="Tumor_Type", y="Age",color="Tumor_Type",title="Age Distribution by Tumor Type")
        fig.update_layout( xaxis_title="Tumor Type", yaxis_title="Age", title_font_size=18, width=1000, height=600,showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Malignant and benign brain tumors have overlapping age distributions (median age both around 50 years), but malignant tumors show a slightly broader range and higher lower quartile, with both types seen in patients from early adulthood to 80+ years old.
        """)
        st.markdown("---")
        
    
        # Tumor Type Distribution by Gender
        st.markdown("### Tumor Type Distribution by Gender")
        fig = px.histogram(df, x="Tumor_Type", color="Gender", title="Tumor Type Distribution by Gender", barmode='group',  text_auto=True)  
        fig.update_layout( xaxis_title="Tumor Type", yaxis_title="Count", width=1000, height=800)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Both malignant and benign brain tumors occur equally in males and females, with nearly identical counts for each gender and tumor type. This indicates there is no significant gender difference in the occurrence of either tumor type in this dataset.
        """)
        st.markdown("---")
    
        # Survival Rate Distribution by Tumor Type
        st.markdown("### Survival Rate Distribution by Tumor Type")
        fig = px.histogram(df,  x="Survival_Rate", color="Tumor_Type",color_discrete_map={'Malignant': '#E74C3C', 'Benign': '#27AE60'}, opacity=0.8,  nbins=40, histnorm='probability density', title="Survival Rate Distribution by Tumor Type", barmode='overlay')
        fig.update_layout(  xaxis_title="Survival Rate", yaxis_title="Density", width=1000, height=600, legend=dict(title="Tumor Type"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Benign tumors are mostly found at higher survival rates, while malignant tumors are spread more evenly across the full range, indicating worse and more variable outcomes for malignant cases.
        """)
        st.markdown("---")

        
        # MRI Results Distribution by Tumor Type
        st.markdown("### MRI Results Distribution by Tumor Type")
        fig = px.histogram(df, x="Tumor_Type",  color="MRI_Result", title="MRI Results Distribution by Tumor Type", barmode='group')  # Side-by-side bars
        fig.update_layout( xaxis_title="Tumor Type", yaxis_title="Count", width=1000, height=600)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Malignant and benign tumor types exhibit almost identical distributions of MRI results, with positive and negative outcomes occurring at nearly equal rates within each group. This parity suggests that, in this dataset, MRI positivity is not strongly associated with tumor type
        """)
        st.markdown("---")

        
        # Chemotherapy Distribution Across Tumor Types
        st.markdown("### Chemotherapy Distribution Across Tumor Types")
        treatments = ["Chemotherapy"]

        for treatment in treatments:
            fig = px.histogram(df, 
                      x="Tumor_Type",
                      color=treatment,
                      title=f"{treatment.replace('_', ' ')} Distribution Across Tumor Types",
                      text_auto=True,
                      width=1000,
                      height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The distribution of chemotherapy treatment is almost identical between malignant and benign tumor types. In both groups, about half of the patients received chemotherapy and half did not, indicating that the decision to administer chemotherapy is not strongly influenced by tumor type alone in this dataset.
        """)
        st.markdown("---")
        
        # Radiation Distribution Across Tumor Types
        st.markdown("### Radiation Distribution Across Tumor Types")
        treatments = ["Radiation_Treatment"]

        for treatment in treatments:
            fig = px.histogram(df, 
                      x="Tumor_Type",
                      color=treatment,
                      title=f"{treatment.replace('_', ' ')} Distribution Across Tumor Types",
                      text_auto=True,
                      width=1000,
                      height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Malignant and benign brain tumors have nearly identical MRI result distributions, with positive and negative findings occurring at almost equal rates for both types. This indicates that, in this dataset, an MRI result alone does not distinguish between malignant and benign tumors
        """)
        st.markdown("---")
        
        # Surgery_Performed Distribution Across Tumor Types
        st.markdown("### Surgery_Performed Distribution Across Tumor Types")
        treatments = ["Surgery_Performed"]

        for treatment in treatments:
            fig = px.histogram(df, 
                      x="Tumor_Type",
                      color=treatment,
                      title=f"{treatment.replace('_', ' ')} Distribution Across Tumor Types",
                      text_auto=True,
                      width=1000,
                      height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The rates of surgery performed are virtually identical between malignant and benign tumor types. In both groups, about half of the patients underwent surgery and half did not, indicating that surgery rates are not significantly influenced by whether a brain tumor is malignant or benign in this dataset.
        """)
        st.markdown("---")

    
    elif selected == "Multivariate Analysis":
        st.subheader("📉 Multivariate Analysis")
        
         # Tumor Size vs. Age (Grouped by Tumor Type)
        st.markdown("### Tumor Size vs. Age (Grouped by Tumor Type)")
        fig = px.scatter(df,  x="Age",  y="Tumor_Size", color_discrete_map={'Malignant': '#E74C3C', 'Benign': '#27AE60'} ,color="Tumor_Type",
                title="Tumor Size vs. Age (Grouped by Tumor Type)", labels={"Age": "Age", "Tumor_Size": "Tumor Size"}, opacity=0.7,
                width=1000,  height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        
         # Tumor Growth Rate vs. Survival Rate by Tumor Type
        st.markdown("### Tumor Growth Rate vs. Survival Rate by Tumor Type")
        fig = px.scatter(df, x="Tumor_Growth_Rate", y="Survival_Rate",color="Tumor_Type",color_discrete_map={'Malignant': '#E74C3C', 'Benign': '#27AE60'} ,title="Tumor Growth Rate vs. Survival Rate by Tumor Type",opacity=0.7)
        fig.update_layout( xaxis_title="Tumor Growth Rate",  yaxis_title="Survival Rate", width=1000, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        
        # Correlation Heatmap of Numerical Features
        st.markdown("### Correlation Heatmap of Numerical Features)")
        numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
        print("Numerical Features:", numerical_features)
        # Calculate correlation matrix
        corr_matrix = df[numerical_features].corr()
        # Create heatmap
        fig = px.imshow(corr_matrix,text_auto=".2f",color_continuous_scale="RdBu_r",title="Correlation Heatmap of Numerical Features",
                width=1000,height=700)
        # Update layout for better appearance
        fig.update_layout(xaxis_title="Features", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The correlation heatmap of numerical features shows that none of the variables—including age, tumor size, survival rate, and tumor growth rate—are strongly correlated with each other (all correlation coefficients are close to zero). This indicates that, in this dataset, each numerical feature provides largely independent information about the patients and tumor characteristics.
        """)
        st.markdown("---")
        
################

with tab4:
    st.header("🧠 Brain Tumor Type Prediction")
    st.markdown("### Enter Patient Information")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .uncertain {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)

    # Define features
    selected_features = ["Age", "Tumor_Size", "Tumor_Growth_Rate", "Survival_Rate",
                        'Gender', 'Location', 'Family_History', 'MRI_Result']

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female'])

    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=15.0, value=5.0, step=0.01)
    with col2:
        tumor_growth_rate = st.number_input("Tumor Growth Rate", min_value=0.0, max_value=5.0, value=1.0, step=0.01)

    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        survival_rate = st.number_input("Survival Rate (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    with col2:
        location = st.selectbox("Tumor Location", ['Frontal', 'Parietal', 'Temporal', 'Occipital'])

    # Row 4
    col1, col2 = st.columns(2)
    with col1:
        mri_result = st.selectbox("MRI Result", ['Positive', 'Negative'])
    with col2:
        family_history = st.selectbox("Family History", ['Yes', 'No'])

    # Prediction button and results
    if st.button("🎯 Predict Tumor Type", type="primary", use_container_width=True):
        # Let's try mapping to the exact categories your model was trained with
        # Based on your original code, these were the categories:
        location_map = {
            'Frontal': 'Brain',
            'Parietal': 'Lung', 
            'Temporal': 'Breast',
            'Occipital': 'Colon'
        }
        
        mri_map = {
            'Positive': 'Abnormal',
            'Negative': 'Normal'
        }
        
        # Create input dataframe with mapped values
        input_df = pd.DataFrame([{
            "Age": age,
            "Tumor_Size": tumor_size,
            "Tumor_Growth_Rate": tumor_growth_rate,
            "Survival_Rate": survival_rate,
            "Gender": gender,
            "Location": location_map[location],  # Map to original training categories
            "Family_History": family_history,
            "MRI_Result": mri_map[mri_result]    # Map to original training categories
        }], columns=selected_features)
        
        try:
            # Load trained model pipeline
            model = joblib.load("best_tumor_model_pipeline.pkl")
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # If prediction is integer (encoded), convert to label
            if isinstance(prediction, (np.integer, int)):
                tumor_type = model.classes_[prediction]
            else:
                tumor_type = prediction
            
            # Try with different test values to see if model works
            st.write("**Testing with different values:**")
            test_cases = [
                {"Age": 25, "Tumor_Size": 1.0, "Tumor_Growth_Rate": 0.5, "Survival_Rate": 95.0, "Gender": "Female", "Location": "Brain", "Family_History": "No", "MRI_Result": "Normal"},
                {"Age": 70, "Tumor_Size": 10.0, "Tumor_Growth_Rate": 4.0, "Survival_Rate": 30.0, "Gender": "Male", "Location": "Lung", "Family_History": "Yes", "MRI_Result": "Abnormal"},
                {"Age": 50, "Tumor_Size": 5.0, "Tumor_Growth_Rate": 2.0, "Survival_Rate": 70.0, "Gender": "Female", "Location": "Breast", "Family_History": "No", "MRI_Result": "Normal"}
            ]
            
            for i, test_case in enumerate(test_cases):
                test_df = pd.DataFrame([test_case], columns=selected_features)
                test_pred = model.predict(test_df)[0]
                st.write(f"Test {i+1}: {test_case} → {test_pred}")
            
            # Display result with custom styling
            if tumor_type == "Benign":
                st.markdown(f"""
                <div class="prediction-result benign">
                    🟢 Prediction: {tumor_type}
                    <br><small>The tumor is predicted to be benign (non-cancerous)</small>
                </div>
                """, unsafe_allow_html=True)
            elif tumor_type == "Malignant":
                st.markdown(f"""
                <div class="prediction-result malignant">
                    🔴 Prediction: {tumor_type}
                    <br><small>The tumor is predicted to be malignant (cancerous)</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result uncertain">
                    🟡 Prediction: {tumor_type}
                    <br><small>The tumor classification is uncertain</small>
                </div>
                """, unsafe_allow_html=True)
            
                
        except FileNotFoundError:
            st.error("❌ Model file not found. Please train the model and save it as 'best_tumor_model_pipeline.pkl'.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 10px;'>
        <p><em>⚠️ This is for educational purposes only. Always consult medical professionals for actual diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)
