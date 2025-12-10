import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from lightgbm import LGBMClassifier

# Page Configuration
st.set_page_config(
    page_title="Chassis Material Rating Predictor",
    layout="wide"
)


MATERIAL_PROPERTIES = {
    'Steel': {
        'E': 207000,
        'G': 79000,
        'mu': 0.30,
        'Ro': 7860,
        'Su_range': '250-1200 MPa',
        'Sy_range': '150-1000 MPa'
    },
    'Aluminium': {
        'E': 71000,
        'G': 26000,
        'mu': 0.33,
        'Ro': 2700,
        'Su_range': '90-600 MPa',
        'Sy_range': '35-550 MPa'
    },
    'Copper Alloy': {
        'E': 120000,
        'G': 44000,
        'mu': 0.34,
        'Ro': 8900,
        'Su_range': '200-1100 MPa',
        'Sy_range': '70-800 MPa'
    },
    'Iron': {
        'E': 170000,
        'G': 65000,
        'mu': 0.29,
        'Ro': 7200,
        'Su_range': '150-500 MPa',
        'Sy_range': '100-350 MPa'
    },
    # 'Titanium': {
    #     'E': 110000,
    #     'G': 42000,
    #     'mu': 0.34,
    #     'Ro': 4500,
    #     'Su_range': '240-1400 MPa',
    #     'Sy_range': '170-1100 MPa'
    # },
    'Other': {
        'E': 150000,
        'G': 60000,
        'mu': 0.30,
        'Ro': 5000,
        'Su_range': '100-1000 MPa',
        'Sy_range': '50-800 MPa'
    }
}


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("Model\chassis_rating_lgbm.pkl")
        scaler = joblib.load('Model\scaler1.pkl')
        label_encoder = joblib.load('Model\label_encoder.pkl')
        material_db = pd.read_csv('Data\material_database.csv')
        return model, scaler, label_encoder, material_db
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please run the training notebook first!")
        st.stop()

model, scaler, label_encoder, material_db = load_artifacts()
# le_material

st.title("🚙Chassis Material Rating Predictor")
st.subheader("Material selection assistant for automotive chassis design")
 

# st.info("""
# **📚 Based on Research:**
# - Nawale, P., et al. (2023). Design automation and CAD customization of an EV chassis. *Journal of Physics: Conference Series*, 2601, 012014.
# - Desai, M., et al. (2019). Optimal Material Selection on Designed Chassis. Springer.
# - Khakurel, H., et al. (2021). Machine learning assisted prediction of Young's modulus. *Scientific Reports*.
# """)


st.sidebar.title("Prediction Settings")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select prediction mode:",
    ["Mode 1: Material Type Selection", "Mode 2: Custom Material Input"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Tool")
st.sidebar.info("""
**Rating Scale (1-5):**
- **1**: Poor performance
- **2**: Below average
- **3**: Average
- **4**: Good performance
- **5**: Excellent performance

""")


if "Mode 1" in mode:
    st.header("Mode 1: Material Type Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Step 1: Select material type")
        
        material_type_m1 = st.selectbox(
            "Choose Material Category:",
            ['Steel', 'Aluminium', 'Iron', 'Copper Alloy', 'Other'],
            key="material_type_mode1"
        )
        
        material_type_encoded_m1 = label_encoder.transform([material_type_m1])[0]
        typical_props_m1 = MATERIAL_PROPERTIES[material_type_m1]
        
        st.info(f"**Selected:** {material_type_m1} - Typical elastic properties will be auto-filled based on this selection.")
        
        st.markdown("---")
        st.subheader("Step 2: Input strength properties")
        
        st.caption(f"Typical range for {material_type_m1}:")
        st.caption(f"   • Su: {typical_props_m1['Su_range']}")
        st.caption(f"   • Sy: {typical_props_m1['Sy_range']}")
        
        col_su, col_sy = st.columns(2)
        
        with col_su:
            Su_m1 = st.number_input(
                "Ultimate Tensile Strength (Su) [MPa]",
                min_value=0.0,
                value=400.0,
                step=10.0,
                key="su_mode1"
            )
        
        with col_sy:
            Sy_m1 = st.number_input(
                "Yield Strength (Sy) [MPa]",
                min_value=0.0,
                value=300.0,
                step=10.0,
                key="sy_mode1"
            )
        
        st.markdown("---")
        st.subheader("Auto-filled properties (typical values)")
        
        # Allow user to choose display units for E/G (model always uses MPa)
        display_moduli_unit_m1 = st.selectbox(
            "Display units for elastic moduli",
            ["MPa", "GPa"],
            index=0,
            help="Choose how E and G are shown. Values are converted internally to MPa for the model."
        )

        col_e, col_g, col_mu, col_ro = st.columns(4)

        with col_e:
            if display_moduli_unit_m1 == "GPa":
                st.metric("Young's Modulus (E)", f"{typical_props_m1['E']/1000:,.3f} GPa")
            else:
                st.metric("Young's Modulus (E)", f"{typical_props_m1['E']:,.0f} MPa")
        with col_g:
            if display_moduli_unit_m1 == "GPa":
                st.metric("Shear Modulus (G)", f"{typical_props_m1['G']/1000:,.3f} GPa")
            else:
                st.metric("Shear Modulus (G)", f"{typical_props_m1['G']:,.0f} MPa")
        with col_mu:
            st.metric("Poisson's Ratio (μ)", f"{typical_props_m1['mu']:.2f}")
        with col_ro:
            st.metric("Density (ρ)", f"{typical_props_m1['Ro']:,.0f} kg/m³")
        
        X_input_m1 = np.array([[
            Su_m1,
            Sy_m1,
            typical_props_m1['E'],
            typical_props_m1['G'],
            typical_props_m1['mu'],
            typical_props_m1['Ro'],
            material_type_encoded_m1
        ]])
    
    with col2:
        st.subheader("Prediction result")
        
        if st.button("🚀 PREDICT RATING", type="primary", use_container_width=True, key="predict_mode1"):
            # Model prediction
            X_scaled_m1 = scaler.transform(X_input_m1)
            predicted_rating_m1 = model.predict(X_scaled_m1)[0]
            prediction_proba_m1 = model.predict_proba(X_scaled_m1)[0]

            st.success(f"### Rating: {predicted_rating_m1} / 5")
            
            st.subheader("Confidence distribution")
            
            fig_m1 = go.Figure(data=[
                go.Bar(
                    x=[f"Rating {i}" for i in range(1, 6)],
                    y=prediction_proba_m1 * 100,
                    marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff', '#9d4edd'],
                    text=[f"{p*100:.1f}%" for p in prediction_proba_m1],
                    textposition='auto',
                )
            ])
            
            fig_m1.update_layout(
                title="Prediction Confidence per Rating",
                xaxis_title="Rating",
                yaxis_title="Confidence (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_m1, use_container_width=True)
            
            st.subheader("Interpretation")
            
            if predicted_rating_m1 == 5:
                st.success("Excellent choice. This material configuration has very good properties for chassis applications.")
            elif predicted_rating_m1 == 4:
                st.success("Good choice. This material configuration performs well for chassis applications.")
            elif predicted_rating_m1 == 3:
                st.warning("Average performance. Consider adjusting Su/Sy or selecting a different material type.")
            elif predicted_rating_m1 == 2:
                st.warning("Below average. This configuration may not meet chassis requirements.")
            else:
                st.error("Poor performance. Not recommended for chassis applications.")
            
            


else:
    st.header("Mode 2: Custom material properties input")
	
    st.info("Advanced mode: manually input all mechanical properties for custom materials or experimental alloys.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input material properties")
        

        material_type = st.selectbox(
            "Material Type (Reference)",
            ['Steel', 'Aluminium', 'Iron', 'Copper Alloy', 'Other'],
            help="Select the closest material category"
        )
        
        material_type_encoded = label_encoder.transform([material_type])[0]
        
        st.markdown("---")
        
        # Strength Properties
        st.markdown("#### Strength properties")
        col_su, col_sy = st.columns(2)
        
        with col_su:
            Su_custom = st.number_input(
                "Ultimate Tensile Strength (Su) [MPa]",
                min_value=0.0,
                value=400.0,
                step=10.0,
                key="su_custom"
            )
        
        with col_sy:
            Sy_custom = st.number_input(
                "Yield Strength (Sy) [MPa]",
                min_value=0.0,
                value=300.0,
                step=10.0,
                key="sy_custom"
            )
        
        # Elastic Properties
        st.markdown("#### Elastic properties")

        # Unit selector for elastic moduli (user-friendly: allow MPa or GPa)
        moduli_unit = st.selectbox(
            "Elastic Moduli Units",
            ["MPa", "GPa"],
            index=0,
            help="Select unit for E and G. Inputs will be converted to MPa for the model."
        )

        col_e, col_g = st.columns(2)

        with col_e:
            E_custom = st.number_input(
                "Young's Modulus (E)",
                min_value=0.0,
                value=207000.0 if moduli_unit == "MPa" else 207.0,
                step=1000.0 if moduli_unit == "MPa" else 0.1,
                key="e_custom",
                help="Measure of material stiffness"
            )
        
        with col_g:
            G_custom = st.number_input(
                "Shear Modulus (G)",
                min_value=0.0,
                value=79000.0 if moduli_unit == "MPa" else 79.0,
                step=1000.0 if moduli_unit == "MPa" else 0.1,
                key="g_custom",
                help="Measure of material rigidity under shear"
            )
        
        # Other Properties
        st.markdown("#### Other properties")
        col_mu, col_ro = st.columns(2)
        
        with col_mu:
            mu_custom = st.number_input(
                "Poisson's Ratio (μ)",
                min_value=0.0,
                max_value=0.5,
                value=0.3,
                step=0.01,
                key="mu_custom",
                help="Ratio of transverse to axial strain"
            )
        
        with col_ro:
            Ro_custom = st.number_input(
                "Density (ρ) [kg/m³]",
                min_value=0.0,
                value=7860.0,
                step=10.0,
                key="ro_custom",
                help="Material mass per unit volume"
            )
        
        # Convert moduli to MPa for the model if necessary
        if moduli_unit == "GPa":
            E_for_model = float(E_custom) * 1000.0
            G_for_model = float(G_custom) * 1000.0
        else:
            E_for_model = float(E_custom)
            G_for_model = float(G_custom)

        # Prepare input (all moduli in MPa)
        X_input_custom = np.array([[
            Su_custom,
            Sy_custom,
            E_for_model,
            G_for_model,
            mu_custom,
            Ro_custom,
            material_type_encoded
        ]])
    
    with col2:
        st.subheader("Prediction result")
        # Input-range warnings removed per user request; predict regardless of dataset bounds
        if st.button("🚀 PREDICT RATING", type="primary", use_container_width=True, key="predict_custom"):
            # Scale input
            X_scaled_custom = scaler.transform(X_input_custom)

            # Predict
            predicted_rating_custom = model.predict(X_scaled_custom)[0]
            prediction_proba_custom = model.predict_proba(X_scaled_custom)[0]

            # Display result
            st.markdown(f'<div class="prediction-box">Rating: {predicted_rating_custom} / 5</div>', unsafe_allow_html=True)


            # Confidence chart
            st.subheader("Confidence distribution")

            fig_custom = go.Figure(data=[
                go.Bar(
                    x=[f"Rating {i}" for i in range(1, 6)],
                    y=prediction_proba_custom * 100,
                    marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff', '#9d4edd'],
                    text=[f"{p*100:.1f}%" for p in prediction_proba_custom],
                    textposition='auto',
                )
            ])

            fig_custom.update_layout(
                title="Prediction Confidence per Rating",
                xaxis_title="Rating",
                yaxis_title="Confidence (%)",
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig_custom, use_container_width=True)

            # Property Summary
            st.subheader("Input summary")
            # Format E/G according to selected moduli unit (MPa or GPa)
            if moduli_unit == "GPa":
                display_E = f"{E_custom:,.3f} GPa"
                display_G = f"{G_custom:,.3f} GPa"
            else:
                display_E = f"{E_custom:,.0f} MPa"
                display_G = f"{G_custom:,.0f} MPa"

            summary_data = {
                "Property": ["Su", "Sy", "E", "G", "μ", "ρ", "Type"],
                "Value": [
                    f"{Su_custom:.1f} MPa",
                    f"{Sy_custom:.1f} MPa",
                    display_E,
                    display_G,
                    f"{mu_custom:.2f}",
                    f"{Ro_custom:,.0f} kg/m³",
                    material_type
                ]
            }
            st.table(pd.DataFrame(summary_data))
                


st.markdown("---")
st.markdown("### References")
st.markdown("""
1. Nawale, P., Kanade, A., Nannaware, B., Sagalgile, A., Chougule, N., & Patange, A. (2023). 
   Design automation and CAD customization of an EV chassis. *Journal of Physics: Conference Series*, 2601, 012014. 
   https://doi.org/10.1088/1742-6596/2601/1/012014
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <b>Academic project</b> | Built with Streamlit and scikit-learn<br>
    Model: Light GBM | Accuracy: ~97%<br>
    Based on 1500+ materials from ANSI, ISO, JIS, BS, NF standards
</div>
""", unsafe_allow_html=True)