import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load('model#gym.pkl')
scaler = joblib.load('scaler#gym.pkl')
columns = joblib.load('columns#gym.pkl')

st.title('GYM PROGRESS PREDICTION 💪')
st.markdown('👉 Please fill the following details')

# Inputs
Age = st.slider('Age', 18, 60, 27)

Gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

Group = st.radio(
    'Group',
    ['Hypertrophy_Only', 'HIIT_Only', 'Concurrent']
)

Duration_Weeks = st.slider(
    'Duration_Weeks',
    1,
    20,
    3
)

Compliance_Rate = st.slider(
    'Compliance_Rate',
    0.40,
    1.20,
    0.60
)

Initial_Body_Fat_Pct = st.slider(
    'Initial_Body_Fat_Pct',
    10.0,
    35.0,
    27.0
)

Final_Body_Fat_Pct = st.slider(
    'Final_Body_Fat_Pct',
    7.0,
    33.6,
    12.5
)

Initial_Lean_Mass_kg = st.slider(
    'Initial_Lean_Mass_kg',
    40.0,
    90.0,
    50.0
)

VO2_Max_Change_Pct = st.slider(
    'VO2_Max_Change_Pct',
    -0.5,
    22.2,
    12.4
)

Dietary_Condition = st.selectbox(
    'Dietary_Condition',
    ['Surplus', 'Deficit', 'Maintenance']
)

# Label encoding mappings
gender_map = {
    'Male': 1,
    'Female': 0
}

group_map = {
    'Concurrent': 0,
    'HIIT_Only': 1,
    'Hypertrophy_Only': 2
}

diet_map = {
    'Deficit': 0,
    'Maintenance': 1,
    'Surplus': 2
}

if st.button('Predict'):

    raw_data = {
        'Age': Age,
        'Gender': gender_map[Gender],
        'Group': group_map[Group],
        'Duration_Weeks': Duration_Weeks,
        'Compliance_Rate': Compliance_Rate,
        'Initial_Body_Fat_Pct': Initial_Body_Fat_Pct,
        'Final_Body_Fat_Pct': Final_Body_Fat_Pct,
        'Initial_Lean_Mass_kg': Initial_Lean_Mass_kg,
        'VO2_Max_Change_Pct': VO2_Max_Change_Pct,
        'Dietary_Condition': diet_map[Dietary_Condition]
    }

    input_df = pd.DataFrame([raw_data])

    st.subheader("Report")
    st.dataframe(input_df)

    try:
    
        prediction = model.predict(input_df)

        if prediction[0] <= Initial_Lean_Mass_kg:

            st.metric(
                label="Predicted Final Muscle Mass",
                value=f"{prediction[0]:.2f} kg"
            )
        elif prediction[0] > Initial_Lean_Mass_kg:
            st.metric(
                label='Well done keep going 🏋️',
                value = f"{prediction[0]:.2f} kg"
            )

        progress_df = pd.DataFrame({
            'stage':['Start', 'Finish'],
            'Mucles':[Initial_Lean_Mass_kg,
                    prediction[0]
                    ]
        })

        st.line_chart(
            progress_df.set_index('stage')
        )

    except Exception as e:
        st.error(f"Error: {e}")


st.sidebar.title("About")

st.sidebar.write(
    "This model predicts Final Muscle Mass using training, body composition, and dietary variables."
)

st.sidebar.image(r'GYM.png', width=200)

