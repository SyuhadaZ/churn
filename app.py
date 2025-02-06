import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model, can be changed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Use Case: Predicting Customer Churn (Improved with more realistic data handling)

# 1. Data Preparation (Simulated, but more realistic handling)
np.random.seed(42)
n_samples = 200
data = {
    'Age': np.random.randint(20, 65, n_samples),
    'Tenure': np.random.randint(0, 10, n_samples),
    'MonthlySpend': np.random.randint(50, 200, n_samples),
    'NumSupportTickets': np.random.randint(0, 5, n_samples),
    'ContractType': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples), # New categorical feature with missing values
    'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Imbalanced dataset
}
df = pd.DataFrame(data)


# Introduce some missing values (more realistic)
df.loc[df.sample(frac=0.1).index, 'InternetService'] = np.nan  # 10% missing values

# 2. Preprocessing
# Handle missing values (using SimpleImputer)
imputer_categorical = SimpleImputer(strategy='most_frequent')
df['InternetService'] = imputer_categorical.fit_transform(df[['InternetService']])

imputer_numerical = SimpleImputer(strategy='mean')
numerical_cols = ['Age', 'Tenure', 'MonthlySpend', 'NumSupportTickets']
df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])



# Encode categorical features
le_contract = LabelEncoder()
df['ContractType'] = le_contract.fit_transform(df['ContractType'])

le_internet = LabelEncoder()
df['InternetService'] = le_internet.fit_transform(df['InternetService'])

# Scale numerical features (important for some models)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# 3. Model Training
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)  # Example model
model.fit(X_train, y_train)


# 4. SHAP Explanation
explainer = shap.TreeExplainer(model)  # For tree-based models
shap_values = explainer.shap_values(X_test)


# 5. Streamlit App
st.title("Customer Churn Prediction Analysis with SHAP")


# --- Global Explanations ---
st.subheader("Global Feature Importance (SHAP Summary Plot)")
st.pyplot(shap.summary_plot(shap_values[1], X_test))

# --- Local Explanations ---

st.subheader("Individual Predictions (SHAP Force Plot)")
customer_index = st.slider("Select a customer (index):", 0, len(X_test) - 1, 0)

st.write(f"**Customer ID:** {customer_index}")

# Display raw data for the selected customer (inverse transform numerical data)
customer_data = X_test.iloc[[customer_index]].copy()
customer_data[numerical_cols] = scaler.inverse_transform(customer_data[numerical_cols]) # Inverse transform
st.write("**Customer Data:**")

# Decode the categorical features for better readability
customer_data['ContractType'] = le_contract.inverse_transform(customer_data['ContractType'])
customer_data['InternetService'] = le_internet.inverse_transform(customer_data['InternetService'])

st.write(customer_data)


st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1][customer_index], X_test.iloc[[customer_index]]))

st.subheader("SHAP Decision Plot")
st.pyplot(shap.decision_plot(explainer.expected_value[1], shap_values[1][customer_index], X_test.iloc[[customer_index]]))

st.subheader("SHAP Dependence Plot")
feature = st.selectbox("Select a feature for dependence plot", X_test.columns)
st.pyplot(shap.dependence_plot(feature, shap_values[1], X_test))


# --- Feature Value Manipulation (Interactive) ---
st.subheader("Interactive Feature Value Manipulation")
manipulated_customer = X_test.iloc[[customer_index]].copy() # Start with a copy

for feature in numerical_cols:
    min_val = X_test[feature].min()
    max_val = X_test[feature].max()
    initial_val = manipulated_customer[feature].values[0] # Get the original scaled value

    # Create a slider for each numerical feature
    scaled_value = st.slider(f"Adjust {feature} (scaled):", min_val, max_val, initial_val)
    manipulated_customer[feature] = scaled_value  # Set the (scaled) value

# Inverse transform numerical data for force plot
manipulated_customer_original_scale = manipulated_customer.copy()
manipulated_customer_original_scale[numerical_cols] = scaler.inverse_transform(manipulated_customer_original_scale[numerical_cols])

st.write("**Manipulated Customer Data (Original Scale):**")
manipulated_customer['ContractType'] = le_contract.inverse_transform(manipulated_customer['ContractType'])
manipulated_customer['InternetService'] = le_internet.inverse_transform(manipulated_customer['InternetService'])

st.write(manipulated_customer)

# Display the force plot with the manipulated values
st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1][customer_index], manipulated_customer))


st.write("Note: This is a simplified example.  For real-world applications, you would use your own data and trained model.")
