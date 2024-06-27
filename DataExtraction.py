import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv('customer_churn_prediction_sample.csv')

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['SubscriptionDuration', 'MonthlySpend', 'TotalSpend', 'Age', 'Tenure', 'ServiceUsage', 'CustomerSupportInteractions']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split data into training and testing sets
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
