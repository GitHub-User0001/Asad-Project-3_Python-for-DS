import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
df = pd.read_csv('loan_approval_data.csv')

# Drop the 'loan_id' column
df = df.drop('loan_id', axis=1)

df.gender.unique()
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

df.married.unique()
df['married'].fillna(df['married'].mode()[0], inplace=True)

df.dependents.unique()
df['dependents'].fillna(df['dependents'].mode()[0], inplace=True)

df.education.unique()
df['education'].fillna(df['education'].mode()[0], inplace=True)

df.self_employed.unique()
df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)

df['loanamount'].unique()
df['loanamount'].fillna(df['loanamount'].mean(), inplace=True)

df.loan_amount_term.unique()
df['loan_amount_term'].fillna(df['loan_amount_term'].mode()[0], inplace=True)

df.credit_history.unique()
df['credit_history'].fillna(df['credit_history'].mode()[0], inplace=True)

df.replace({"loan_status": {'y': 1, 'n': 0}}, inplace=True)

# gender and loan_status

sns.countplot(x='gender', hue='loan_status', data=df)

# married and loan_status

sns.countplot(x='married', hue='loan_status', data=df)

# education and loan_status

sns.countplot(x='education', hue='loan_status', data=df)

# self_employed and loan_status

sns.countplot(x='self_employed', hue='loan_status', data=df)

# credit_history and loan_status

sns.countplot(x='credit_history', hue='loan_status', data=df)

# property_area and loan_status

sns.countplot(x='property_area', hue='loan_status', data=df)

column = ['gender', 'married', 'education', 'self_employed', 'credit_history', 'property_area']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in column:
    df[i] = le.fit_transform(df[i])

# Prepare the feature matrix X and the target variable y
X = df.drop('loan_status', axis=1)
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy/indicator variables
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
