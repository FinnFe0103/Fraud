import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def load_data(pca_bool=False, smote_bool=False):
    # Load data
    df = pd.read_csv("AlarmGrundlag_ModelParametre_Merged1.1.csv", delimiter=";")
    df = df.drop(columns=["Customer_Refnr", "RUN_DATE", "CASE_CLOSE_DATE", "SCENARIO_NAME", "ALERT_ID", "CASE_ID", "Customer_Risk_Profile_Current"])
    df['CASE_STATUS_CODE'] = df['CASE_STATUS_CODE'].replace({'C': 0, 'R': 1})
    df.dropna(subset=['Customer_Risk_Profile_BeforeAlert'], inplace=True)
    df = pd.get_dummies(df, columns=['Customer_Risk_Profile_BeforeAlert'], prefix='RiskGroup')
    # Replace infinities with NaN for easier handling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with any NaNs that might have been infinities initially
    df.dropna(inplace=True)

    # Log1p transform
    columns_to_log = ['Express_Ratio_SumDKK', 'Express_Ratio_Count', 'MobilePay_Count_DebitCreditRatio', 'MobilePay_Sum_DebitCreditRatio']
    for column in columns_to_log:
        df[column] = np.log1p(df[column])

    # Split data into X and y
    y = df['CASE_STATUS_CODE']
    X = df.drop('CASE_STATUS_CODE', axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check if scaling is needed (for SMOTE or PCA)
    scale_data = smote_bool or pca_bool
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values  # Use values to ensure consistency in data type handling
        X_test_scaled = X_test.values

    # Apply SMOTE
    if smote_bool:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled = X_train_scaled

    # Apply PCA
    if pca_bool:
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_resampled)
        X_test_pca = pca.transform(X_test_scaled)
    else:
        X_train_pca = X_train_resampled
        X_test_pca = X_test_scaled

    # Convert arrays back to DataFrame to maintain original structure
    if not pca_bool:
        if scale_data:
            # Reverse scaling if PCA is not applied
            X_train_pca = scaler.inverse_transform(X_train_pca)
            X_test_pca = scaler.inverse_transform(X_test_pca)
        # Convert numpy arrays back to DataFrames
        X_train_df = pd.DataFrame(X_train_pca, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test_pca, columns=X_test.columns)
    else:
        # If PCA is applied, create DataFrames from PCA output without column names
        X_train_df = pd.DataFrame(X_train_pca)
        X_test_df = pd.DataFrame(X_test_pca)

    return X_train_df, X_test_df, y_train, y_test
