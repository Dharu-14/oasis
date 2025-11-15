import pandas as pd
import numpy as np
df1 = pd.read_csv("data1.csv")
df2 = pd.read_csv("data2.csv")
print("Dataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)
def clean_dataset(df):
    print("\n==============================")
    print("Cleaning New Dataset")
    print("==============================")
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    for column in df.columns:
        if df[column].dtype == 'object':
            # For categorical/text columns → fill with mode
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            # For numeric columns → fill with median
            df[column] = df[column].fillna(df[column].median())

    print("\nMissing values handled successfully.")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicates Removed: {before - after}")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.lower()

    print("Standardization complete.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"{col}: {outliers} outliers detected")
        # Cap outliers within the bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                           np.where(df[col] > upper_bound, upper_bound, df[col]))

    print(" Dataset cleaned successfully.\n")
    return df
clean_df1 = clean_dataset(df1)
clean_df2 = clean_dataset(df2)
combined_df = pd.concat([clean_df1, clean_df2], ignore_index=True)
print("Combined Dataset Shape:", combined_df.shape)
clean_df1.to_csv("cleaned_data1.csv", index=False)
clean_df2.to_csv("cleaned_data2.csv", index=False)
combined_df.to_csv("combined_cleaned_data.csv", index=False)
print("\n All cleaning tasks complete.")
print("Saved as: cleaned_data1.csv, cleaned_data2.csv, and combined_cleaned_data.csv")


