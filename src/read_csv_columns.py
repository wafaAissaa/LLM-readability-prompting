import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("Qualtrics_Annotations_B.csv")

# Print the column names
print("Column names in Qualtrics_Annotations_B.csv:")
print(df.columns.tolist())