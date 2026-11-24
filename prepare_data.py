# AI helped generate this code.

import pandas as pd
import yaml

# Load settings
with open("config.yaml") as file:
    config = yaml.safe_load(file)

# The original CSV uses semicolons between columns.
df = pd.read_csv(config["raw_data_url"], sep=";")

# The original 37-column CSV had no missing values or exact duplicate rows.
# Keep all rows and select the two inputs and the outcome.
columns = ["Admission grade", "Previous qualification (grade)", "Target"]
df[columns].to_csv(config["clean_data_path"], index=False)

print(f"Saved {len(df)} rows to {config['clean_data_path']}.")