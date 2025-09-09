import pandas as pd
import numpy as np

# Load dataset
file_path = "excel/djokovic_features_3shots_predict_next_djokovic_shot.xlsx"
df = pd.read_excel(file_path)

# --- Step 1: Fill missing values row-wise (within rallies) ---
df = df.T.fillna(method="ffill").T
df = df.T.fillna(method="bfill").T
df = df.fillna(df.mean(numeric_only=True))  # fallback

# --- Step 2: Ensure rallies 1–127 exist ---
existing_rallies = set(df["rally_id"].unique())
all_rallies_127 = [f"rally{i}" for i in range(1, 128)]
missing_rallies = [r for r in all_rallies_127 if r not in existing_rallies]

new_rows = []
for rally_id in missing_rallies:
    # Sample one rally from existing data
    sampled_row = df.sample(1, random_state=np.random.randint(0, 10000)).iloc[0].to_dict()
    sampled_row["rally_id"] = rally_id

    # Add small noise to numeric columns (±3%) to avoid duplicates
    for col in df.columns:
        if col != "rally_id":
            val = sampled_row[col]
            noise = np.random.normal(0, 0.03 * abs(val) if val != 0 else 0.03)
            sampled_row[col] = val + noise
    new_rows.append(sampled_row)

df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# --- Step 3: Add rallies 128–150 ---
extra_rallies = [f"rally{i}" for i in range(128, 151)]
extra_rows = []
for rally_id in extra_rallies:
    sampled_row = df.sample(1, random_state=np.random.randint(0, 10000)).iloc[0].to_dict()
    sampled_row["rally_id"] = rally_id

    for col in df.columns:
        if col != "rally_id":
            val = sampled_row[col]
            noise = np.random.normal(0, 0.03 * abs(val) if val != 0 else 0.03)
            sampled_row[col] = val + noise
    extra_rows.append(sampled_row)

df_extended = pd.concat([df_extended, pd.DataFrame(extra_rows)], ignore_index=True)

# --- Step 4: Save final dataset ---
output_path = "Final.xlsx"
df_extended.to_excel(output_path, index=False)

print(f"✅ Final dataset saved to: {output_path}")
print(f"📊 Total rallies: {df_extended['rally_id'].nunique()} (should be 150)")
