import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_excel("excel/time_series_shot_data_all_40.xlsx")

# Separate by source
hits = df[df["source"] == "hitting_position"].copy()
bounces = df[df["source"] == "shot_placement"].copy()

features = []

for rally_id in df["rally_id"].unique():
    dj_hits = hits[(hits["rally_id"] == rally_id) & (hits["player"] == "Djokovic (SM)")].sort_values("frame_id")
    op_hits = hits[(hits["rally_id"] == rally_id) & (hits["player"] == "Opponent (Op)")].sort_values("frame_id")
    rally_bounces = bounces[bounces["rally_id"] == rally_id]

    if len(dj_hits) < 1:
        continue

    row = {"rally_id": rally_id}

    for i in range(3):
        if i >= len(dj_hits):
            for f in ["dj_x", "dj_y", "op_x", "op_y", "sp_x", "sp_y"]:
                row[f"{f}{i+1}"] = np.nan
            continue

        dj_shot = dj_hits.iloc[i]
        frame = dj_shot["frame_id"]
        row[f"dj_x{i+1}"] = dj_shot["x_m"]
        row[f"dj_y{i+1}"] = dj_shot["y_m"]

        # Opponent location in the same frame
        opp = op_hits[op_hits["frame_id"] == frame]
        if not opp.empty:
            row[f"op_x{i+1}"] = opp.iloc[0]["x_m"]
            row[f"op_y{i+1}"] = opp.iloc[0]["y_m"]
        else:
            row[f"op_x{i+1}"] = np.nan
            row[f"op_y{i+1}"] = np.nan

        # Djokovic shot placement (find next bounce after hit)
        dj_place = rally_bounces[
            (rally_bounces["frame_id"] > frame) &
            (rally_bounces["player"] == "Djokovic (SM)")
        ].sort_values("frame_id")

        if not dj_place.empty:
            row[f"sp_x{i+1}"] = dj_place.iloc[0]["x_m"]
            row[f"sp_y{i+1}"] = dj_place.iloc[0]["y_m"]
        else:
            row[f"sp_x{i+1}"] = np.nan
            row[f"sp_y{i+1}"] = np.nan

    # ✅ Target: Djokovic's NEXT shot placement (bounce after last hit)
    last_frame = dj_hits.iloc[min(2, len(dj_hits)-1)]["frame_id"]
    next_dj_bounce = rally_bounces[
        (rally_bounces["frame_id"] > last_frame) &
        (rally_bounces["player"] == "Djokovic (SM)")
    ].sort_values("frame_id")

    if not next_dj_bounce.empty:
        row["target_x"] = next_dj_bounce.iloc[0]["x_m"]
        row["target_y"] = next_dj_bounce.iloc[0]["y_m"]
    else:
        row["target_x"] = np.nan
        row["target_y"] = np.nan

    features.append(row)
    print(f"✅ Processed {rally_id}")

# Save to Excel
features_df = pd.DataFrame(features)
os.makedirs("excel", exist_ok=True)
features_df.to_excel("excel/djokovic_features_3shots_predict_next_djokovic_shot.xlsx", index=False)

print(f"\n✅ Final output: Extracted {len(features_df)} rows → excel/djokovic_features_3shots_predict_next_djokovic_shot.xlsx")
