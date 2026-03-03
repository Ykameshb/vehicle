import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    df = pd.read_csv("dataset/vehicle_data.csv")

    # Keep only Longitude & Latitude
    df = df[['Longitude', 'Latitude']]
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler


if __name__ == "__main__":
    data, scaler = load_and_preprocess_data()
    print("Processed shape:", data.shape)