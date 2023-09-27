import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def processing_data(data):
    binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

    for column in binary_columns:
        data[column] = data[column].map({"yes": 1, "no": 0})

    numerical_columns = ["bedrooms", "bathrooms", "stories", "parking", "area"]

    numerical_df = data[numerical_columns]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_status = encoder.fit_transform(data[['furnishingstatus']])
    status_columns = encoder.get_feature_names_out(['furnishingstatus'])
    encoded_df = pd.DataFrame(encoded_status, columns=status_columns)

    processed_data = pd.concat([scaled_df, encoded_df, data[binary_columns], data["price"]], axis=1)

    output_file_path = "data/houses_processed.csv"
    processed_data.to_csv(output_file_path, index=False)

    X = processed_data.drop(["price"], axis=1)
    y = processed_data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

    return X_train, X_test, y_train, y_test, scaler, encoder
