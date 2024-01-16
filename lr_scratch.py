import numpy as np

from data_preprocessing import load_data, processing_data


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ssr / sst)


def train_linear_regression(X_train, y_train):
    X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    coefficients = np.linalg.inv(X_train_with_bias.T.dot(X_train_with_bias)).dot(X_train_with_bias.T).dot(y_train)

    bias = coefficients[0]
    weights = coefficients[1:]

    return bias, weights


def predict(X_test, bias, weights):
    X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    y_pred = X_test_with_bias.dot(np.concatenate(([bias], weights)))

    return y_pred


def main():
    file_path = "data/houses_initial.csv"
    data = load_data(file_path)
    
    X_train, X_test, y_train, y_test, scaler, encoder, processed_data = processing_data(data)

    bias, weights = train_linear_regression(X_train, y_train)
    y_pred = predict(X_test, bias, weights)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("From scratch metrics \n")
    print(f"Mean Squared Error on Test Data: {mse}")
    print(f"Root Mean Squared Error on Test Data: {rmse}")
    print(f"R-squared on Test Data: {r2} \n")
    print("-" * 65, "\n")


if __name__ == "__main__":
    main()
