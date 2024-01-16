import numpy as np

from data_preprocessing import load_data, processing_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2


def main():
    file_path = "data/houses_initial.csv"
    data = load_data(file_path)
    
    X_train, X_test, y_train, y_test, scaler, encoder = processing_data(data)

    model = train_linear_regression(X_train, y_train)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)

    print(f"Mean Squared Error on Test Data: {mse}")
    print(f"Root Mean Squared Error on Test Data: {rmse}")
    print(f"R-squared on Test Data: {r2} \n")
    print("-" * 65, "\n")


if __name__ == "__main__":
    main()
