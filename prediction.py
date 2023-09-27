from data_preprocessing import load_data, processing_data
from lr_sklearn import train_linear_regression

file_path = "data/houses_initial.csv"
data = load_data(file_path)

X_train, X_test, y_train, y_test, scaler, encoder = processing_data(data)

model = train_linear_regression(X_train, y_train)


def calculate_predicted_price(num_bedrooms, num_bathrooms, num_stories, num_parking, area_size,
                              mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishing_status):
    num_bedrooms = int(num_bedrooms)
    num_bathrooms = int(num_bathrooms)
    num_stories = int(num_stories)
    num_parking = int(num_parking)
    area_size = int(area_size)
    mainroad = int(mainroad)
    guestroom = int(guestroom)
    basement = int(basement)
    hotwaterheating = int(hotwaterheating)
    airconditioning = int(airconditioning)
    prefarea = int(prefarea)
    furnishing_status = encoder.transform([[furnishing_status]])

    unscaled_num_bedrooms = (num_bedrooms - scaler.data_min_[0]) / scaler.data_range_[0]
    unscaled_num_bathrooms = (num_bathrooms - scaler.data_min_[1]) / scaler.data_range_[1]
    unscaled_num_stories = (num_stories - scaler.data_min_[2]) / scaler.data_range_[2]
    unscaled_num_parking = (num_parking - scaler.data_min_[3]) / scaler.data_range_[3]
    unscaled_area_size = (area_size - scaler.data_min_[4]) / scaler.data_range_[4]

    binary_features = [mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea]

    prediction_input = [unscaled_num_bedrooms, unscaled_num_bathrooms, unscaled_num_stories, unscaled_num_parking, unscaled_area_size] + binary_features + furnishing_status.tolist()[0]

    predicted_price = model.predict([prediction_input])
    predicted_price_original_scale = predicted_price[0]

    return f"{round(predicted_price_original_scale)} $"

if __name__ == "__main__":
    calculate_predicted_price()
