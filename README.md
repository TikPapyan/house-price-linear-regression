# House Price Prediction

This project implements a linear regression model to predict house prices based on various features such as the number of bedrooms, bathrooms, stories, parking slots, area size, amenities, and furnishing status. The project is structured into several files for data preprocessing, model training, prediction, and user interface.

## Project Structure

```bash
.
├── data
│   └── houses_initial.csv
├── data_preprocessing.py
├── interface.py
├── lr_sklearn.py
├── main.py
└── prediction.py
```

`data`

This directory contains the dataset used for training and prediction, houses_initial.csv. The dataset contains various features related to houses and their corresponding prices.

`data_preprocessing.py`

This script handles data preprocessing tasks, such as scaling numerical features, one-hot encoding categorical features, and splitting the dataset into training and testing sets. It also provides functions to load and process the data.

`lr_sklearn.py`

This script is responsible for training a linear regression model using the scikit-learn library. It includes functions for model training and evaluation, calculating metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

`prediction.py`

This script allows users to input details of a house and get a predicted price based on the trained linear regression model. It handles the scaling of input features and uses the trained model for prediction.

`interface.py`

This script provides a user-friendly interface for users to input house details and obtain price predictions. It uses the Gradio library to create a simple web-based interface.

`main.py`

This script serves as the entry point for the project. It runs the main functions from lr_sklearn.py and interface.py to train the model and launch the user interface.

## Usage

Run the **main.py** script:

```bash
python3 main.py
```

Use the user interface provided by interface.py to input house details and get price predictions.
