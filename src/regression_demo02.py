import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def multiple_regression():
    df = pandas.read_csv('../datasets/car_data.csv')
    X = df[['Weight', 'Volume']]
    y = df['CO2']

    multiple_regression_model = linear_model.LinearRegression()
    multiple_regression_model.fit(X, y)
    predicted_co2 = multiple_regression_model.predict([[3300, 1300]])
    print(f'Predicted CO2: {predicted_co2}')


multiple_regression()


def multiple_regression_with_scale():
    scaler = StandardScaler()
    df = pandas.read_csv('../datasets/car_data.csv')
    X = df[['Weight', 'Volume']]
    # print(X)
    scaled_X = scaler.fit_transform(X)
    # print(scaled_X)
    y = df['CO2']

    multiple_regression_model = linear_model.LinearRegression()
    multiple_regression_model.fit(scaled_X, y)
    scaled_inputs = scaler.transform([[3300, 1300]])
    predicted_co2 = multiple_regression_model.predict(scaled_inputs)
    print(f'Predicted CO2: {predicted_co2}')


multiple_regression_with_scale()
