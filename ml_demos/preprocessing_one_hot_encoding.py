import pandas as pd
from sklearn import linear_model

cars = pd.read_csv('../data/car_data.csv')
print(cars)
# one-hot-encoding by the pandas.get_dummies()
ohe_car_brands = pd.get_dummies(cars[['Car']], drop_first=True)
print(ohe_car_brands)

X = pd.concat([cars[['Volume', 'Weight']], ohe_car_brands], axis=1)
y = cars['CO2']

linr = linear_model.LinearRegression()
linr.fit(X, y)

# predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = linr.predict([[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

print(f'Predicted CO2: {predictedCO2}')
