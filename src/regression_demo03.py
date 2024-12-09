import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import r2_score

numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

model_1 = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
line_1 = numpy.linspace(0, 6, 100)

training_r2_value = r2_score(train_y, model_1(train_x))
print(f'training set => r2 score: {training_r2_value}')

testing_r2_value = r2_score(test_y, model_1(test_x))
print(f'testing set => r2 score: {testing_r2_value}')

print(f'x => y: 5 => {model_1(5)}')

plt.scatter(train_x, train_y)
plt.plot(line_1, model_1(line_1))
plt.show()
