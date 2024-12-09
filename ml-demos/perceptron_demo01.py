def perceptron(inputs_array, weights_array, threshold_value):
    sum_value = 0.0
    for node, weight in zip(inputs_array, weights_array):
        sum_value += node * weight
    return sum_value > threshold_value


inputs = [1, 0, 1, 0, 1]
weights = [0.7, 0.6, 0.5, 0.3, 0.4]
threshold = 1.5

output = perceptron(inputs, weights, threshold)
print(f'the result of a perceptron: {output}')
