from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from layer import Layer
from functions.cost import CostFunction
from functions.activation import ReLUActivation, SoftMaxActivation
 
# d = read_csv('data/mines/mines_data.csv')
# df = d.values
    
n_samples = 500
n_classes = 3
train = 0.8
(sd_x, sd_y) = spiral_data(samples=n_samples, classes=n_classes)
p = np.random.permutation(len(sd_x))
sd_x, sd_y = sd_x[p], sd_y[p]
n_train_samples = int((n_samples * n_classes) * train)
X_train, y_train =  (sd_x[:n_train_samples], sd_y[:n_train_samples])
X_test, y_test = (sd_x[n_train_samples:], sd_y[n_train_samples:])
labels = [0, 1, 2]


x_values = np.array(X_train)
y_values = np.array(y_train)

input_layer = Layer(2, 500)
l1 = Layer(500, 400)
l1_activation = ReLUActivation()
l2 = Layer(400, 300)
l2_activation = ReLUActivation()
output_layer = Layer(300, len(labels))
output_activation = SoftMaxActivation()

epochs = 100
costs = []
previous_cost = 9999

piw, pib = input_layer.weights, input_layer.biases
pl1w, pl1b = l1.weights, l1.biases
pl2w, pl2b = l2.weights, l2.biases
pl3w, pl3b = output_layer.weights, output_layer.biases

for epoch in range(epochs):
    y_predicted = []
    print(f"Epoch {epoch}")
    correct = 0
    for (x, label) in zip(X_train, np.array(y_train)):
        input_layer.forward(np.atleast_2d(x))
        l1.forward(input_layer.output)
        l1_activation.forward(l1.output)
        l2.forward(l1_activation.output)
        l2_activation.forward(l2.output)
        output_layer.forward(l2_activation.output)
        output_activation.forward(output_layer.output)

        current_y_predicted = labels[output_activation.output.argmax(axis=1)[0]]
        if(current_y_predicted == label):
            correct += 1

        y_predicted.append(current_y_predicted)
    
    cost = CostFunction(y_values, np.array(y_predicted)).mean_squared_error()
    costs.append(cost)
    if(cost < previous_cost):
        print(f"decreased cost (epoch {epoch}): {previous_cost} -> {cost} ({round((correct / len(X_train)) * 100, 2)}%")
        previous_cost = cost
        piw, pib = input_layer.weights, input_layer.biases
        pl1w, pl1b = l1.weights, l1.biases
        pl2w, pl2b = l2.weights, l2.biases
        pl3w, pl3b = output_layer.weights, output_layer.biases
    else:
        input_layer.weights, input_layer.biases = piw, pib
        l1.weights, l1.biases = pl1w, pl1b
        l2.weights, l2.biases = pl2w, pl2b
        output_layer.weights, output_layer.biases = pl3w, pl3b
        factor = (cost - previous_cost)
        if factor <= 0:
            factor = np.random.randint(0.1, 2)
        input_layer.nudge_w_and_b(factor)
        l1.nudge_w_and_b(factor)
        l2.nudge_w_and_b(factor)
        output_layer.nudge_w_and_b(factor)

plt.plot(costs)
plt.title("Cost Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

print("Final Predictions")
final_total = len(X_test)
final_correct = 0
for (x, label) in zip(X_test, np.array(y_test)):
    input_layer.forward(np.atleast_2d(x))
    l1.forward(input_layer.output)
    l1_activation.forward(l1.output)
    l2.forward(l1_activation.output)
    l2_activation.forward(l2.output)
    output_layer.forward(l2_activation.output)
    output_activation.forward(output_layer.output)
    current_y_predicted = labels[output_activation.output.argmax(axis=1)[0]]
    if(current_y_predicted == label):
        final_correct += 1


print(f"Final Accuracy: {round((final_correct / final_total) * 100)}%")
print(f"Final Cost: {costs[-1]}")

