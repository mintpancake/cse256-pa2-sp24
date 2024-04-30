# Epoch: 0, Train Accuracy: 44.64627151051625, Test Accuracy: 33.333333333333336
# Epoch: 1, Train Accuracy: 44.64627151051625, Test Accuracy: 33.333333333333336
# Epoch: 2, Train Accuracy: 56.35755258126195, Test Accuracy: 49.6
# Epoch: 3, Train Accuracy: 64.5793499043977, Test Accuracy: 57.06666666666667
# Epoch: 4, Train Accuracy: 74.90439770554494, Test Accuracy: 64.8
# Epoch: 5, Train Accuracy: 80.87954110898661, Test Accuracy: 70.8
# Epoch: 6, Train Accuracy: 84.51242829827916, Test Accuracy: 72.8
# Epoch: 7, Train Accuracy: 91.92160611854685, Test Accuracy: 79.33333333333333
# Epoch: 8, Train Accuracy: 94.07265774378585, Test Accuracy: 82.4
# Epoch: 9, Train Accuracy: 95.88910133843213, Test Accuracy: 83.86666666666666
# Epoch: 10, Train Accuracy: 93.78585086042065, Test Accuracy: 81.06666666666666
# Epoch: 11, Train Accuracy: 97.4187380497132, Test Accuracy: 84.26666666666667
# Epoch: 12, Train Accuracy: 96.17590822179733, Test Accuracy: 83.06666666666666
# Epoch: 13, Train Accuracy: 97.89674952198853, Test Accuracy: 85.33333333333333
# Epoch: 14, Train Accuracy: 98.18355640535373, Test Accuracy: 85.73333333333333

import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
train_accuracy = [44.64627151051625, 44.64627151051625, 56.35755258126195, 64.5793499043977, 74.90439770554494, 80.87954110898661, 84.51242829827916, 91.92160611854685, 94.07265774378585, 95.88910133843213, 93.78585086042065, 97.4187380497132, 96.17590822179733, 97.89674952198853, 98.18355640535373]
test_accuracy = [33.333333333333336, 33.333333333333336, 49.6, 57.06666666666667, 64.8, 70.8, 72.8, 79.33333333333333, 82.4, 83.86666666666666, 81.06666666666666, 84.26666666666667, 83.06666666666666, 85.33333333333333, 85.73333333333333]

plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()
plt.savefig('plots\plot_1.png')