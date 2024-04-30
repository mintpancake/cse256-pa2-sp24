# Iteration: 99, Train Perplexity: 580.19140625, Test Perplexity hbush: 709.6650390625, Test Perplexity obama: 679.6985473632812, Test Perplexity wbush: 772.5736694335938
# Iteration: 199, Train Perplexity: 392.3570556640625, Test Perplexity hbush: 547.9898681640625, Test Perplexity obama: 522.1256713867188, Test Perplexity wbush: 612.529052734375
# Iteration: 299, Train Perplexity: 272.218505859375, Test Perplexity hbush: 463.7820739746094, Test Perplexity obama: 424.9862365722656, Test Perplexity wbush: 522.8549194335938
# Iteration: 399, Train Perplexity: 197.73545837402344, Test Perplexity hbush: 414.879638671875, Test Perplexity obama: 379.97589111328125, Test Perplexity wbush: 480.94805908203125
# Iteration: 499, Train Perplexity: 154.59033203125, Test Perplexity hbush: 394.35595703125, Test Perplexity obama: 352.42041015625, Test Perplexity wbush: 468.0076904296875

import matplotlib.pyplot as plt

train_perplexity = [580.19140625, 392.3570556640625, 272.218505859375, 197.73545837402344, 154.59033203125]
test_perplexity_hbush = [709.6650390625, 547.9898681640625, 463.7820739746094, 414.879638671875, 394.35595703125]
test_perplexity_obama = [679.6985473632812, 522.1256713867188, 424.9862365722656, 379.97589111328125, 352.42041015625]
test_perplexity_wbush = [772.5736694335938, 612.529052734375, 522.8549194335938, 480.94805908203125, 468.0076904296875]

iterations = [100, 200, 300, 400, 500]

plt.plot(iterations, train_perplexity, label='Train Perplexity')
plt.plot(iterations, test_perplexity_hbush, label='Test Perplexity hbush')
plt.plot(iterations, test_perplexity_obama, label='Test Perplexity obama')
plt.plot(iterations, test_perplexity_wbush, label='Test Perplexity wbush')

plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.title('Perplexity vs Iteration')
plt.legend()
plt.savefig('plots\plot_2.png')