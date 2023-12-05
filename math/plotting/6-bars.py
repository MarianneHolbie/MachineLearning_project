#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# define elements of bars
people = ['Farrah', 'Fred', 'Felicia']
fruit_type = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# define position of bars on x-axis
x_pos = np.arange(len(people))

# define width of bars
width = 0.5

# plot bars
for i in range(len(fruit)):
    plt.bar(x_pos, fruit[i], width, bottom=np.sum(fruit[:i], axis=0),
            color=colors[i], label=fruit_type[i])

# add labels
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))
plt.title('Number of Fruit per Person')
plt.xticks(x_pos, people)
plt.legend()

plt.show()
