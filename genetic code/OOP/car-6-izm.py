import numpy as np
import matplotlib.pyplot as plt


data = []
with open("CarPrice_Assignment.csv", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    head = lines[0].split(',')
    for line in lines[1:]:
        if line.strip():
            values = line.split(',')
            row = dict(zip(head, values))
            data.append(row)


columns = ["horsepower", "curbweight", "citympg", "highwaympg", "price", "enginesize"]
data_special = []
for row in data:
    data_special.append([float(row[col]) for col in columns])
array = np.array(data_special)
#print('array:', array)
weights = np.array([0.05, 0.2, 0.1, 0.1, 0.5, 0.05])
data_weight = array * weights
#print('data_weight:', data_special)


means = np.mean(data_weight, axis=0)
stds = np.std(data_weight, axis=0)
standard = (data_weight - means) / stds
#print('standard:', standard)


cov_matrix = np.cov(standard, rowvar=False)
values, vectors = np.linalg.eigh(cov_matrix)
#print('cov_matrix:', cov_matrix)
#print('values:', values)
#print('vectors:',vectors)



n = 4 #скок мерный график
components = vectors[:, :n]
new_world = np.dot(standard, components)
comp_4 = new_world[:, 3]*300 #жирные точки


fig, proect = plt.subplots(1, 3)

proect[0].scatter(new_world[:, 0], new_world[:, 1], sizes = comp_4, c='blue', alpha=0.5)
proect[0].set_title("Проекция 1 и 2")
'''
proect[0].set_xlim([-3, 3])
proect[0].set_ylim([-3, 3])
'''

proect[1].scatter(new_world[:, 0], new_world[:, 2], sizes = comp_4, c='green', alpha=0.5)
proect[1].set_title("Проекция: 1 и 3")
'''
proect[1].set_xlim([-4, 4])
proect[1].set_ylim([-4, 4])
'''

proect[2].scatter(new_world[:, 1], new_world[:, 2], sizes = comp_4,c='red', alpha=0.5)
proect[2].set_title("Проекция: 2 и 3")

'''
proect[2].set_xlim([-3, 3])
proect[2].set_ylim([-3, 3])
'''

fig = plt.figure()
graph = fig.add_subplot(111, projection='3d') #1 строка, 1 столбец, 1 график
graph.scatter(new_world[:, 0], new_world[:, 1], new_world[:, 2], sizes = comp_4,c='blue', alpha=0.5)

graph.set_xlim([-1, 1])
graph.set_ylim([-1, 1])

plt.tight_layout()
plt.show()
