import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

data = []
with open("CarPrice_Assignment.csv", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    head = lines[0].split(',')
    for line in lines[1:]:
        if line.strip():
            values = line.split(',')
            row = dict(zip(head, values))
            data.append(row)

columns = ["horsepower", "enginesize", "curbweight", "price"]
data_special = []
for row in data:
    data_special.append([float(row[col]) for col in columns])
array = np.array(data_special)

weights = np.array([0.9, 7.4, 9.0, 0.01])
standard = array * weights

x = standard[:, 0]  
y = standard[:, 1]  
z = standard[:, 2]  
sizes = standard[:, 3]  

a = 9
b = 100
labels = []
for i in range(len(x)):
    if y[i] > a * x[i] + b:
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)


color_map = {0: 'blue', 1: 'orange'}
colors = [color_map[label] for label in labels]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x, y, z, s=sizes, c=colors, alpha=0.6)
ax.set_xlabel("Мощность")
ax.set_ylabel("Объём двигателя")
ax.set_zlabel("Масса")
ax.set_title("3D график (размер точки соответствует цене)")


z_slice = ax.get_zlim()[0]
ax.scatter(x[labels == 0], y[labels == 0], zs=z_slice, s=sizes[labels == 0],
           c='blue', alpha=0.6, label='Cluster 0 (проекция)')
ax.scatter(x[labels == 1], y[labels == 1], zs=z_slice, s=sizes[labels == 1],
           c='orange', alpha=0.6, label='Cluster 1 (проекция)')



plt.tight_layout()
plt.show()
