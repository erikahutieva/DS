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


def get_cluster(x, y, a, b):
    labels = []
    for x_val, y_val in zip(x, y):
        func = a * x_val + b
        if y_val > func:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)

a, b,c, d, e, f = 9, 100, 150, 7500,16, 9000
xy = get_cluster(x, y, a, b)
#print(xy)
xz = get_cluster(x, z, c, d)
#print(xz)
yz = get_cluster(y, z, e, f)  
#print(yz)
colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple', 5: 'brown'}




fig = plt.figure()
ax3d = fig.add_subplot(1, 1, 1, projection='3d')
ax3d.scatter(x, y, z, s=sizes, c='red', alpha=0.5)
ax3d.set_xlabel("Мощность")
ax3d.set_ylabel("Объём двигателя")
ax3d.set_zlabel("Масса")
ax3d.set_title("график (цена 4 измер)")

fig2, axs = plt.subplots(1, 3)
for clust in [0, 1]:
    idx = (xy == clust)  # фильтр по кластеру
    axs[0].scatter(x[idx], y[idx], s=sizes[idx], c=colors[clust], alpha=0.5)
axs[0].set_title("Мощность vs Объём двигателя")

for clust in [2, 3]:
    idx = (xz == (clust - 2)) 
    axs[1].scatter(x[idx], z[idx], s=sizes[idx], c=colors[clust], alpha=0.5)
axs[1].set_title("Мощность vs Масса")





for clust in [4, 5]:
    idx = (yz == (clust - 4)) 
    axs[2].scatter(y[idx], z[idx], s=sizes[idx], c=colors[clust], alpha=0.5)
axs[2].set_title("Объём двигателя vs Масса")

#print(sizes)


plt.show()
