import sys
import random
from sklearn import svm

data_file = sys.argv[1]
labels_file = sys.argv[2]
test_file = sys.argv[3]

data = []
with open(data_file, "r") as file:
    line = file.readline()
    while line:
        temp = []
        for i in line.split():
            temp.append(float(i))
        data.append(temp)
        line = file.readline()

labels = {}

with open(labels_file, "r") as file:
    line = file.readline()
    while line:
        label = line.split()
        labels[int(label[1])] = int(label[0])
        line = file.readline()

test_data = []
test_labels = {}

train_data = []
train_labels = {}

train_labels_list = []
test_labels_list = []

# reading data from test file
with open(test_file, "r") as file:
    line = file.readline()
    while line:
        temp = []
        for i in line.split():
            temp.append(float(i))
        test_data.append(temp)
        line = file.readline()

# Splitting into training data and test data (80:20)
# i = 0
# random_numbers = random.sample(range(0, len(data)), 1600)
# random_numbers_set = set(random_numbers)
# for number in random_numbers:
#     test_data.append(data[number])
#     test_labels[i] = labels.get(number)
#     test_labels_list.append(labels.get(number))
#     i += 1

j = 0
for i in range(0, len(data)):
    # if i not in random_numbers_set:
        train_data.append(data[i])
        train_labels[j] = labels.get(i)
        train_labels_list.append(labels.get(i))
        j += 1

contingency_table = []
chi_squares = []

for j in range(0, len(data[0])):
    contingency_table = [[1, 1, 1], [1, 1, 1]]

    for i in range(0, len(train_data)):
        label = train_labels.get(i)
        contingency_table[label][int(train_data[i][j])] += 1

    o1 = contingency_table[0][0]
    o2 = contingency_table[0][1]
    o3 = contingency_table[0][2]
    o4 = contingency_table[1][0]
    o5 = contingency_table[1][1]
    o6 = contingency_table[1][2]

    n = o1 + o2 + o3 + o4 + o5 + o6

    e1 = n * ((o1 + o4) / n) * ((o1 + o2 + o3) / n)
    e2 = n * ((o2 + o5) / n) * ((o1 + o2 + o3) / n)
    e3 = n * ((o3 + o6) / n) * ((o1 + o2 + o3) / n)
    e4 = n * ((o1 + o4) / n) * ((o4 + o5 + o6) / n)
    e5 = n * ((o2 + o5) / n) * ((o4 + o5 + o6) / n)
    e6 = n * ((o3 + o6) / n) * ((o4 + o5 + o6) / n)

    chi_square = 0
    if e1 > 0:
        a1 = ((o1 - e1)**2) / e1
    else:
        a1 = 0
    if e2 > 0:
        a2 = ((o2 - e2) ** 2) / e2
    else:
        a2 = 0
    if e3 > 0:
        a3 = ((o3 - e3)**2) / e3
    else:
        a3 = 0
    if e4 > 0:
        a4 = ((o4 - e4)**2) / e4
    else:
        a4 = 0
    if e5 > 0:
        a5 = ((o5 - e5) ** 2) / e5
    else:
        a5 = 0
    if e6 > 0:
        a6 = ((o6 - e6)**2) / e6
    else:
        a6 = 0
    chi_square = a1 + a2 + a3 + a4 + a5 + a6

    chi_squares.append(chi_square)

selected_columns = []
total_features = 12

for i in range(0, total_features):
    max = chi_squares[0]
    index = 0
    for j in range(1, len(chi_squares)):
        if max < chi_squares[j]:
            max = chi_squares[j]
            index = j
    selected_columns.append(index)
    chi_squares[index] = -1

# print(selected_columns)
reduced_dataset = []
for i in range(0, len(train_data)):
    temp_list = []
    for j in selected_columns:
        temp_list.append(train_data[i][j])
    reduced_dataset.append(temp_list)

clf = svm.SVC(gamma='auto')
clf.fit(reduced_dataset, train_labels_list)

reduced_test_dataset = []
for i in range(0, len(test_data)):
    temp_list = []
    for j in selected_columns:
        temp_list.append(test_data[i][j])
    reduced_test_dataset.append(temp_list)
# print(reduced_dataset)
predict = clf.predict(reduced_test_dataset)
print("Predictions:")
for i in range(0, len(predict)):
    print(predict[i], i)
print("Total no. of features:", total_features)
print("Selected column indexes:")
print(selected_columns)
# print("Accuracy: " ,clf.score(reduced_test_dataset, test_labels_list))
# print(labels_list)
