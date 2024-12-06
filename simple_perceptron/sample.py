
from random import seed
from random import randrange
from csv import reader

with open("sonar.all-data.csv" ,"r+") as file:
    read = reader(file)
    data = []
    for row in read:
        if not row:
            continue
        if row[-1] == "R":
            row[-1] = 1
        else:
            row[-1] = 0
        row = [float(x) for x in row]
        data.append(row)
file.close
# print(data[0])

def predict(row,weights):
    s = weights[0]
    for i in range(len(row)-1):
        s+= weights[i+1]*row[i]
    # print(s)
    return 1.0 if s>=0 else 0.0

def train_weights(data,n_epoch,l_rate):
    weights = [0.0 for i in range(len(data[0])+1)]
    predicted  = []
    for epoch in range(n_epoch):
        total_error = 0
        for row in data:
            prediction = predict(row,weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate*error
            for i in range(len(row)):
                weights[i+1] = weights[i+1] + l_rate*error*row[i]
            total_error+=error**2
        # print("epoch ",epoch ,"  .Total error ====== ",total_error)
    return weights


def perceptron(train_data,test_data,n_epoch,l_rate):
    weights = train_weights(train_data,n_epoch,l_rate)
    predicted = []
    for row in test_data:
        predicted.append(predict(row,weights))
    # print(predicted.count(1.0))
    return predicted
# predicted = perceptron(data,500,0.01)

def dataset_split(dataset,kfold):
    splits = []
    dataset_copy = list(dataset)
    # print(len(dataset),len(dataset_copy))
    n_foldsize= int(len(dataset)/kfold)
    for i in range(kfold):
        batch =[]
        while len(batch) < n_foldsize:
            index = randrange(len(dataset_copy))
            batch.append(dataset_copy.pop(index))
        splits.append(batch)
    return splits
# seed(1)

def calculate_score(actual,predicted):
    s = 0
    for i in range(len(actual)):
        if predicted[i]==actual[i]:
            s+=1
    return(float(s/len(actual))*100)


splits = dataset_split(data,5)
scores = []
for split in splits:
    train_data = list(splits)
    train_data.remove(split)
    train_data = sum(train_data,[])
    test_data = []
    actual = []
    for row in split:
        actual.append(row[-1])
        row_copy  = list(row)
        row_copy[-1] = None
        test_data.append(row_copy)
    predicted = perceptron(train_data,test_data,5000,0.001)
    score = calculate_score(actual,predicted)
    scores.append(score)

print(sum(scores)/5)
