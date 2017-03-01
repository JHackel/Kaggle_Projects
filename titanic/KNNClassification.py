from sklearn import datasets, metrics, model_selection
import math
import operator

def euclideanDistance(point1, point2, length):
    distance = 0
    for i in range(length):
        distance += pow((point1[i] - point2[i]),2)
    return math.sqrt(distance)

def manhattanDistance(point1, point2, length):#euclideanDistance possible substitute
    distance = 0
    for i in range(length):
        distance += abs(point1[i] - point2[i])
    return distance

def getNeighborIndexes(train,testInstance,k):
    distances = []
    length = len(testInstance)
    for i in range(len(train)):
        distance = euclideanDistance(train[i],testInstance,length)
        distances.append((i,distance))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def kNNClassification(train,trainTarget,test,k):
    if(k > len(train)):
        return #error
    predictions = []
    for i in range(len(test)):
        neighborIndexes = getNeighborIndexes(train,test[i],k)
        neighbors = []
        for j in range(len(neighborIndexes)):
            neighbors.append(trainTarget[neighborIndexes[j]])
        predictions.append(max(set(neighbors), key = neighbors.count))
    return predictions