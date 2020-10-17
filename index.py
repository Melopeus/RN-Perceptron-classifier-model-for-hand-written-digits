import gzip
import numpy
import pickle
import random


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


w = numpy.random.rand(10, 784)
deltaW = numpy.zeros((10, 784))
b = numpy.zeros((10, 1), dtype="float")
deltaB = numpy.zeros((10, 1))
learningRate = 0.1


def activation(x):
    if(x > 0):
        return 1
    return 0


def onlineTraining():
    numIterations = 2
    allClassified = False
    trainInputNormal = numpy.array(train_set[0][:])
    trainInputAnswer = 0
    while(not allClassified and numIterations > 0):
        allClassified = True
        c = list(zip(trainInputNormal, train_set[1]))
        trainInput, trainInputAnswer = zip(*c)  
        # trainInput = toate imaginile trainInputAnswer = ce cifra e scrisa
        for i, x in enumerate(trainInput, 0):
            if(i%1000 == 0):
                print("Epoch:%d Img: %d"%(numIterations,i))
            for node in range(0, 10):  
                expected = 1 if node == trainInputAnswer[i] else 0
                y = x@w[node] + b[node]
                output = activation(y)
                w[node] = w[node] + (expected - output)*x*learningRate
                b[node] = b[node] + (expected - output)*learningRate
                if(output != expected):
                    allClassified = False
        numIterations -= 1


def miniBatchTraining():
    global w
    global b
    global deltaB
    global deltaW
    numIterations = 2  # epochs
    allClassified = False
    trainInputNormal = numpy.array(train_set[0][:])  
    trainInputAnswer = 0
    while(not allClassified and numIterations > 0):
        allClassified = True
        c = list(zip(trainInputNormal, train_set[1]))
        trainInput, trainInputAnswer = zip(*c)
        for i, x in enumerate(trainInput, 0):
            for node in range(0, 10):
                expected = 1 if node == trainInputAnswer[i] else 0
                y = x@w[node] + b[node]
                output = activation(y)
                if(i == len(trainInput)):
                    print("Input: %d -- Node: %d -- Result %d" %
                          (trainInputAnswer[1][i], node, output))
                deltaW[node] = deltaW[node] + (expected - output)*x*learningRate
                deltaB[node] = deltaB[node] + (expected - output)*learningRate
                if(output != expected):
                    allClassified = False
            if(i%10 == 0):
                w += deltaW
                b += deltaB
                deltaW = numpy.zeros((10, 784))
                deltaB = numpy.zeros((10, 1))
            if(i%500 == 0):
                print("item {}".format(i))
        print('Epoch: {}'.format(numIterations))
        if(numIterations % 15 == 0):
            test()

        numIterations -= 1

def predict(img):
    output = numpy.zeros((10,1))
    for node in range(0, 10):
        y = img@w[node] + b[node]
        output[node] = y
    return output
    


def test():
    testInput = test_set[0]
    correct = 0
    for i,img in enumerate(testInput,0):
        output = predict(img)
        for j,val in enumerate(output, 0):
            if(val == max(output)):
                if(j == test_set[1][i]):
                    correct += 1
    # print(correct,len(testInput) )
    print("Procentaj: %f" % (correct/len(testInput)))

onlineTraining()
#miniBatchTraining()
test()

