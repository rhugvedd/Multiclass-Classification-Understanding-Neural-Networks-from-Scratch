import matplotlib.pyplot as plt
import time
import numpy as np

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def TanhForward(Z):

    A = np.tanh(Z)
    return A, Z

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    
    return A, cache

def softmax(Z):
    Z_max = np.max(Z, axis=0, keepdims=True)
    Z -= Z_max  # Subtract the maximum for numerical stability
    Exp = np.exp(Z)
    A = Exp / np.sum(Exp, axis=0, keepdims=True)

    return A, Z

def PReLU(Z, alpha = 0.1):
    fn = np.maximum(alpha * Z, Z)
    return fn, Z

def PReLU_Backward(dA, Z, alpha = 0.1):
    
    dZ = dA * (Z >= 0) + dA * alpha * (Z < 0)
    dAlpha = np.sum(dA * (Z < 0))
    return dZ, dAlpha

def ReluBackward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    return dZ

def SigmoidBackward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def TanhBackward(dA, Z):
    
    A = np.tanh(Z)
    dZ = dA * (1 - A**2)
    return dZ

def InitParams(LayerDims, StartLearningRate):
    parameters = {}
    L = len(LayerDims)           

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(LayerDims[l], LayerDims[l-1]) * np.sqrt(2 / LayerDims[l-1])
        # parameters['W' + str(l)] = np.random.randn(LayerDims[l], LayerDims[l-1]) / np.sqrt(LayerDims[l-1]) #*0.01

        parameters['Lw' + str(l)] = np.ones((LayerDims[l], LayerDims[l-1])) * StartLearningRate
        
        parameters['b' + str(l)] = np.zeros((LayerDims[l], 1))
        
        parameters['Lb' + str(l)] = np.ones((LayerDims[l], 1)) * StartLearningRate

    return parameters

def LinearForward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    cache = (A_prev, W, b)
    
    return Z, cache

def LinearActivationForward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, LinearCache = LinearForward(A_prev, W, b)
        A, ActivationCache = sigmoid(Z)
    
    elif activation == "relu":
        Z, LinearCache = LinearForward(A_prev, W, b)
        A, ActivationCache = relu(Z)

    elif activation == "tanh":
        Z, LinearCache = LinearForward(A_prev, W, b)
        A, ActivationCache = TanhForward(Z)
    
    elif activation == "PReLU":
        Z, LinearCache = LinearForward(A_prev, W, b)
        A, ActivationCache  = PReLU(Z, alpha = 0.1)

    elif activation == "softmax":
        Z, LinearCache = LinearForward(A_prev, W, b)
        A, ActivationCache  = softmax(Z)

    else:
        print("Wrong Activation!")

    cache = (LinearCache, ActivationCache)

    return A, cache

def ForwardProp(X, parameters, NoOfLayers, InnerLyrActivn, FinlLyrActivn):
    caches = []
    A = X
    L = NoOfLayers
    
    for l in range(1, L):
        A_prev = A 
        A, cache = LinearActivationForward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = InnerLyrActivn)
        caches.append(cache)
    
    AL, cache = LinearActivationForward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = FinlLyrActivn)
    caches.append(cache)
    
    return AL, caches

def Cost(AL, Y, Activation):
    m = Y.shape[1]

    if(Activation == 'sigmoid'):
        cost = (1 / m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    else:
        cost = (1 / m) * np.sum(-1 * Y * np.log(AL))

    cost = np.squeeze(cost)  
    
    return cost

def LinearBackward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def initialize_adam(parameters, NoOfLayers):
    
    v = {}
    s = {}

    for l in range(1, NoOfLayers + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v, s 

def rmsprop_optimizer(parameters, grads, squared_gradients, learning_rate, decay_rate, epsilon=1e-8):

    for l in range(1, NoOfLayers + 1):

        squared_gradients["dW" + str(l)] = decay_rate * squared_gradients["dW" + str(l)] + (1 - decay_rate) * np.power(grads["dW" + str(l)], 2)
        squared_gradients["db" + str(l)] = decay_rate * squared_gradients["db" + str(l)] + (1 - decay_rate) * np.power(grads["db" + str(l)], 2)

        parameters["W" + str(l)] = parameters["W" + str(l)] - ((learning_rate / (np.sqrt(squared_gradients["dW" + str(l)] + epsilon))) * grads["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - ((learning_rate / (np.sqrt(squared_gradients["db" + str(l)] + epsilon))) * grads["db" + str(l)])

    return parameters, squared_gradients

def AdaGrad(parameters, learning_rate, accumulated_grads, grads, NoOfLayers, epsilon=1e-8):

    for l in range(1, NoOfLayers + 1):
        accumulated_grads['dW' + str(l)] += np.power(grads['dW' + str(l)], 2)
        accumulated_grads['db' + str(l)] += np.power(grads['db' + str(l)], 2)

        parameters['W' + str(l)] = parameters['W' + str(l)] - ((learning_rate / (np.sqrt(accumulated_grads['dW' + str(l)]) + epsilon)) * grads['dW' + str(l)])
        parameters['b' + str(l)] = parameters['b' + str(l)] - ((learning_rate / (np.sqrt(accumulated_grads['db' + str(l)]) + epsilon)) * grads['db' + str(l)])

    return parameters, accumulated_grads

def SGDNesterov(X, Y, parameters, learning_rate, momentum, v, InnerLyrActivn, FinlLyrActivn, cost_total):
    
    predicted_params = {}
    
    for l in range(1, NoOfLayers + 1):

        predicted_params["W" + str(l)] = parameters["W" + str(l)] - (momentum * v["dW" + str(l)])
        predicted_params["b" + str(l)] = parameters["b" + str(l)] - (momentum * v["db" + str(l)])

    AL, caches = ForwardProp(X, predicted_params, NoOfLayers, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn = FinlLyrActivn)
    cost_total += Cost(AL, Y, Activation = FinlLyrActivn)
    predgrads = BackwardProp(AL, Y, caches, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn = FinlLyrActivn)    

    for l in range(1, NoOfLayers + 1):
        v["dW" + str(l)] = (momentum * v["dW" + str(l)]) + (learning_rate * predgrads["dW" + str(l)])
        v["db" + str(l)] = (momentum * v["db" + str(l)]) + (learning_rate * predgrads["db" + str(l)])

        parameters["W" + str(l)] = parameters["W" + str(l)] - v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - v["db" + str(l)]

    return parameters, v, cost_total

def update_parameters_with_adam(parameters, grads, v, s, t, NoOfLayers,  learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    v_corrected = {}
    s_corrected = {}

    for l in range(1, NoOfLayers + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]
        
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads['db' + str(l)], 2)
        
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected

def LinearActivationBackward(dA, cache, activation):
    LinearCache, ActivationCache = cache
    
    if activation == "relu":
        dZ = ReluBackward(dA, ActivationCache)
        dA_prev, dW, db = LinearBackward(dZ, LinearCache) 
    elif activation == "sigmoid":
        dZ = SigmoidBackward(dA, ActivationCache)
        dA_prev, dW, db = LinearBackward(dZ, LinearCache)
    elif activation == "tanh":
        dZ = TanhBackward(dA, ActivationCache)
        dA_prev, dW, db = LinearBackward(dZ, LinearCache)
    elif activation == "PReLU":
        dZ, dAlpha = PReLU_Backward(dA, ActivationCache, alpha = 0.1)
        dA_prev, dW, db = LinearBackward(dZ, LinearCache)
    else:
        print("Error! Please make sure you have passed the value correctly in the \"activation\" parameter")
    
    return dA_prev, dW, db

def BackwardProp(AL, Y, caches, InnerLyrActivn, FinlLyrActivn):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    
    CrntCache = caches[L-1]

    if(FinlLyrActivn == 'softmax'):
        dZL = AL - Y
        LinearCache, ActivationCache = CrntCache
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = LinearBackward(dZL, LinearCache)
    else:
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = LinearActivationBackward(dAL, CrntCache, activation = FinlLyrActivn)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = LinearActivationBackward(grads["dA" + str(l + 1)], current_cache, activation = InnerLyrActivn)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def GradDescent(parameters, grads, LearningRate, NoOfLayers):
    L = NoOfLayers

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - LearningRate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - LearningRate * grads["db" + str(l+1)]
        
    return parameters

def SGDNestrovInit(parameters, NoOfLayers) :
    
    v = {}

    for l in range(1, NoOfLayers + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v

def RMSpropInit(parameters, NoOfLayers):
    squared_gradients = {}

    for l in range(1, NoOfLayers + 1):
        squared_gradients["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        squared_gradients["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return squared_gradients

def AdaGradInit(parameters, NoOfLayers):
    accumulated_grads = {}

    for l in range(1, NoOfLayers + 1):
        accumulated_grads["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        accumulated_grads["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return accumulated_grads

def LearningModel(X, Y, LayersDims, DecayRate, Momentum, Optimizer, LearningRate, NumEpochs, PrintCost, InnerLyrActivn, FinlLyrActivn, MiniBatchSize, ContinueConfEpoch):
    
    costs = []
    parameters = InitParams(LayersDims, LearningRate)
    grads = {}
    
    print("Optimizer: " + Optimizer)

    if Optimizer == ADAM:
        v, s = initialize_adam(parameters, NoOfLayers)
        t = 0
        print("LearningRate: " + str(LearningRate))
    elif Optimizer == RMS_PROP:
        squared_gradients = RMSpropInit(parameters, NoOfLayers)
        print("Learning Rate: " + str(LearningRate))
        print("Decay Rate: " + str(DecayRate))
    elif Optimizer == SGD_NESTEROV:
        v = SGDNestrovInit(parameters, NoOfLayers)
        print("Learning Rate: " + str(LearningRate))
        print("Meomentum: " + str(Momentum))
    elif Optimizer == ADA_GRAD:
        accumulated_grads = AdaGradInit(parameters, NoOfLayers)
        print("Learning Rate: " + str(LearningRate))

    NoOfMinibatch = int(X.shape[1] / MiniBatchSize)
    
    Epoch = 0

    while(True):

        cost_total = 0

        for MiniBatch in range(NoOfMinibatch):
            
            if(Optimizer != SGD_NESTEROV):
                AL, caches = ForwardProp(X[:, (MiniBatch * MiniBatchSize) : ((MiniBatch + 1) * MiniBatchSize)], parameters, NoOfLayers, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn = FinlLyrActivn)
                cost_total += Cost(AL, Y[:, (MiniBatch * MiniBatchSize) : ((MiniBatch + 1) * MiniBatchSize)], Activation = FinlLyrActivn)
                grads = BackwardProp(AL, Y[:, (MiniBatch * MiniBatchSize) : ((MiniBatch + 1) * MiniBatchSize)], caches, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn = FinlLyrActivn)
            
            if Optimizer == ADAM:
                t = t + 1
                parameters, v, s, _, _, = update_parameters_with_adam(parameters, grads, v, s, t, NoOfLayers,  LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
            elif(Optimizer == ADA_GRAD):
                parameters, accumulated_grads = AdaGrad(parameters, LearningRate, accumulated_grads, grads, NoOfLayers, epsilon=1e-8)
            elif(Optimizer == SGD_NESTEROV):
                parameters, v, cost_total = SGDNesterov(X[:, (MiniBatch * MiniBatchSize) : ((MiniBatch + 1) * MiniBatchSize)], Y[:, (MiniBatch * MiniBatchSize) : ((MiniBatch + 1) * MiniBatchSize)], parameters, LearningRate, Momentum, v, InnerLyrActivn, FinlLyrActivn, cost_total)
            elif(Optimizer == GRAD_DESC):
                parameters = GradDescent(parameters, grads, LearningRate, NoOfLayers)
            elif(Optimizer == RMS_PROP):
                parameters, squared_gradients = rmsprop_optimizer(parameters, grads, squared_gradients, LearningRate, DecayRate)
            else:
                print("Wrong Optimizer")
        
        cost = cost_total / NoOfMinibatch

        if PrintCost:
            print("Cost after Epoch {}: {}".format(Epoch, np.squeeze(cost)))
        
        costs.append(cost)

        Epoch = Epoch + 1

        if((Epoch % ContinueConfEpoch == 0) and (str(input("Continue? ")) == 'N')):
            break

    return parameters, costs

def predict(X, Y, parameters, NoOfLayers, InnerLyrActivn, FinlLyrActivn):
    m = X.shape[1]
    
    predictions, caches = ForwardProp(X, parameters, NoOfLayers, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn = FinlLyrActivn)
    pred_ans = np.zeros((1, m))
    orig_ans = np.zeros((1, m))

    for col in range(0, predictions.shape[1]):
        pred_ans[0][col] = np.argmax(predictions[:, col])
        orig_ans[0][col] = np.argmax(Y[:, col])

    Accuracy = np.sum((pred_ans == orig_ans)/m)
    
    return pred_ans, Accuracy

def SaveNN(SaveParams, NetworkName):
    if(SaveParams):
        for l in range(1, NoOfLayers + 1):
            file_path = './NeuralNet/' + NetworkName + '/W' + str(l) + '.csv'
            np.savetxt(file_path, parameters['W' + str(l)], delimiter=',')

            file_path = './NeuralNet/' + NetworkName + '/b' + str(l) + '.csv'
            np.savetxt(file_path, parameters['b' + str(l)], delimiter=',')

        print("Neural Network trained successfully..!")

#####################################################################################################################

if __name__ == "__main__":
    print()
    print("Reading Dataset - >")
    print()

    np.random.seed(1)

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    TrainX = train_images.T
    TrainY = train_labels.T
    TestX = test_images.T
    TestY = test_labels.T

    LayersDims = [784, 1000, 1000, 10]
    NoOfLayers = len(LayersDims) - 1

    print("Train X Shape: " + str(TrainX.shape))
    print("Train Y Shape: " + str(TrainY.shape))
    print("Test X Shape: " + str(TestX.shape))
    print("Test Y Shape: " + str(TestY.shape))
    print()
    print("Learning commenced - >")
    print()

    StartTime = time.time()

    InnerLyrActivn = "relu"
    FinlLyrActivn = "softmax"
    SaveParams = False
    NetworkName = "MNIST"

    ADAM = "Adam"
    RMS_PROP = "RMSProp"
    SGD_NESTEROV = "SGDNestrov"
    ADA_GRAD = "AdaGrad"
    GRAD_DESC = "GradDesc"

    parameters, costs = LearningModel   (
                                            TrainX, TrainY, LayersDims, 
                                            Optimizer = ADAM, 
                                            NumEpochs = 20,
                                            PrintCost = True,
                                            LearningRate = 0.0001,
                                            DecayRate = 0.9,
                                            Momentum = 0.95,
                                            InnerLyrActivn = InnerLyrActivn,
                                            FinlLyrActivn = FinlLyrActivn,
                                            MiniBatchSize = 128,
                                            ContinueConfEpoch = 1
                                        )

    print()
    print("Time taken to train: " + str(time.time() - StartTime))

    PredTrain, TrainAcc = predict(TrainX, TrainY, parameters, NoOfLayers, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn=FinlLyrActivn)
    print("Train Accuracy: "  + str(TrainAcc * 100) + "%")

    PredTest, TestAcc = predict(TestX, TestY, parameters, NoOfLayers, InnerLyrActivn = InnerLyrActivn, FinlLyrActivn=FinlLyrActivn)
    print("Test Accuracy: "  + str(TestAcc * 100) + "%")

    SaveNN(SaveParams, NetworkName)

    Labels = label_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]


    while True:
        imageindx = np.random.randint(1, 10000)
        plt.imshow(TestX[:, imageindx].reshape(28,28), cmap=plt.cm.gray)
        true_label = Labels[int(np.argmax(TestY[:, imageindx]))]
        pred_label = Labels[int(PredTest[:, imageindx])]
        plt.title(f"This is: {true_label}\nOur prediction: {pred_label}")
        plt.show()