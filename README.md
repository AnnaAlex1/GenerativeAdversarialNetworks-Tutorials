# MyFirstGAN_Tutorial

https://www.youtube.com/watch?v=OljTVUVzPpM&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=2


### Environment
1. List of environments:
```bash
    conda info --envs
```
or
```bash
    conda env list
```

2. Activate:
```bash
    conda activate myenv
```

3. Deactivate:
```bash
    conda deactivate
```

## Theory

### Sequential Learning?

Machine learning models that input or output data sequences are known as sequence models. Text streams, audio clips, video clips, time-series data, and other types of sequential data are examples of sequential data. Recurrent Neural Networks (RNNs) are a well-known method in sequence models.

in many cases, such as with language, voice, and time-series data, one data item is dependent on those that come before or after it. Sequence data is another name for this type of information.


### nn.Linear 
Applies a linear transformation to the incoming data: y=xAT+by=xAT+b

(in_features, out_features, bias)

- in_features (int) – size of each input sample
- out_features (int) – size of each output sample
- bias (bool) – If set to False, the layer will not learn an additive bias. Default: True

### Leaky.reLU

ReLU stands for rectified linear unit, and is a type of activation function.


Applies the element-wise function:
```
    LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
    LeakyReLU(x)={  x  <-  if x≥0 ,   negative_slope*x <- otherwise ​ }
```

Negative slope is a hyperparameter.

LeakyReLU is usually considered the better choice.


### Sigmoid Function (Nonlinear Activation Function)

A Sigmoid function is a mathematical function which has a characteristic S-shaped curve. There are a number of common sigmoid functions, such as the logistic function, the hyperbolic tangent, and the arctangent. All sigmoid functions have the property that they map the entire number line into a small range such as between 0 and 1, or -1 and 1, so one use of a sigmoid function is to convert a real value into one that can be interpreted as a probability.

In machine learning, the term sigmoid function is normally used to refer specifically to the logistic function, also called the logistic sigmoid function.
Logistic function maps any real value to the range (0, 1).

Applies the element-wise function:
```
    Sigmoid(x)=σ(x)=1/(1+exp(−x))​
```


### Hyperbolic Tangent (Tanh) function (Nonlinear Activation Function)

The tanh (Hyperbolic Tangent) activation function is the hyperbolic analogue of the tan circular function used throughout trigonometry. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).
Compared to the Sigmoid function, tanh produces a more rapid rise in result values. The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.
 

Applies the Hyperbolic Tangent (Tanh) function element-wise.
```
Tanh(x)=tanh(x)= ( exp(x)−exp(−x)​ ) / ( exp(x)+exp(−x) )

```




* https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6



### Transforming and augmenting images

Machine Learning- Working with images
https://towardsdatascience.com/3-things-you-need-to-know-before-working-with-images-in-machine-learning-6a2ab6f6b822

Transforms are common image transformations available in the torchvision.transforms module. They can be chained together using Compose. 

Image to tensor:
https://bekahhw.github.io/What-the-heck-does-it-mean-to-make-an-image-a-tensor


### Datasets

#### Loading Datasets

- root is the path where the train/test data is stored,
- train specifies training or test dataset,
- download=True downloads the data from the internet if it’s not available at root.
- transform and target_transform specify the feature and label transformations



```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
```

#### Dataloader
The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```



### Optimization
The Adam optimizer is also an optimization techniques used for machine learning and deep learning, and comes under gradient decent algorithm. When working with large problem which involves a lot of data this method is really efficient for it. It requires less memory and is efficient, the optimizer is combination of momentum and RMSP algorithm which are gradient decent methodologies. The optimizer is relatively easy to configure where the default configuration parameters do well on most problems.

torch.optim.Adam(): 
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html


### Binary Cross Entropy Loss

Binary crossentropy is a loss function that is used in binary classification tasks. These are tasks that answer a question with only two choices (yes or no, A or B, 0 or 1, left or right). Several independent such questions can be answered at the same time, as in multi-label classification.
Formally, this loss is equal to the average of the categorical crossentropy loss on many two-category tasks.

The binary crossentropy is very convenient to train a model to solve many classification problems at the same time, if each classification can be reduced to a binary choice (i.e. yes or no, A or B, 0 or 1).

Sigmoid is the only activation function compatible with the binary crossentropy loss function. You must use it on the last block before the target block.

The binary crossentropy needs to compute the logarithms of y^iy^​i​ and (1−y^i)(1−y^​i​), which only exist if y^iy^​i​ is between 0 and 1. The sigmoid activation function is the only one to guarantee that independent outputs lie within this range.


torch.nn.BCELoss()
Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html


### SummaryWriter
The SummaryWriter class is your main entry to log data for consumption and visualization by TensorBoard.

The SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries and events to it. The class updates the file contents asynchronously. This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training.


### zero_grad()
In PyTorch, for every mini-batch during the training phase, we typically want to explicitly set the gradients to zero before starting to do backpropragation (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes. This accumulating behaviour is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple mini-batches. So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.

Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters, and the newly-computed gradient. It would therefore point in some other direction than the intended direction towards the minimum (or maximum, in case of maximization objectives).