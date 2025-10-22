notes on pytorch and stuff

## PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.
### PyTorch is an open-source deep learning framework that uses Tensors (like GPU-accelerated NumPy arrays) as its fundamental data structure and enables efficient neural network training through automatic differentiation powered by a dynamic computation graph. This "define-by-run" approach offers researchers and developers great flexibility and a highly Pythonic, easy-to-debug experience.
    (gemini's two sentence summary on pytorch)

data --> model --> train --> evaluate

* using MNIST classification w pytorch because it's like the "hello world" of computer vision


# Training + Validation Datasets
split dataset into 3 parts when building ML/NN models:

1. training set - used to train model, compute lost, adjust weights using gradient descent

2. validation set - used to evaluate training model, adjusting hyperparameters + pick best version of model

3. test set - used ot final check model predictions on new unseen data to evaluate how well model is performing


# Model
* use nn.Linear to create model instead of defining + initializing matrices manually
* nn.Linear expects each training example to be a vector --> tensor needs to be flattened out into a vector before being passed into model

* output for each image is vector of size 10
* each element of vector signifies probability of a particular target label (0 to 9)
* ## predicted label for image = one with highest probability


# Softmax Function
* use for converting output to probabilities such that it lies btwn 0 to 1
* ### function that turns a vector of K real values into a vector of K real values that sum to 1
    * input values can be +/-/0, or > 1
    * ## softmax transforms them into values btwn 0-1
        * allows em to be interpreted as probabilities

* if input is small/neg --> value becomes small prob
* if input is large --> values becomes large prob
* *always* remain between 0 and 1

* aka softargmax func / multi-class logistic regresstion
* softmax is a generalization of logistic regress that can be used for multi-class classification
* can be used in classifier only when classes are mutually exclusive

* usual to append softmax func as final layer of neural network


# Evaluation metric + loss function
* evaluate model by finding percentage of labels that were predicted correclty (accuracy of predictions)
* ## accuracy func
    * good evaluation metric but bad for loss function
    * does not take into account hte actual probs predicted by model --> cannot provide sufficient feedback for incremental improvements
* ## cross-entropy
    * loss function used for classification problems
    * quatify difference btwn 2 probabilities distributions
    * one-hot distribution

### cross-entropy pseudo-code to train model
for epoch in range(num_epochs):

    # training phase
    for batch in train_loader:
        * generate predictions
        * calculate loss
        * compute gradients
        * update weights
        * reset gradients
    
    # validation phase
    for batch in val_loader:
        * generate
        * calculate loss
        * calculate metrics (accuracy etc.)
    
    # calculate average validation loss + metrics
    # log epoch, loss + metrics for inspection