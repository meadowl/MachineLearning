# Treasure Hunters Inc.

  We do the treasure hunting and monster fighting for you
1. Set up a new git repository in your GitHub account
2. Think up a map-like environment with treasure, obstacles and opponents
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how reinforcement learning can be used to find treasure efficiently while avoiding obstacles and opponents
5. Build one or more reinforcement policies to model situational assessments, actions and rewards programmatically
6. Document your process and results
7. Commit your source code, documentation and other supporting files to the git repository in GitHub

## Citation

Sources for Relevant Paper And Their Authors, All Data Cited That Was Used In This Homework\

+ Book: Machine Learning 
+ Author: Tom Mitchell

## Requirements

+ Pandas
+ Numpy
+ Queue
+ Random


## Engine Methods

There will be an individual section for each method utilized.\
The methods are listed below.

+ Elementary MDP Makrov Decision Process
+ Q Learing
+ Temporal Difference

## Elementary MDP Makrov Decision Process

For starters, there was a basic engine process that was dubbed "basic_engine".\
It was used to verify and establish the world that my machine lives in.\
Basic commands were formulated and implemented, for navigating the enviroments.\
They are listed below.

```Python
     def getself(mymap):

     def getgoal(mymap):

     def move_left(myself):

     def move_right(myself):

     def move_up(myself):

     def move_down(myself):

     def look_around(myself,mymap):
``` 

These commands are designed to do what they are at face value.\
Retrieving initial goal location, initial starting position from searching the map for the key markers for those values.\
The movement commands, retrieve the hypothetical movement in the enviroment.\
Whilst "look_around" provides a direct evaluation for each cardnial direction.

This allows for a simple pattern matching evaluation method to attempt to navigate the world.\
Adding some contigencies and methods, that check for values.\
Building in the idea for remembering the last location the agent moved from.\
And we've almost already implemented a basic MDP, if it wasn't for the fact that no method of learning or computing on the map has happened yet.

With "markov_engine" the initial map pattern learning was developed.\
This has a very basic method, that uses what is esentially a breadth first search from the goal state.\
Not looking for the start, but designed to propigate every traversable path with a degenerative value.\
So say the goal reward is 1000, then the spaces nearest to goal would be values like 850.\
Values that are farther and farther away, would become lower like 10 or even 1.\
Which allowed for a simple actor to move along optimal transitions, or at least attempt to.

There are some mathematical ideas that form the backbone to clearly state them:\
+ A Set of States
+ A Model of Transition of States
+ An Agents Actions
+ A Form of Reward
+ A Form of Learning

But apart from all of that, in essence.\
The learning process itself takes place on the modifications to the map's reward values from transitions.\
Once abstracted to a model of transitions, the agent can take actions from it's current state.\
The movement model, and self-location updates are left largely unchanged in any of the other engines from this point forward.\
So now, it becomes a matter of implementing different ways of formally learning more than what a breadth first search can tell you.\
And leaving the learning process to be influenced by the random nature of upcoming implemenations.

## Q Learing
## Temporal Difference


## Usage

Here is the end result, the data printed out is the accuracy rating of the nodes.\
After the nodes have been trained on some training data, before having the test data used on it.\
The comparison versus the actual expected classifications.

For AustralianCredit, it is the approval or denial of a credit card based on personal metrics.\
For AdultIncome, it is the persons ability to make above or below 50k a year in the USA.

For reference, anything above .50, is better than a random two-sided coin flip.

**Example Usage from Calling "AustralianCredits()" Function**
```Bash
     Starting
     Starting Weights and Answer Estimation
     Ending Weights and Answer Estimation
     [0.6166666666666667, 0.6166666666666667]
     
     MainPart2

     [0.6166666666666667]
``` 

**Example Usage from Calling "AustralianCredits()" Function Training/Testing Sets Swapped**
```Bash
     Starting
     Starting Weights and Answer Estimation
     Ending Weights and Answer Estimation
     [0.63, 0.63]
     
     MainPart2
     
     [0.63]
``` 

![Image From Command Prompt](CreditRun.JPG?raw=true "Title")
![Image From Command Prompt](CreditRun_DataFlipped.JPG?raw=true "Title")

**Example Usage from Calling "AdultIncomes()" Function**
```Bash
     Starting
     Starting Weights and Answer Estimation
     Ending Weights and Answer Estimation
     [0.7276, 0.7276]

     MainPart2

     [0.7276]
``` 


**Example Usage from Calling "AdultIncomes()" Function Training/Testing Sets Swapped**
```Bash
     Starting
     Starting Weights and Answer Estimation
     Ending Weights and Answer Estimation
     [0.7224, 0.7224]

     MainPart2

     [0.7224]
``` 

![Image From Command Prompt](AdultRun.JPG?raw=true "Title")
![Image From Command Prompt](AdultRun_DataFlipped.JPG?raw=true "Title")

## Explaining The Output, with AdultIncomes() as Example.

**Getting My Training and Testing Data**

First, I read in the data file, selecting the rows of data I wanted to use, including the column that I was classifying against.\
And in the case of Adult Income, I had to convert two attributes from text and into numerical values.\
The classification of if someone makes more than 50k, represented with a 1 or a 0.\
And the determination of Job Type, indicated by some amount of 100's.\
Of course, for both sets, I played around with what data to use, and found some like Job Types to work better than the others.\
Not all were carefully selected.

```Python
     def income_converter(x):
     if  x == " <=50K":
          return 0
     return 1

     def job_converter(x):
          if x == " Private":
               return 100
          if x == " Federal-gov":
               return 200
          if x == " Local-gov":
               return 300
          if x == " State-gov":
               return 400
          if x == " Self-emp-not-inc":
               return 500
          return 0

     AdultIncome = pd.read_table("adult.data", header=None, sep=",", converters={14:income_converter,1:job_converter}, usecols=[0,1,2,12,14])
``` 

Next, I set the number of training nodes, as well as their starting values.\
Generating one set of nodes with 4 inputs weights, and a singluar node with 2 input weights.\
Forming what will be a Double Layer Hidden Node Network.\
The inputs flowing into the first two nodes, and the results of those flowing into the third node before being processed to output.

```Python
     NumberOfNodes = 2

     OneRandomWeights = np.random.rand(NumberOfNodes,4)
     OneRandomBiases = np.random.rand(NumberOfNodes)

     OneRandomWeights_b = np.random.rand(1,2)
     OneRandomBiases_b = np.random.rand(1)
``` 

After handling all that, I now process the testing information.\
First grabbing the head of the first set of 5000 entries, and then grabbing the tail of the rest of the entries.\
These are the same things that I swap when I try and do some simple validation in my testing with the swaps.\
For both I start by calculating the pop, lets me free up my ouput vector to its own argument.\
Next I translate both of my tables into explicit arrays so that I can process them in my neural nodes with how I implemented them.\
Great! Now I'm ready to experiment and see just how well my Neural Network will predict the catagorization!

```Python
     AdultIncome_Part1 = AdultIncome.tail(5000)
     AdultIncome_Part2 = AdultIncome.head(5000)

     ExpectedOutputAdultIncome_Array = AdultIncome_Part1.pop(14)
     AdultIncomeValues = AdultIncome_Part1.values
     ExpectedOutputAdultIncome = ExpectedOutputAdultIncome_Array.values

     ExpectedOutputAdultIncome_Array_Part2 = AdultIncome_Part2.pop(14)
     TestAdultIncome_Input = AdultIncome_Part2.values
     TestAdultIncome_Output = ExpectedOutputAdultIncome_Array_Part2.values
``` 

First I'm going to call bulk_training with the first layer of nodes.\
Allowing me to train on some values, and figure out my new baises and weights.\
Next I calculate what my neuron network will classify the values as using this new weights.\
I have to massage the output so my next fuctions can take it with transposing the array.\
Then, due to using ReLu at the end of this process (I swapped what my sigmoid function actually does mid development)\
I have to calculate some ratio based on the average output, to determine if it is in the binary classification.\
Once I do so, I now have a list of "yes, it is over 50k" and "no, it is less than or equal to 50k".\
Which now I can COMPARE to the ORIGINAL VALUES!\
And after seeing how accurate my yes and no results were, I print the ratio of my predictions to the screen for each of the nodes.

**I do the same for the double layered network, with finding derivatives across the chain.**\
Key difference is that I have to feed the output of the first neurons into the input of the second neuron.\
And make sure that I carefully recurse with the back calculation of the partial derivatives.\
Which I then print the result of my accuracy in the same way I did for the previous step.

```Python
     def AdultIncomes():
          print("Starting Weights and Answer Estimation")
          #print([OneRandomWeights, OneRandomBiases])
          #print(neuron_layer(OneRandomWeights,AustralianCreditValues,OneRandomBiases,sigmoid_neuron))
          listofnewweightsbiases = bulk_training(OneRandomWeights,OneRandomBiases, AdultIncomeValues, sigmoid_neuron, ExpectedOutputAdultIncome)
          print("Ending Weights and Answer Estimation")
          #print(listofnewweightsbiases)
          result = neuron_layer(listofnewweightsbiases[0],TestAdultIncome_Input,listofnewweightsbiases[1],sigmoid_neuron)
          columnsorted = list(zip(*result))
          averagedcolumns = DataProcessing(columnsorted)
          error_calc_values = DataInformation(averagedcolumns, TestAdultIncome_Output)
          print(error_calc_values)

          layer_1_outputs = neuron_layer(OneRandomWeights,AdultIncomeValues,OneRandomBiases,sigmoid_neuron)
          listofnewweightsbiases_layer1 = bulk_training(OneRandomWeights,OneRandomBiases, AdultIncomeValues, sigmoid_neuron, layer_1_outputs)
          layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0],AdultIncomeValues,listofnewweightsbiases_layer1[1],sigmoid_neuron)
          #print(layer_2_inputs) 
          listofnewweightsbiases_layer2 = bulk_training(OneRandomWeights_b,OneRandomBiases_b, layer_2_inputs, sigmoid_neuron, ExpectedOutputAdultIncome)
          layer_1_outputs = neuron_layer(listofnewweightsbiases_layer1[0],TestAdultIncome_Input,listofnewweightsbiases_layer1[1],sigmoid_neuron)
          layer_2_outputs = neuron_layer(listofnewweightsbiases_layer2[0],layer_1_outputs,listofnewweightsbiases_layer2[1],sigmoid_neuron)
          columnsorted2 = list(zip(*layer_2_outputs))
          averagedcolumns2 = DataProcessing(columnsorted2)
          error_calc_values2 = DataInformation(averagedcolumns2, TestAdultIncome_Output)
          print("\nMainPart2\n")
          #print(layer_1_outputs)
          #print(layer_2_outputs)
          print(error_calc_values2)
``` 

So.\
From running these functions, I was able to gather several outputs.\
Specifically, the decimial point results, signify the accuracy for the network nodes.\
Obtained by comparing my classification outputs with their actual classifications.\
Hence why swapping the training and testing sets were important, so that I could verify the accuracy and method of training.\
And that is how I got the results that I showed in classification.

## Explaining My Neural Network Implementation

At this point, you've seen the results of my Networks and how to use the API of my Networks.\
So, all that remains is to talk about the implemenation of said API.

Let me be frank, if you dig around in the API, you'll likely see a lot of confusing testing main() functions that I built.\
Along with all sorts of recusive Array functions, that were part of my process to actually get this sort of Nerual Network to work.\
So, let me use this explanation to show the key parts of what I have done and implementated.

**Forward Method**

I start, by abstracting over the idea of what my neuron is actually supposed to compute.\
Which is to take the weights and the inputs, smash them together - add the bias.\
Then plug that into whatever activation function that I've chosen.

```Python
     def my_neuron(weightList, inputList, bias, my_function):
     dotproduct = np.dot(weightList,inputList)
     fullproduct = dotproduct + bias
     return my_function(fullproduct)
```

Which I speciallize with the ReLU function, which I implemented over the original Sigmoid function.\
As the sigmoid, arctan and other methods, converged to the 1.0 value to quickly to make any sense of my results after training.

```Python
     def my_sigmoid(value):
     value2 = value * -1
     sigmoid = 1 / (1 + np.exp(value2))
     #return sigmoid
     if value > 0:
          return value
     return 0
```

The next step is to have some way of abstracting an entire layer.\
Which is a chain of looping over inputs, where the called function then loops over the weights/bias for each input.\
This results in a neuron that takes every input as an argument to be calculated on by every weight/bias.\
So, this is a full connected neural network, for better or for worse - but it gives me a list of results.

```Python
     def neuron_layer(EveryNeurons_weightList, Every_inputList, EveryNeurons_bias, Individual_neuron):
     full_output = []
     for inputList in Every_inputList:
          single_output = singleInput_neuron_layer(EveryNeurons_weightList, inputList, EveryNeurons_bias, Individual_neuron)
          full_output.append(single_output)
     return full_output
```

**Backwards Derivative Method**

At this point I'll admit there are a lot of helper functions, about ten or so, that for various reasons will loop through and call other looping functions.\
So I'll list them all below, and explain in general what they do, going from the called function of bulk_training().

Bulk training itterates over multiple training attempts, I set this itteration to 1, as my data sets were large enough to not need repeated training attempts.\
Multiple training attempts, will train each neuron's weight and return all of the new trained weights, along with the trained biases.\
That is from calling Training Attempt, which trains only one neuron at a time, giving individual weight + bias adjustments.\
Then the layer1_weight_adjustment, will call the helper functions also labled layer1 so it can compute and resolve individual adjustments to each weight and bias.\
Layer1 bias chain will compute by what value to change the bias value by.\
Layer1 derivative chain will compute by what value to change each of the weight values by.\

Both of these layer1 processes, involve a lot of recalcuations, essentially baking in the forward method indirectly.\
By solving for the current evaluation of the dot product between the weights and inputs, and plugging into the derivative of the ReLu.\
Which because of how things got coded originally, goes by the title of the my_derivative_sigmoid.

And after a lot of following loops and calls, you finally get to where bulk_training provides several updated weights and updated biases.

```Python
     def my_derivative_sigmoid(value):
          sigmoid = my_sigmoid(value) * (1 - my_sigmoid(value))
          #return sigmoid #(1-(np.tanh(value) * np.tanh(value) ))#adj2(sigmoid)
          if value > 0:
               return 1
          return 0

     def layer1_derivative_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):

     def layer1_bias_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):

     def layer1_weight_adjustment(weightList, bias, expectedValue, actualValue, Adenduminputs):

     def training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue):

     def multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):

     def bulk_training(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
```

## Showing the Internals with main2()

Below, I'm including a sample run where I experimented and printed out a lot of the internals.\
Here you can see the staring weights of \[\[\[0.1, 0.1, 0.1\], \[1, 1, 1\], \[-1, -1, -1\]\], \[0.1, 1, -1\]\].\
And the newly calculated weights that get printed out.\
Plus a couple of ways that sample inputs were one through, along with showing how high the ReLu values can get between true and false.\
One thing to note:\
In this brief showcase, there were starting negative weights which don't exist in my actual data sets or inital weight values for the real data processing.\
So the error that happens when you give the ReLu a negative value, doesn't immediately snap to zero.\
But in this case, it does snap to zero on several computations, which shows why I'd need to remodify my algorithims if I ever dealt with negative weights or inputs.\
For that case I would likely pick the modified ReLu, so that the general behavior isn't altered.

```Bash
     Starting

     Starting Weights and Answer Estimation
     [[[0.1, 0.1, 0.1], [1, 1, 1], [-1, -1, -1]], [0.1, 1, -1]]
     Ending Weights and Answer Estimation
     [[array([20.1, 20.1, 20.1]), array([21, 21, 21]), array([-1, -1, -1])], [4.1, 5, -1]]
     [[4.1, 5, 0], [124.7, 131, 0], [486.50000000000006, 509, 0], [607.1, 635, 0]]
     C:\Users\drago\Desktop\ML_HW3\HW3.py:81: RuntimeWarning: overflow encountered in exp
       sigmoid = 1 / (1 + np.exp(value2))
     [[11416.85, 11425.939999999999, 0], [320142.94999999995, 320373.98, 0], [1246321.25, 1247218.1, 0], [1555047.35, 1556166.1400000001, 0]]

     Main2

     [[4.1, 5, 0], [607.1, 635, 0]]
     [[11416.85, 11425.939999999999, 0], [1555047.35, 1556166.1400000001, 0]]

     Main2_Method2

     [[4.1, 5, 0], [607.1, 635, 0]]
     [[351.25, 360.34, 0], [44653.75, 45772.54, 0]]
     [Finished in 0.7s]
``` 