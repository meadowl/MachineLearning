# Says One Neuron To Another

Neural network architectures
1. Set up a new git repository in your GitHub account
2. Pick two datasets from [Wikipedia](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how neural networks can be used to accomplish the task for the specific dataset
5. Build a neural network to model the prediction process programmatically
6. Document your process and results
7. Commit your source code, documentation and other supporting files to the git repository in GitHub


## Citation

Sources for Relevant Paper And Their Authors, All Data Cited That Was Used In This Homework\
They were found also on [Wikipedia](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research) without links to the dataset.

[AustralianCredit](https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval))
+ [License: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/citation_policy.html)
+ Visibility: Public
+ Dataset owner: (confidential)
+ Last updated: N/A

[AdultData](https://archive.ics.uci.edu/ml/datasets/adult)
+ [License: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/citation_policy.html)
+ Visibility: Public
+ Dataset owner: Ronny Kohavi and Barry Becker
+ Last updated: 1996-05-01


## Requirements

+ Pandas
+ Numpy
+ Collections

## Usage

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
And in the casae of Adult Income, I had to convert two attributes from text and into numerical values.\
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
And after seeing how accurate my yes and no results were, I print the ratio of my predictions to the screen for each of the nodes.\

**I do the same for the double layered network, with finding derivatives across the chain.**
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

After running the total number of words across Shakespeare's plays was possible.
```Bash
     Starting
     25730
```

For refrence after reading through the file, it was found that:\
11114 lines is 10% of the original documents lines\
1111 lines is 1% of the original documents lines

Due to the complexity of the text and runtime, only 1% of Shakespeare's lines were utlized.\
Anything approaching 10% or greater resulted in runtimes that were not calculable and used RAM in excess of 8 Gigabytes.

Now massaging the data into the rest of the program can be done with reading in newspec.txt\
Which was accomplished with Linux command line tool Sed to help clean up most unwanted lines.

```Bash
     sed -i '/ACT/d' spec.txt
     sed -i '/SCENE/d' spec.txt
     sed -i '/Enter/d' spec.txt
     sed -i '/BISHOP/d' spec.txt
     sed '1112,$d' spec.txt > newspec.txt
```

## Figuring Out Building The Matrix

Which allows one to build a reasonable transition matrix for the words probabilites as they relate to one another.

For example:\
The times "castle" follows "the" is 2 times, so the edge is represened as \["the"\]\["castle"\] = 2\
Where \["castle"\]\["the"\] is still set to 0, since "castle" never preceeds "the".

Then after filling the matrix with all representitive edges and their associated occurances.\
The matrix then is itterated over again to calculate the weights.

For example:\
The times "cow" follows "the" is 1 times, so the edge is represened as \["the"\]\["cow"\] = 1\
But now, assuming these to be the only edges - need to calculated their weights.\
So with a total occurance of 3 times for that row, 3 is then divided across the edges resulting in\
\["the"\]\["castle"\] = 2/3 And \["the"\]\["cow"\] = 1/3

Now a matrix containing all weighted edges for transitions has been produced.\
And this matrix can now be used to model the probabilities.

Note that in the case of this implementation, the matrix produced holds both the probability and counts per edge.\
For instance for the strings "the cow" and "the castle":

```Python
     externalgraph[("the","castle")] = [2,2/3]
     externalgraph[("the","cow")] = [1,1/3]
     externalgraph["the","cow"][0] = 1
     externalgraph["the","cow"][1] = 1/3
``` 

This provides easy access to both count, as well as the calculated weight of the edge all at once.

All of this is handled by:
```Python
     def make_externalGraph_commonList(file_name):
``` 

Which handles everything important to massaging the text data.\
From cleaning up the text so that only well-formated words remain without punctuations.\
To sorting and generating a list that contains every unique word.\
As well as initializing every possible pair on the matrix with either 0 or some number of times it occured.\
And doing the initial calcualtion of every pairs probability of occuring.

## How The Transition Matrix Is Used

After constructing the transition matrix,\
There is now an N x N initialized matrix for the N unique words in the text-file that was processed.\
Now it becomes possible to pick the highest weighted edge based on the previous words.

Because specifically Shakespeare writes Shakespeare,\
This means that the one observable state always will relate back to the one observable state with 100% certainty.\
So all calculations rely soley on the calculated hidden-transition matrix.\
And all that's need are functions that will in the case of Goal 1, pick the next most probable path foward based on weights.\
And in the case of Goal 2, take into account the user string with rebalanced weights and then pick the most probable path forward again.

Which leads to some defined functions:
```Python
     def pick_edge(commonMatrix, commonList, previousWord):

     def iterate_text(commonMatrix, commonList, seedWord):

     def rebalance_matrix_weights(commonMatrix, commonList):

     def modify_matrix(commonMatrix, commonList, userWords):

     def iterate_text_user(commonMatrix, commonList, userWords):
``` 

For instance assuming the external graph above with "the castle" and "the cow".\
Here is how these primary functions accomplish the two goals of the project.

Goal 1:\
The pick_edge would given the previous word "the", pick the higher weighted edge to return "castle".\
The iterate_text would do this pick_edge for 20 words, given some word to start from.

Goal 2:\
The rebalance_matrix_weights simply recalculates the stored weighted edges, which is useful because.\
The modify_matrix always increments the stored occurances of edges from some new user input.\
The iterate_text_user predicts 20 words, given some user input to add into the stored matrix, and seeds from the last word in the user's string.
