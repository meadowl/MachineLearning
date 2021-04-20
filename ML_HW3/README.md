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

Sources for Relevant Paper And Their Authors, All Data Cited That Was Used In This Homework

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

**Example Usage from Calling "main()" Function**
```Bash
     Starting
     Loading Edges...
     Unique Words: 2068
     Shakespeare Text Generation Menu
     Option 1: Generate Seeded Text
     Option 2: Generate From User Input
     Type 1 or 2 To Select Option, 3 to Exit
     Input Integer Now:
     1
     the devil alone falstaff sweats to be hanged sirs you are ye fat room while i will i will i will


     Shakespeare Text Generation Menu
     Option 1: Generate Seeded Text
     Option 2: Generate From User Input
     Type 1 or 2 To Select Option, 3 to Exit
     Input Integer Now:
     2
     Please Provide A User String Now:
     Thou never see
     thou never see thee lend me thy lantern quoth pick purse that's flat he loves him i will i will i will i


     Shakespeare Text Generation Menu
     Option 1: Generate Seeded Text
     Option 2: Generate From User Input
     Type 1 or 2 To Select Option, 3 to Exit
     Input Integer Now:
     3
``` 

![Image From Command Prompt](PythonExe.jfif?raw=true "Title")

## Figuring Out Space-Time Complexity

After massaging the data into plain lowercase words without punctuations, using the Counter method from collections - obtaining a list of unique words became possible via the len function.

```Python
     ShakespeareList = Counter(ShakespeareString.split())
     print(len(ShakespeareList))
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
