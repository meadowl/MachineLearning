# Hidden Markov Modeling

Probabilistic states and transitions
1. Set up a new git repository in your GitHub account
2. Pick a text corpus dataset such as
https://www.kaggle.com/kingburrito666/shakespeare-plays
or from https://github.com/niderhoff/nlp-datasets
3. Choose a programming language (Python, C/C++, Java)
4. Formulate ideas on how machine learning can be used to learn 
word correlations and distributions within the dataset
5. Build a Hidden Markov Model to be able to programmatically 
+ 1. Generate new text from the text corpus
+ 2. Perform text prediction given a sequence of words
6. Document your process and results
7. Commit your source code, documentation and other supporting 
files to the git repository in GitHub

## Citation

Sources for Relevant Paper And Their Authors, All Data Cited That Was Used In This Homework

[ShakespearePlays](https://www.kaggle.com/kingburrito666/shakespeare-plays/metadata)
+ License: Unknown
+ Visibility: Public
+ Dataset owner: LiamLarsen
+ Last updated: 2017-04-27, Version 4

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

[Image From Command Prompt](PythonExe.jfif)

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

11114 lines is 10% of the original documents lines\
1111 lines is 1% of the original documents lines\

Due to the complexity of the text and runtime, only 1% of Shakespeare's lines were utlized.\
Anything approaching 10% or greater resulted in runtimes that were not calculable and used RAM in excess of 8 Gigabytes.\

Now massaging the data into the rest of the program can be done with reading in newspec.txt\

```Bash
     sed -i '/ACT/d' spec.txt
	sed -i '/SCENE/d' spec.txt
     sed -i '/Enter/d' spec.txt
	sed -i '/BISHOP/d' spec.txt
	sed '1112,$d' spec.txt > newspec.txt
```

## Figuring Out Matrix Usage

Which allows one to build a reasonable transition matrix for the words probabilites as they relate to one another.\

For example:\
The times "castle" follows "the" is 2 times, so the edge is represened as \["the"\]\["castle"\] = 2\
Where \["castle"\]\["the"\] is still set to 0, since "castle" never preceeds "the".\

Then after filling the matrix with all representitive edges and their associated occurances.\
The matrix then is itterated over again to calculate the weights.\

For example:\
The times "cow" follows "the" is 1 times, so the edge is represened as \["the"\]\["cow"\] = 1\
But now, assuming these to be the only edges - need to calculated their weights.\
So with a total occurance of 3 times for that row, 3 is then divided across the edges resulting in\
\["the"\]\["castle"\] = 2/3 And \["the"\]\["cow"\] = 1/3\

Now a matrix containing all weighted edges for transitions has been produced.\
And this matrix can now be used to model the probabilities.