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
1. Generate new text from the text corpus
2. Perform text prediction given a sequence of words
6. Document your process and results
7. Commit your source code, documentation and other supporting 
files to the git repository in GitHub

## Citation

Sources for Relevant Paper And Their Authors
All Data Cited That Was Used In This Homework

[ShakespearePlays](https://www.kaggle.com/kingburrito666/shakespeare-plays/metadata)
License: Unknown
Visibility: Public
Dataset owner: LiamLarsen
Last updated: 2017-04-27, Version 4

## Requirements

+ pandas
+ numpy
+ matplotlib
+ collections

## Usage

**PlaceHolder ExampleUsage**
```Python
     ShakespeareList = Counter(ShakespeareString.split())
     print(len(ShakespeareList))
``` 

## Figuring Out Space-Time Complexity

After massaging the data into plain lowercase words without punctuations, using the Counter method from collections - obtaining a list of unique words became possible via the len function.

```Python
     ShakespeareList = Counter(ShakespeareString.split())
     print(len(ShakespeareList))
``` 

After running the total number of words across Shakespeare's plays was possible.
```
     Starting
     25730
```