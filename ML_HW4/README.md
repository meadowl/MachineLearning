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

There are some mathematical ideas that form the backbone to clearly state them:
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

With the ability to initialize and have the agent make decisions over the map.\
All that remains is implementing the randomized Q Learning.\
This is encapsulated in the randomized_training function, with it being used in a modified engine.

```Python
     def randomized_training(mymap):

     def q_engine():
```

So fundamentally the idea with the Q Learning, is that you can drop the agent into a random position in the grid.\
And then, you can allow it to make a singular optimal action to its next location.\
This is then recorded via some learning ratio to update the transition's reward value, for future traversals.\
Ideally you repeat this some large number of times, and at some point sensible rewards will be establised.\
Such that the agent now modifies the transition matrix, to now have heavier weights on favorable transitions to the ultimate goal state.

Now that the transition matrix is modified significantly.\
You are able to drop the agent into the enviroment, where it will be able to take multiple steps and eventually reach the goal.\
In essence since the transition matrix encapsulates the learning, there isn't any modification to how the agent moves.\
In each state the agent finds itself in, the learned rewards for comparison between all of it's movements it already there.\
So it must simply take the path of best reward like it already was programed to do.

This is good to know, as in the next learning method, the learning again is represented strictly by how the transition matrix is altered.

## Temporal Difference

Now that Q Learing was done, there is a varient called Temporal Learning, that looks ahead several steps.\
Before modifying the reward value for the square that was randomly selected to be moved on.\
The method of looking ahead is recursively calculated from the "temporal_calculation", with the training function adjusted.\
Barely any changes again in the engine function apart from debugs and swapping the training mechinism.

```Python
     def temporal_calculation(mymap, current_loc, mylambda, iteration_depth, max_depth):

     def temporal_training(mymap):

     def temporal_engine():
```

Basically (1 - lambda)\[Q1 + lambda_Q2 + lambda^2_Q3+...\] is the general method for Q Learning.\
So this allows for a recursive definition to be made for training on the idea of looking forward several steps.\
With some discount that is the value of lambda between 0 and 1, to modify each further iteration.\
For the case of this program, up to Q3 was generated to verify and test that the implemenation was functional.\
And a lambda of 0.5 was picked, for a similar reason.\
Now this is used to be trained on for the transition matrix.

As a result, after training, the agent again moves through the modified transition matrix.\
Only this time every adjusted value was based on a long-term, multi-movement training mechanism.

## Summary and Discussion of Approach

First, here are some examples of maps that were used, in modifying the Map.txt file:

**Goal in the middle, surrounded by enemies, starting point at corner.**
```Bash
   0  1  2  3  4  5  6  7  8  9
0  P  P  P  P  P  P  P  P  P  P
1  P  P  P  P  P  P  P  P  P  P
2  P  P  P  P  P  P  P  P  P  P
3  P  P  P  P  P  P  P  P  P  P
4  P  P  P  P  P  P  P  P  P  P
5  P  P  P  E  G  E  P  P  P  P
6  P  P  P  P  E  P  P  P  P  P
7  P  P  P  P  P  P  P  P  P  P
8  P  P  P  P  P  P  P  P  S  P
9  P  P  P  P  P  P  P  P  P  P
```

In this case, it ran perfectly, able to find a path in most of the engine cases.\
Basic Engine Failed, but that wasn't one that had any learning implemented.

**Goal at the end, a false path, a good path, no other options.**
```Bash
   0  1  2  3  4  5  6  7  8  9
0  G  P  P  P  P  P  E  E  E  E
1  P  E  E  E  E  P  E  E  E  E
2  P  E  E  E  E  P  E  E  E  E
3  P  P  E  E  E  E  E  E  E  E
4  E  P  E  E  E  E  E  E  E  E
5  E  P  E  E  E  E  E  E  E  E
6  E  P  E  E  E  E  E  E  E  E
7  E  P  E  E  E  E  E  E  E  E
8  E  P  P  P  P  P  E  E  E  E
9  E  E  E  E  E  P  P  P  S  E
```

Happily for even Basic Engine, every way was able to solve this problem without issues.

**Goal in the middle, completely surrounded by enemies, unwinable map.**
```Bash
   0  1  2  3  4  5  6  7  8  9
0  P  P  P  P  P  P  P  P  P  P
1  P  P  P  P  P  P  P  P  P  P
2  P  P  P  P  P  P  P  P  P  P
3  P  P  P  P  P  P  P  P  P  P
4  P  P  P  P  E  P  P  P  P  P
5  P  P  P  E  G  E  P  P  P  P
6  P  P  P  P  E  P  P  P  P  P
7  P  P  P  P  P  P  P  P  P  P
8  P  P  P  P  P  P  P  P  S  P
9  P  P  P  P  P  P  P  P  P  P
```

As expected, every Engine failed magnificently.\
Each one exterted some interesting way of getting lost in an unwinable state space.\
By wandering around in a perpetual no-reward movement strategy.\
Since there was no way to learn a way into the goal state, as enemies prevented all routes.

**Discusion of Comparison Between Results**

Something notable, was perhaps due to the limited size of the world only being a possible of a 100 locations.\
That once any of the algorithims were trained after a reasonable itteration of say 1000 or such,\
The results were nearly identical and it became hard to distinguish what advantage the individual implementations had.\
Perhaps for more complex scenarios, larger state spaces, or other conditions these would become more apparent.

Interestingly, there was a shared point of falure on training, and that was over-training.\
This was the first time I encountered this problem. And it led to some interesting things worth discussing.\
The transition matrix would become trained at maxed out reward values given enough time, and even if you were careful to make sure that\
each state space couldn't be more than the goal space - you end up with several spots converging to say 999 across the entire map.\
Which now complicated the agent's movements, as every place it wants to move looks as appealing as the last.\
And it likely won't find the actual goal state, instead it will be content to keep accepting the 999 values instead of moving to the goal.\
So you must be careful to inspect the resulting transition matrix during trainings, and ensure that you've set some qualitative value that likely won't give you over training.

Avoiding this one pitfall of overtraining, and everything runs smoothly.\
Except in the case where no path exists to the goal state, and makes training impossible.