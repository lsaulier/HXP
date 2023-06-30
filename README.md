The implemented code for the *Experimental Results* of the paper *"Predicate-based explanation of a Reinforcement Learning agent via action importance evaluation"* is divided into four folders. Before testing our implementation, it's *necessary* to install packages of requirements.txt using the 
following pip command: 

```bash
pip install -r requirements.txt
```

Then, before running test or training files, the user must be in the problem directory:
```bash
cd 'FrozenLake'
cd 'DroneCoverage'
cd 'Connect4'
```

Find below the main commands to use:
```bash
#####  Frozen Lake  #####
# Training of an Agent for the 4x4 map with 10,000 episodes. The name of the trained policy is '4x4_test' (not required command)
python3 train.py -policy '4x4_test'
# Test the default policy trained in a 4x4 map. The user can ask at each timestep, HXP of maximum length 5.  HXP highlights the most important action and associated state.
python3 test.py
#####  Drone Coverage  #####
# Training of the Agents with 40,000 episodes. (not required command)
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 40000
# Test the default learnt policy. The user can ask at each timestep, HXP of maximum length 5. HXP highlights the most important action and associated state.
python3 test.py
#####  Connect 4  #####
#  Training of the Agents with 200,000 episodes. (not required command)
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 200000 
#  Test the default learnt policy. The user can ask at each time-step, HXP of maximum length 5. HXP highlights the most important action and associated state.
python3 test.py
```

# Code Structure #


## Frozen Lake (FL) ##

### File Description ###

The Frozen Lake folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, and store learnt Q-table into JSON file.


* **test.py**: parameterized python file which loads learnt Q-table and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts in the initial state of the chosen map. The agent's policy is used and must be provided by the user if the map is not '4x4'. 
      At each time-step except the first one, the user can ask for an (approximate) HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user must provide at least the *-spec_his* and *-pre* parameters.
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the RL agent.


* **env.py**: contains a *MyFrozenLake* class: the Frozen Lake environment (inspired by https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)


* **HXp.py**: performs HXP and approximate HXP for a given predicate and history.


* **Q-tables folder**: contains all learnt Q-tables. Each file name starts by *Q_*.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (approximate) HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all CSV files in which histories are stored. In a file, the final state of each history respects the same predicate.

By default, running **train.py** starts a training of Agent of 10,000 episodes, on the 4x4 map and **test.py**
runs a classic testing loop on the 4x4 map. To test HXP for other agents and maps, the user must set the parameters *-map* and *-policy*. 

### Examples ###

The number of training episodes must be higher enough for the agent's learning. 
The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
# Training of an Agent for the 4x4 map with 10,000 episodes. The name of the trained policy is '4x4_test' (not required command)
python3 train.py -policy '4x4_test'
# Training of an Agent on 10x10 map with 500,000 episodes and save Q-table in JSON file with a name finishing by "10x10_test"
python3 train.py -map "10x10" -policy "10x10_test" -ep 500000
```
**Test:**
```bash
#####  Test in user mode a policy  ##### 

# Test the default policy trained in a 4x4 map. The user can ask at each timestep, HXP of maximum length 5. HXP highlights the most important action and associated state.
python3 test.py
# Test the default learnt policy on a 10x10 map.The user can ask at each timestep, HXP of maximum length 8. HXP highlights the 2 most important actions and associated states. 
python3 test.py -policy '10x10_probas262' -map '10x10' -k 8 -n 2

#####  Test HXP from a specific history  ##### 

#  Compute an HXP for a length-8 history, in the 10x10 map. The 'region' predicate is studied and the region is the set of the following states:[14, 15, 16, 24, 25, 26]. This HXP highlights the most important action.
python3 test.py -map 10x10 -policy 10x10_probas262 -strat pi -k 8 -spec_his 12 1 22 2 23 2 24 3 25 3 15 1 16 2 17 3 7  -pre 'specific_part'  -pre_info '[14, 15, 16, 24, 25, 26]'
#  Compute an approximate HXP (one last deterministic transition) for a length-10 history, in the 10x10 map. The 'holes' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -map 10x10 -policy 10x10_probas262 -strat last_1 -k 10 -spec_his 56 2 57 3 47 2 37 1 38 2 39 3 29 2 19 0 18 0 17 3 7  -pre 'holes' -n 2

#####  Produce x histories whose last state respects a certain predicate  #####

#  Find 50 length-5 histories whose last state respects the goal predicate in the 10x10 map. Histories are stored in the 'goal_k5_50hist.csv' file.
python3 test.py -find_histories -ep 50 -pre goal -map 10x10 -policy 10x10_probas262 -csv 'goal_k5_50hist.csv' -k 5
```

## Drone Coverage (DC) ##

### File Description ###

The Drone Coverage folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, save info into text files and neural network into *dat* files.


* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts in an initial configuration. The agents policy is used. 
      At each time-step except the first one, the user can ask for an (approximate) HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the agents.


* **env.py**: contains a DroneCoverageArea class: the Drone Coverage environment.


* **DQN.py**: this file is divided into 3 parts, it's inspired by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py
and https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py. 
It contains a *DQN* class which is the neural network used to approximate *Q*, an *ExperienceBuffer* class to store agents 
experiences and functions to calculate the DQN's loss.


* **HXp.py**: performs HXP and approximate HXP for a given predicate and history.


* **Models folder**: contains already learnt policies for agents. 
The names of the produced DQNs show partially the hyperparameter values. In order, values correspond to the timestep 
limit of training, the timestep where epsilon reaches the value of 0.1 (exploration rate), timestep frequency of 
synchronization of the target neural network, the time horizon of an episode and the use of double-Q learning extension.


* **Logs folder**: contains log files from the learning phase of the agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (approximate) HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all CSV files in which histories are stored. In a file, the final state of each history respects the same predicate. Moreover, this folder contains histories stored as a text file.

By default, running **train.py** starts a training of the agents of 100,000 episodes and **test.py**
runs a classic testing loop where the user can ask for an (approximate) HXP. To test (approximate) HXP from a specific configuration, 
the user must set a text file for the *-spec_his* parameter. Examples of the used convention for representing a history within a text file can be found in the **Histories** folder.  


### Examples ###

The number of training time-steps must be greater than or equal to 40,000 according to the other default parameters values. 
The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
#  Train Agents on 10x10 map with a timestep limit of 40000. It saves info into "Test_Logs" folder and neural networks into "Test_Models" folder
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 40000
#  Train Agents on 10x10 map with a timestep limit of 30000. It saves info into "Test_Logs" folder and neural networks into "Test_Models" folder. The transition function is deterministic since there is no wind. The batch size is set to 16 
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 30000 -no_w -batch 16
```
**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy on a 10x10 map with 4 agents. Agents start at random positions. The user can ask at each time-step, HXP of maximum length 5. HXP the most important action and associated state.
python3 test.py
#  Test the default learnt policy on a 10x10 map with 4 agents. Agents start at random positions. The user can ask at each timestep, HXP of maximum length 4. HXP highlights the 2 most important actions and associated states.
python3 test.py -k 4 -n 2

#####  Test HXP from a specific history  ##### 

#  Compute an HXP for a given length-5 history, based on the actions of the Blue drone (Blue id = 1). The 'global region' predicate is studied. This HXP highlights the most important action.
python3 test.py -id 1 -spec_his All_regions_k5_Ag1.txt -strat pi -pre 'all regions'
#  Compute an approximate HXP (one last deterministic transition) for a length-5 history, based on the actions of the Green drone (Green id = 1). The 'local prefect cover' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -id 2 -spec_his H0L5P_perfect\ coverAg2.txt -strat last_1 -pre 'one perfect cover' -n 2

#####  Produce x histories whose last state respects a certain predicate  #####

# Find 30 length-4 histories whose last state of the Blue drone respects the local perfect cover predicate. Histories are stored in the 'localmaxcover_k6_100hist.csv' file.
python3 test.py -find_histories -ep 30 -pre 'one perfect cover' -id 1 -csv 'localperfectcover_k4_30hist.csv' -k 4
```

## Connect4 (C4) ##

### File Description ###

The Connect4 folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, save info into text files and neural network into *dat* files.
                Self-play methods (inspired by: https://github.com/Unity-Technologies/ml-agents/tree/release_20_docs) are used for the learning of both Players behavior.   


* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts with an empty board. The agent's policy is used for Player 1 and Player 2. To avoid similar plays over the episodes, the Player 2 has 30% of probability to play randomly.  
      At each time-step except the first one, the user can ask for an (approximate) HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the agents.


* **env.py**: contains a class *Connect4*: the Connect4 environment.


* **DQN.py**: this file is divided into 3 parts, it's inspired by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py, 
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py and https://codebox.net/pages/connect4. 
It contains a *DQN* class which is the neural network used to approximate *Q*, an *ExperienceBuffer* class to store agents 
experiences and functions to calculate the DQN's loss.


* **HXp.py**: performs HXP and approximate HXP for a given predicate and history.


* **Models folder**: contains already learnt policies for agents.


* **Logs folder**: contains log files from the learning phase of the agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (approximate) HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all CSV files in which histories are stored. In a file, the final state of each history respects the same predicate. Moreover, this folder contains histories stored as a text file.

By default, running **train.py** starts a training of the agents of 1,000,000 episodes and **test.py**
runs a classic testing loop where the user can ask for an (approximate) HXP. To test (approximate) HXP from a specific configuration, 
the user must set a text file for the *-spec_his* parameter. Examples of the used convention for representing a history within a text file can be found in the **Histories** folder.  


### Examples ###

The number of training time-steps must be greater than or equal to 200,000 according to the other default parameters values. 
The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
#  Train Agents with a time-step limit of 1,000,000.
python3 train.py -model 'Test_Models' -log 'Test_Logs'
#  Train Agents with a timestep limit of 100,000. Only the last two models of the opponent are saved. The learning agent play against these models. Each 50,000 time-steps, the agent who learns change (This is done to alternate between Player 1 and Player 2 learning phase). 
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 100000 -window 2 -player_change 50000
```
**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy. The user can ask at each time-step, HXP of maximum length 5. HXP the most important action and associated state.
python3 test.py
#  Test the default learnt policy. The user can ask at each timestep, HXP of maximum length 6. HXP highlights the 2 most important actions and associated states.
python3 test.py -k 6 -n 2

#####  Test HXP from a specific history  ##### 

#  Compute an approximate HXP (three last deterministic transitions) for a given length-6 history. The 'control middle column' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -spec_his  '6_2_win_hist.txt'  -strat last_3 -k 6 -pre 'control_midcolumn' -n 2
#  Compute an approximate HXP (two last deterministic transitions) for a length-5 history. The 'lose' predicate is studied. This approximate HXP highlights the most important action.
python3 test.py -spec_his  'H0L5P_lose_exp.txt' -strat last_2 -pre 'lose'

#####  Produce x histories whose last state respects a certain predicate  #####

#  Find 100 length-5 histories whose last configuration respects the '3 in a row' predicate. Histories are stored in the '3inarow_k5_100hist.csv' file.
python3 test.py -find_histories -ep 100 -pre '3inarow' -csv '3inarow_k5_100hist.csv' -k 5
```

## Additional files / folder ##

The following files and folder are located at the root of the project and are used for the computation of similarity scores, rate of similar most important action selected and run times:

* **similarity.py**: computes HXP and approximate HXPs for each history in a CSV file. In addition to action importance scores, the similarity scores and run-times for each approach is computed then stored for each history. Average and standard deviation of similarity scores and run-times are also computed.  
                     Find below some examples of this file use:
```bash
# Compute action importance scores, similarity scores and run-times for each history in the 'goal_k5_50hist.csv' file. This is done for the Frozen Lake problem, with the default learnt policy (10x10_probas262) in the 10x10 map. The output file is located in the Similarity folder.
python3 similarity.py -file '/FrozenLake/Histories/10x10/50-histories/goal_k5_50hist.csv' -pre goal -problem FL -new_file 'Similarity/FrozenLake/10x10/goal_k5_50hist.csv' -policy 10x10_probas262
# Compute action importance scores, similarity scores and run-times for each history in the 'localperfectcover_k4_30hist.csv' file. This is done for the Drone Coverage problem, with the default learnt policy (tl1600000e750000s50000th22ddqnTrue) in the 10x10 map. The output file is located in the Similarity folder.
python3 similarity.py -file '/DroneCoverage/Histories/30-histories/localperfectcover_k4_30hist.csv' -pre 'one perfect cover' -id 1 -problem DC -new_file 'Similarity/DroneCoverage/10x10/localperfectcover_k4_30hist.csv' -policy '/DroneCoverage/Models/Agent/tl1600000e750000s50000th22ddqnTrue-best_11.69.dat'
# Compute action importance scores, similarity scores and run-times for each history in the '3inarow_k5_100hist.csv' file. This is done for the Connect4 problem, with the default learnt policy (bestPlayerP1_98_P2_96). The output file is located in the Similarity folder.
python3 similarity.py -file '/Connect4/Histories/100-histories/3inarow_k5_100hist.csv' -pre '3inarow' -problem C4 -new_file 'Similarity/Connect4/3inarow_k5_100hist.csv' -policy '/Connect4/Models/bestPlayerP1_98_P2_96.dat'
```

* **same_impaction_rate.py**: Given a CSV file obtained after the use of the *similarity.py* file, this file computes the rate of same most important action returned by the HXP and approximate HXPs.
                    Find below some examples of this file use:
```bash
# Compute, for histories from the Frozen Lake problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/FrozenLake/10x10/goal_k5_50hist.csv'
# Compute, for histories from the Drone Coverage problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/DroneCoverage/10x10/localperfectcover_k4_30hist.csv'
# Compute, for histories from the Connect4 problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/Connect4/3inarow_k5_100hist.csv'
``` 

* **Similarity folder**: For each problem, this folder contains all the results obtained with the *similarity.py* file, i.e. files containing for each studied history, action importance scores of different approaches, similarity scores and run-times.

## Remarks ##

For each problem, the user can ask for an (approximate) HXP for all predicates presented in the paper.
If the user wants to compute similarity and the rate of same important action on new histories, he must run successively the *test.py* of a problem, the *similarity.py* and *same_impaction_rate.py* with appropriate parameters.
As a technical detail, for the DC and C4 problem, if CUDA is available, the training and use of DQN will be with the GPU, otherwise with the CPU.