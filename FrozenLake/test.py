import csv
import os
import queue
import time

import numpy as np
import argparse
from env import MyFrozenLake
from agent import Agent
from HXp import HXp, HXpMetric, valid_history

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps or History", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store utility from an HXp", type=str, required=False)

    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)

    parser.add_argument('-equiprobable', '--equiprobable', dest="equiprobable", action="store_true", help="Equiprobable transitions", required=False)
    parser.add_argument('-no_equiprobable', '--no_equiprobable', action="store_false", dest="equiprobable", help="Equiprobable transitions", required=False)
    parser.set_defaults(equiprobable=False)

    parser.add_argument('-pre_info', '--predicate_additional_info', default=None, help="Specify a state", type=str, required=False)
    parser.add_argument('-pre', '--predicate', default="goal", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-spec_his', '--specific_history', nargs="+", default=0, help="Express the specific history", type=int, required=False)
    parser.add_argument('-strat', '--HXp_strategy', default="pi", help="Exploration strategy for generating HXp", type=str, required=False)
    parser.add_argument('-strats', '--HXp_strategies', default="[pi, last_1, last_2]", help="Exploration strategies for similarity measures", type=str, required=False)

    args = parser.parse_args()

    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    EQUIPROBABLE = args.equiprobable
    K = args.length_k

    NUMBER_EPISODES = args.nb_episodes
    CSV_FILENAME = args.csv_filename
    # only used for HXp
    HXP_STRATEGY = args.HXp_strategy
    HXP = args.HXP
    PREDICATE = args.predicate
    PREDICATE_INFO = args.predicate_additional_info
    N = args.n
    FIND_HISTORIES = args.find_histories
    HXP_STRATEGIES = args.HXp_strategies


    # String to int predicate info
    if PREDICATE_INFO is not None:
        if PREDICATE_INFO[0] == '[':
            PREDICATE_INFO = [int(i) for i in PREDICATE_INFO[1:-1].split(', ')]
        else:
            PREDICATE_INFO = int(PREDICATE_INFO)
            print(PREDICATE_INFO)

    #  Fill the specific history list (convert string into int list)
    temp_history = args.specific_history
    SPECIFIC_HISTORY = []
    if isinstance(temp_history, list):
        for elm in list(temp_history):
            if elm not in ['[', ',', ' ', ']']:
                SPECIFIC_HISTORY.append(int(elm))
        print("Specific history : {}".format(SPECIFIC_HISTORY))

    # Path to obtain the Q table
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
    # Path to store actions utility in case of HXp
    if HXP:
        utility_dirpath = 'Utility' + os.sep + MAP_NAME
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store similarity measure in case of HXp
    if HXP and FIND_HISTORIES:
        hist_dirpath = 'Histories' + os.sep + MAP_NAME + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Envs initialisation
    if EQUIPROBABLE:
        env = MyFrozenLake(map_name=MAP_NAME)
    else:
        env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2])

    #  Agent initialization
    agent = Agent(POLICY_NAME, env)

    #  Load Q table
    agent.load(agent_Q_dirpath)

    # Compute HXp from a specific history
    if SPECIFIC_HISTORY and HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)
        # Compute HXp
        start_time = time.perf_counter()
        HXpMetric(specific_history, env, HXP_STRATEGY, agent, utility_csv, PREDICATE,
                  property_info=PREDICATE_INFO, n=N)
        final_time_s = time.perf_counter() - start_time
        print("Explanation achieved in: {} second(s)".format(final_time_s))

    elif FIND_HISTORIES and HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            obs = env.reset()
            done = False
            history.put(obs)  # initial state
            while not done:
                action, _ = agent.predict(obs)
                if history.full():
                    history.get()
                    history.get()
                history.put(action)
                obs, reward, done, info = env.step(action)
                history.put(obs)
                if valid_history(obs, done, reward, PREDICATE, PREDICATE_INFO) and history.full():
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:  # deal with specific_parts predicate (more than 1 history per episode)
                        break
        # Store infos into CSV
        with open(hist_csv, 'a') as f:
            writer = csv.writer(f)
            # First Line
            line = ['History']
            writer.writerow(line)
            # Data
            for data in storage:
                writer.writerow(data)
    else:
        sum_reward = 0
        misses = 0
        steps_list = []
        nb_episode = NUMBER_EPISODES
        # test loop
        for episode in range(1, nb_episode + 1):
            obs = env.reset()
            done = False
            score = 0
            steps = 0
            if HXP:
                history = queue.Queue(maxsize=K*2+1)

            while not done:

                steps += 1
                env.render()
                action, _ = agent.predict(obs)
                #  Compute HXp
                if HXP:
                    history.put(obs)
                    if steps != 1 and steps >= N + 1:
                        # Compute HXp
                        HXp(history, env, HXP_STRATEGY, agent, n=N, csv_filename=utility_csv)
                    # Update history
                    if history.full():
                        # Update history
                        history.get()
                        history.get()
                    history.put(action)

                obs, reward, done, info = env.step(action)
                score += reward

                # Store infos
                if done and reward == 1:
                    steps_list.append(steps)
                elif done and reward == 0:
                    misses += 1

            sum_reward += score
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Score: {}'.format(sum_reward/nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to reach the goal position'.format(np.mean(steps_list)))
            print('Fall {:.2f} % of the times'.format((misses / nb_episode) * 100))
            print('----------------------------------------------')
