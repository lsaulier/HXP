import csv
import queue
import time

from DQN import DQN
from agent import Agent
from env import Connect4
from HXp import HXp, HXpMetric, valid_history
import torch
import os
import argparse
from copy import deepcopy

###### AGENT's POLICY #################
'''
best model: bestPlayerP1_98_P2_96.dat
other model: badPlayer_82.dat
best without saving multiple NNs to change the opponent during the training:
   - tl1000000e800000s5000ddqnTrue-last.dat
   - tl1000000e800000s5000ddqnTruewindow10lmr5pc100000swap25000save20000-980000_steps.dat
'''
######################################

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model_dir', default="Models"+os.sep+"bestPlayerP1_98_P2_96.dat", help="Agent's model", type=str, required=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-ep', '--nb_episodes', default=1,
                        help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps", type=int, required=False)
    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-pre', '--predicate', default="win", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="pi", help="Type of strategy for (approximate) HXp",
                        type=str, required=False)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)
    parser.add_argument('-rand', '--random', dest="random", action="store_true", help="Player 2 performs random choices", required=False)
    parser.add_argument('-no_rand', '--no_random', action="store_false", dest="random", help="Player 2 doesn't perform random choices", required=False)
    parser.set_defaults(random=True)
    parser.add_argument('-strats', '--HXp_strategies', default="[pi, last_1, last_2]", help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)

    args = parser.parse_args()

    # Get arguments
    PATHFILE_MODEL = args.model_dir
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K = args.length_k
    NUMBER_EPISODES = args.nb_episodes
    HXP = args.HXP
    STRATEGY = args.strategy
    PREDICATE = args.predicate
    CSV_FILENAME = args.csv_filename
    RENDER = args.render
    history_file = args.specific_history
    SPECIFIC_HISTORY = []
    FIND_HISTORIES = args.find_histories
    N = args.n
    HXP_STRATEGIES = args.HXp_strategies
    RANDOM = args.random

    #  Fill pre-defined history
    if HXP and history_file != "":
        file = open('Histories' + os.sep + history_file, 'r')
        lines = file.readlines()
        state = []
        cpt = 0
        for idx, line in enumerate(lines):
            if idx % 2:
                SPECIFIC_HISTORY.append(int(line[:-1]))
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in line[2:-3].split('], [')]
                SPECIFIC_HISTORY.append(tmp_row)

    # Path to store actions utility in case of HXp
    if HXP:
        utility_dirpath = 'Utility'
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store similarity measure in case of HXp
    if HXP and FIND_HISTORIES:
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Env
    env = Connect4()
    #  Agents
    player_1 = Agent('Yellow', env)
    player_2 = Agent('Red', env, random=RANDOM)
    agents = [player_1, player_2]

    #  Load net(s) -----------------------------------------------------------------------------------------------------
    net = DQN((env.rows, env.cols), env.action_space.n).to(DEVICE)
    net.load_state_dict(torch.load(PATHFILE_MODEL, map_location=DEVICE))
    #  Test ------------------------------------------------------------------------------------------------------------
    if SPECIFIC_HISTORY and HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)
        # Compute HXp
        start_time = time.perf_counter()
        HXpMetric(specific_history, env, STRATEGY, player_1, player_2, utility_csv, PREDICATE, net=net, n=N)
        final_time_s = time.perf_counter() - start_time
        print("Explanation achieved in: {} second(s)".format(final_time_s))

    elif FIND_HISTORIES and HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            #  Reset env
            state = env.reset()
            done = False
            history.put(deepcopy(state))  # initial state
            while not done:
                #  Choose action
                action = player_1.choose_action(state, net, device=DEVICE)
                #  History update
                if history.full():
                    history.get()
                    history.get()
                history.put(deepcopy(action))
                #  Step
                reward, done, state, _, _ = env.step(agents, action, epsilon=0.3, net=net, device=DEVICE)
                #  History update
                history.put(deepcopy(state))
                if valid_history(player_1, env, state, done, reward, list(history.queue)[0], PREDICATE) and history.full():
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:
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
        rewards = []
        s_a_list = []
        episodes = NUMBER_EPISODES
        for i in range(episodes):
            done = False
            env.reset()
            env.render()
            if HXP:
                history = queue.Queue(maxsize=K * 2 + 1)
            cpt = 0
            while not done:
                #  Choose action
                state = env.board
                s_a_list.append(deepcopy(state))
                action = player_1.choose_action(state, net, device=DEVICE)
                s_a_list.append(action)
                #  HXP
                if HXP:
                    history.put(s_a_list[-2])
                    # Compute HXp
                    if cpt and cpt >= N:
                        HXp(history, env, STRATEGY, player_1, player_2, csv_filename=utility_csv, net=net, n=N)
                    # History update
                    if history.full():
                        history.get()
                        history.get()
                    history.put(s_a_list[-1])
                #  Step and update observation
                reward, done, new_state, _, _ = env.step(agents, action, net=net, device=DEVICE)
                #  Render
                env.render()
                if done:
                    s_a_list = []
                    rewards.append((reward+1)/2)
                cpt += 1

            # Compute last HXp
            HXp(history, env, STRATEGY, player_1, player_2, csv_filename=utility_csv, net=net, n=N)

        if episodes > 1:
            print(sum(rewards))
            print((sum(rewards)/len(rewards))*100)
            print('Win rate: {}% over {} episodes vs Random agent'.format(((sum(rewards)/len(rewards))*100), episodes))


