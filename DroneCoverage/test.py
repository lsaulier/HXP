import csv
import queue
import time

from DQN import DQN
from agent import Agent
from env import DroneAreaCoverage
from HXp import HXp, HXpMetric, valid_history
import torch
import os
import argparse
import numpy as np

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Path
    parser.add_argument('-model', '--model_dir', default="Models"+os.sep+"Agent"+os.sep+"tl1600000e750000s50000th22ddqnTrue-best_11.69.dat", help="Agent's model", type=str, required=False)

    parser.add_argument('-map', '--map_name', default="10x10", help="Map's name", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1,
                        help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-horizon', '--time_horizon', default=20, help="Time horizon of an episode", type=int, required=False)
    parser.add_argument('-rand', '--random_starting_position', action="store_true", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.add_argument('-no_rand', '--no_random_starting_position', action="store_false", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.set_defaults(random_starting_position=True)
    parser.add_argument('-move', '--step_move', default="stop", help="Type of transition with wind", type=str, required=False)
    parser.add_argument('-view', '--view_range', default=5, help="View range of a drone", type=int, required=False)
    parser.add_argument('-w', '--wind', action="store_false", dest='windless', help="Wind's presence in the environment", required=False)
    parser.add_argument('-no_w', '--no_wind', action="store_true", dest='windless', help="Wind's presence in the environment", required=False)
    parser.set_defaults(windless=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps", type=int, required=False)

    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="pi", help="Type of strategy for (approximate) HXp",
                        type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-id', '--agent_id', default=0, help="Drone to focus on during HXp", type=int, required=False)
    parser.add_argument('-pre', '--predicate', default="perfect cover", help="Predicate to verify in the history", type=str,
                        required=False)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)
    parser.add_argument('-strats', '--HXp_strategies', default="[pi, last_1, last_2]", help="Exploration strategies for similarity measures", type=str, required=False)

    args = parser.parse_args()
    # Get arguments
    PATHFILE_MODEL = args.model_dir
    MAP_NAME = args.map_name
    NUMBER_EPISODES = args.nb_episodes
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUMBER_AGENTS = args.number_agents
    VIEW_RANGE = args.view_range
    WINDLESS = args.windless
    RANDOM_STARTING_POSITION = args.random_starting_position
    MOVE = args.step_move
    LIMIT = args.time_horizon
    K = args.length_k
    # only used if SPECIFIC_STATE
    CSV_FILENAME = args.csv_filename
    RENDER = args.render
    # used for HXp
    HXP = args.HXP
    history_file = args.specific_history
    AGENT_ID = args.agent_id
    STRATEGY = args.strategy
    PREDICATE = args.predicate
    FIND_HISTORIES = args.find_histories
    N = args.n
    HXP_STRATEGIES = args.HXp_strategies

    if history_file != "":
        history_file = 'Histories' + os.sep + history_file

    #  Fill pre-defined history
    SPECIFIC_HISTORY = []
    if HXP and history_file != "":
        file = open(history_file, 'r')
        lines = file.readlines()
        state = []
        cpt = 0
        for idx, line in enumerate(lines):
            if idx % 2:
                tmp_row = [int(t) for t in line[1:-2].split(', ')]
                SPECIFIC_HISTORY.append(tmp_row)
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in line[2:-3].split('], [')]
                SPECIFIC_HISTORY.append(tmp_row)

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
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Environment
    env = DroneAreaCoverage(map_name=MAP_NAME, windless=WINDLESS)
    #  Agents
    agents = []
    for i in range(NUMBER_AGENTS):
        agent = Agent(i + 1, env, view_range=VIEW_RANGE)
        agents.append(agent)

    env.initPos(agents, RANDOM_STARTING_POSITION)
    env.initObs(agents)

    #  Load net -----------------------------------------------------------------------------------------------------
    net = DQN(np.array(agent.observation[0]).shape, np.array(agent.observation[1]).shape, agent.actions).to(DEVICE)
    net.load_state_dict(torch.load(PATHFILE_MODEL, map_location=DEVICE))

    #  Test ------------------------------------------------------------------------------------------------------------

    if SPECIFIC_HISTORY and HXP:
        # Clean environment
        env.clear_map()
        for agent in agents:
            agent.set_env(env)
        # Fill in the history
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)
        # Compute HXp
        start_time = time.perf_counter()
        HXpMetric(specific_history, env, STRATEGY, agents, utility_csv, PREDICATE, agent_id=AGENT_ID, net=net, n=N)
        final_time_s = time.perf_counter() - start_time
        print("Explanation achieved in: {} second(s)".format(final_time_s))

    elif FIND_HISTORIES and HXP:
        env.reset(agents, rand=RANDOM_STARTING_POSITION)
        nb_scenarios = NUMBER_EPISODES
        storage = []
        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            #  Reset env
            env.reset(agents, rand=RANDOM_STARTING_POSITION)
            done = False
            #  Store current agent's positions
            old_positions = [agent.get_obs()[1] for agent in agents]
            history.put(old_positions)
            cpt_stop = 0
            cpt = 0
            while cpt <= LIMIT:
                #  Choose action
                actions = []
                for agent in agents:
                    action = agent.chooseAction(net, epsilon=0, device=DEVICE)
                    actions.append(action)
                #  History update
                if cpt_stop < 2:
                    if history.full():
                        history.get()
                        history.get()
                    history.put(actions)
                #  Step
                _, _, _, dones, _ = env.step(agents, actions, move=MOVE)
                #  Store current agent's positions
                positions = [agent.get_obs()[1] for agent in agents]
                if old_positions == positions:
                    cpt_stop += 1
                #  History update
                if cpt_stop < 2:
                    history.put(positions)
                    if valid_history(agents, AGENT_ID, PREDICATE) and history.full():
                        data = [list(history.queue)]
                        storage.append(data)
                        break  # since we start from random configs, this will allow more diverse histories

                #  Check end of episode
                if dones.count(True) == len(dones) or dones[AGENT_ID -1]:
                    break

                cpt += 1

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

        env.reset(agents, rand=RANDOM_STARTING_POSITION)
        cpt = 0
        env.render(agents)
        if HXP:
            history = queue.Queue(maxsize=K * 2 + 1)

        while cpt <= LIMIT:
            #  Choose action
            actions = []
            for agent in agents:
                action = agent.chooseAction(net, epsilon=0, device=DEVICE)
                actions.append(action)
            #  Compute HXp
            if HXP:
                history.put([agent.get_obs()[1] for agent in agents])
                # Compute HXp
                if cpt and cpt >= N:
                    HXp(history, env, STRATEGY, agents, csv_filename=utility_csv, net=net, n=N)
                # Update history
                if history.full():
                    # Update history
                    history.get()
                    history.get()
                history.put(actions)

            #  Step
            _, _, _, dones, _ = env.step(agents, actions, move=MOVE)
            #  Render

            env.render(agents)
            #  Extract rewards
            rewards = env.getReward(agents, actions, dones, reward_type="B")
            #  Check end of episode
            if dones.count(True) == len(dones):
                break

            #  Display infos
            print("Dones True : {}".format(dones.count(True)))
            print("Rewards : {}".format(rewards))
            print("Cumulative reward : {}".format(sum(rewards)))
            print('-------')

            cpt += 1