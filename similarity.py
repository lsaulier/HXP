
import argparse
import csv
import os
import queue
import numpy as np
import torch
from scipy.spatial import distance

#  FL
import FrozenLake.env as FL_env
import FrozenLake.agent as FL_agent
import FrozenLake.HXp as FL_HXp

#  DC
import DroneCoverage.env as DC_env
import DroneCoverage.agent as DC_agent
import DroneCoverage.HXp as DC_HXp
import DroneCoverage.DQN as DC_DQN

#  C4
import Connect4.env as C4_env
import Connect4.agent as C4_agent
import Connect4.HXp as C4_HXp
import Connect4.DQN as C4_DQN

#  Parse a history of a problem
#  Input: history (str) and studied problem (str)
#  Output: state-action sequence, i.e. a history (int list list)
def parse(str_history, problem):
    # Remove []
    str_history = str_history[1:-1]
    # Parse String
    if problem == 'FL':
        history = [int(i) for i in str_history.split(', ')]
    elif problem == 'DC':
        #  Create string history
        str_states_actions = str_history.split('], [[')
        str_states = ['[['+ sa.split(']]')[0] + ']]' if idx else sa.split(']]')[0] + ']]' for idx, sa in enumerate(str_states_actions)]
        str_actions = [sa.split(']], ')[1] + ']' for sa in str_states_actions[:-1]]
        str_history = []
        for i in range(len(str_actions)):
            str_history.append(str_states[i])
            str_history.append(str_actions[i])
        str_history.append(str_states[-1])
        #  Create history
        history = []
        for idx, elm in enumerate(str_history):
            if idx % 2:
                tmp_row = [int(t) for t in elm[1:-1].split(', ')]
                history.append(tmp_row)
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in elm[2:-2].split('], [')]
                history.append(tmp_row)
    else:
        #  Create string history
        str_states_actions = str_history.split(', [[')
        str_states = ['[['+ sa.split(']]')[0] + ']]' if idx else sa.split(']]')[0] + ']]' for idx, sa in enumerate(str_states_actions)]
        str_actions = [sa[-1] for sa in str_states_actions[:-1]]
        str_history = []
        for i in range(len(str_actions)):
            str_history.append(str_states[i])
            str_history.append(str_actions[i])
        str_history.append(str_states[-1])
        #  Create history
        history = []
        for idx, elm in enumerate(str_history):
            if idx % 2:
                history.append(int(elm))
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in elm[2:-2].split('], [')]
                history.append(tmp_row)

    # Put in a queue
    queue_history = queue.Queue(maxsize=len(history))
    for elm in history:
        queue_history.put(elm)
    return queue_history

#  Write the first line of a CSV file
#  Input: strategies used for the computation of action importance scores (str list)
#  Output: first line of the CSV file (str list)
def first_line(strats):
    line = ['History']
    for method in strats:
        line.append(method)
        line.append('time of '+method)
    for i in range(1, len(strats)):
        line.append(strats[0] + ' -- ' + strats[i])
    return line


if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', '--file', default="", help="History file", type=str, required=True)
    parser.add_argument('-new_file', '--new_file', default="", help="Store history, importance scores and time", type=str, required=True)
    parser.add_argument('-pre', '--predicate', default="", help="predicate to verify", type=str, required=True)
    parser.add_argument('-pre_info', '--predicate_additional_info', default=None, help="Specify a state", type=str, required=False)
    parser.add_argument('-problem', '--problem', default="", help="considered problem", type=str, required=True)
    parser.add_argument('-strats', '--HXp_strategies', default="[pi, last_1, last_2, last_3, last_4]",
                        help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-policy', '--policy', default="", help="Policy name/file", type=str,
                        required=True)
    parser.add_argument('-map', '--map_name', default="10x10", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int,
                        required=False)
    parser.add_argument('-id', '--agent_id', default=0, help="Drone to focus on during HXp", type=int, required=False)

    args = parser.parse_args()

    # Get arguments
    FILE = args.file
    NEW_FILE = args.new_file
    PREDICATE = args.predicate
    ADD_INFO = args.predicate_additional_info
    PROBLEM = args.problem
    STRATEGIES = args.HXp_strategies.split(', ')
    STRATEGIES[0] = STRATEGIES[0][1:]
    STRATEGIES[-1] = STRATEGIES[-1][:-1]
    POLICY = args.policy
    # FL argument(s)
    MAP = args.map_name
    # DC argument(s)
    NUMBER_AGENTS = args.number_agents
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ID = args.agent_id

    # Initialize agent(s) and environment
    if PROBLEM == 'FL':

        # Convert additional information
        if ADD_INFO is not None:
            if ADD_INFO[0] == '[':
                ADD_INFO = [int(i) for i in ADD_INFO[1:-1].split(', ')]
            else:
                ADD_INFO = int(ADD_INFO)

        agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
        #  Env initialization
        env = FL_env.MyFrozenLake(map_name=MAP, slip_probas=[0.2, 0.6, 0.2])
        #  Agent initialization
        agent = FL_agent.Agent(POLICY, env)
        #  Load Q table
        agent.load(agent_Q_dirpath)

    elif PROBLEM == 'DC':
        #  Env initialization
        env = DC_env.DroneAreaCoverage(map_name=MAP)
        #  Agent initialization
        agents = []
        for i in range(NUMBER_AGENTS):
            agent = DC_agent.Agent(i + 1, env)
            agents.append(agent)
        env.initPos(agents, True)
        env.initObs(agents)
        #  Net loading
        abs_dir_path = os.getcwd()
        net = DC_DQN.DQN(np.array(agent.observation[0]).shape, np.array(agent.observation[1]).shape, agent.actions).to(DEVICE)
        net.load_state_dict(torch.load(abs_dir_path + POLICY, map_location=DEVICE))

    else:
        #  Env initialization
        env = C4_env.Connect4()
        #  Agents initialization
        player_1 = C4_agent.Agent('Yellow', env)
        player_2 = C4_agent.Agent('Red', env)
        #  Net Loading
        abs_dir_path = os.getcwd()
        net = C4_DQN.DQN((env.rows, env.cols), env.action_space.n).to(DEVICE)
        net.load_state_dict(torch.load(abs_dir_path + POLICY, map_location=DEVICE))

    # Read first cell of all lines to get the History
    abs_dir_path = os.getcwd()
    with open(abs_dir_path + FILE, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            print(row)
            if row[0] not in ['History', '']:
                d = []
                history = parse(row[0], PROBLEM)
                # Run HXp for each strategy
                if PROBLEM == 'FL':
                    scores, times = FL_HXp.HXpSimilarityMetric(history, env, STRATEGIES, agent, property=PREDICATE, property_info=ADD_INFO)
                elif PROBLEM == 'DC':
                    scores, times = DC_HXp.HXpSimilarityMetric(history, env, STRATEGIES, agents, property=PREDICATE, agent_id=ID, net=net)
                else:
                    scores, times = C4_HXp.HXpSimilarityMetric(history, env, STRATEGIES, player_1, player_2, property=PREDICATE, net=net)
                # Store data (history - strategy imp score - strategy computational time)
                d.append(list(history.queue))
                for i in range(len(scores)):
                    d.append(scores[i])
                    d.append(times[i])
                # Compute and store data (similarity scores)
                for i in range(1, len(scores)):
                    d.append(distance.euclidean(scores[0], scores[i]))
                data.append(d)

    # Write stored data
    with open(abs_dir_path + os.sep + NEW_FILE, 'a') as f:
        writer = csv.writer(f)
        # First Line
        line = first_line(STRATEGIES)
        writer.writerow(line)
        # Data
        for d in data:
            writer.writerow(d)
        # Means & Std
        index = 2
        line_avg = [''] * index
        line_std = [''] * index
            # Time
        for i in range(index, len(data[0]) - len(STRATEGIES) + 2, 2):
            times = [d[i] for d in data]
            avg = np.mean(times, axis=0)
            std = np.std(times, axis=0)
            if i != len(data[0]) - len(STRATEGIES):
                line_avg.extend([avg, ''])
                line_std.extend([std, ''])
            else:
                line_avg.extend([avg])
                line_std.extend([std])
            # Similarity
        for i in range(len(data[0]) - len(STRATEGIES) + 1, len(data[0])):
            sim = [d[i] for d in data]
            avg = np.mean(sim, axis=0)
            std = np.std(sim, axis=0)
            line_avg.extend([avg])
            line_std.extend([std])
        writer.writerow(line_avg)
        writer.writerow(line_std)

