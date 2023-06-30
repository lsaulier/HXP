import argparse
import os
from env import MyFrozenLake
from agent import Agent

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4map_test", help="Common part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=10000, help="Number of training episodes", type=int, required=False)

    args = parser.parse_args()
    
    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes

    # Paths to store Q table
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"

    #  Env initialization
    env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2])

    #  Agent initialization
    agent = Agent(POLICY_NAME, env)

    #  Train
    agent.train(NB_EPISODES)
    print("End of training")

    #  Save Q table
    agent.save(agent_Q_dirpath)

    # Delete agent
    del agent
