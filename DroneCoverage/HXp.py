from copy import deepcopy
import time

import numpy as np
import csv


#  Main function which manages user queries about action and state importance
#  Input: the history (int list), environment (DroneAreaCoverage), strategy exploration (str) agents (Agent list),
#  CSV file to store scores (str), policy (DQN) and number of action to highlight (int)
#  Output: None
def HXp(history, env, strat='pi', agents=None, csv_filename="", net=None, n=2):
    answer = False
    good_answers = ["yes", "y"]
    history = list(history.queue)
    print("History : {}".format(history))
    while not answer:

        question = "Do you want an HXp?"
        action_HXp = input(question)

        # Provide an HXp
        if action_HXp in good_answers:

            property_question = "Which predicate do you want to test? (cover/max reward/no drones/region) \n"
            property = input(property_question)
            id_question = "From which agent do you want the important actions? Blue (1), Green (2), Red (3) or Yellow (4)?"
            agent_id = int(input(id_question))
            context_question = "Does the predicate have to be analyzed for all agents or only the selected one? (one/all)"
            number = input(context_question)
            if property == 'cover':
                cover_type_question = "Perfect or Imperfect cover? (perfect/imperfect)"
                cover_type = input(cover_type_question)
                predicate = number + ' ' + cover_type + ' ' + property
                print('Predicate: {}'.format(predicate))
                HXp_actions, _, _ = HXp_action(history, env, strat, agents, predicate, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)

            elif property == 'max reward':
                predicate = number + ' ' + property
                print('Predicate: {}'.format(predicate))
                HXp_actions, _, _ = HXp_action(history, env, strat, agents, predicate, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)

            elif property == 'no drones':
                predicate = number + ' ' + property
                print('Predicate: {}'.format(predicate))
                HXp_actions, _, _ = HXp_action(history, env, strat, agents, predicate, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)

            elif property == 'region':
                predicate = number + ' ' + property + 's' if number == 'all' else number + ' ' + property
                if number == 'one':
                    region_question = "Region? (1/2/3/4)"
                    region = input(region_question)
                    predicate = predicate + ' ' + region
                print('Predicate: {}'.format(predicate))
                HXp_actions, _, _ = HXp_action(history, env, strat, agents, predicate, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)
            else:
                print('None property was selected!')
                HXp_actions = 0
            # Render important actions
            if HXp_actions:
                render_actionImportance(HXp_actions, env, agent_id)

        answer = True

    return


#  Compute HXp for a history with each strategy
#  Input: the history (int  list list), environment (DroneCoverageArea), strategies to use (str list), agents
#  (Agent list), CSV file to store scores (str), predicate to verify (str), agent to explain (int), the policy (DQN) and
#  number of action to highlight (int)
#  Output: importance scores for each action in the history for each strategy used (float list list)
def HXpSimilarityMetric(history, env, strats, agents, csv_filename='trash.csv', property=None, agent_id=0, net=None, n=2):

    history = list(history.queue)
    print("History : {}".format(history))
    scores = []
    times = []
    for strat in strats:
        _, importance_scores, HXp_time = HXp_action(history, env, strat, agents, property, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)

        times.append(HXp_time)
        scores.append(importance_scores)

    return scores, times


#  Compute HXp for a history with each strategy
#  Input: the history (int  list list), environment (DroneCoverageArea), strategy to use (str), agents (Agent list), CSV
#  file to store scores (str), predicate to verify (str), agent to explain (int), the policy (DQN) and number
#  of action to highlight (int)
#  Output: importance scores for each action in the history (float list)
def HXpMetric(history, env, strat, agents, csv_filename, property, agent_id=0, net=None, n=2):

    history = list(history.queue)
    print("History : {}".format(history))
    HXp_actions, scores, _ = HXp_action(history, env, strat, agents, property, csv_filename=csv_filename, agent_id=agent_id, net=net, n=n)
    # Render important actions
    print("Rendering of important actions using the {} strategy".format(strat))
    render_actionImportance(HXp_actions, env, agent_id)
    return scores


#  Compute the action importance of each action in the history
#  Input: the history (int list), environment (DroneCoverageArea), exploration strategy (str), agents (Agent list),
#  predicate to verify (str), CSV file to store scores (str), agent to explain (int), the policy (DQN) and
#  number of action to highlight (int)
#  Output: list of important actions, states, time-steps associated (int list list) and list of scores (float list)
def HXp_action(H, env, strat='pi', agents=None, property='perfect cover', csv_filename="", agent_id=0, net=None, n=2):
    HXp_actions = []
    score_list = []
    available_actions = [0, 1, 2, 3, 4]
    firstLineCsvTable(strat, property, csv_filename)  # head of table in CSV file
    fixed_k = ((len(H)-1)//2) - 1
    env_copy = deepcopy(env)
    agents_copy = deepcopy(agents)
    for agent in agents_copy:
        agent.dead = False
    start_time = time.perf_counter()
    for i in range(0, len(H)-1, 2):
        positions, actions = H[i], H[i+1]
        actions_histo = deepcopy(actions)
        env_copy.clear_map()
        for idx, agent in enumerate(agents_copy):
            agent.env = env_copy
            env_copy.set_initPos(agent, positions[idx])

        env_copy.set_lastactions(actions_histo)
        env_copy.initObs(agents_copy)
        agents_copy_2 = deepcopy(agents_copy)
        env_copy_2 = deepcopy(env_copy)

        score = actionImportance(actions_histo, env_copy_2, available_actions, fixed_k, strat, agents_copy_2, property, csv_filename, net, agent_id)
        score_list.append([score, actions, deepcopy(agents_copy), i // 2])

    # Rank the scores to select the n most important action
    print('Strategy: {}'.format(strat))
    tmp_list = [elm[0] for elm in score_list]
    scores = [round(item, 3) for item in tmp_list]
    print("Scores : {}".format(scores))
    top_idx = np.sort(np.argpartition(np.array(tmp_list), -n)[-n:])
    print("Top {} index(es) : {}".format(n, top_idx))
    for i in range(n):
        HXp_actions.append(score_list[top_idx[i]][1:])
    final_time_s = time.perf_counter() - start_time
    return HXp_actions, scores, final_time_s


#  Compute the importance of an action 'a' from a state 's'
#  Input: agents' actions (int list), environment (DroneCoverageArea), available actions from s (int list), trajectories
#  length (int), exploration strategy (str), agents (Agent list),  predicate to verify (str), CSV file to store scores
#  (str), policy (DQN) and agent to explain (int)
#  Output: importance of a from s (float)
def actionImportance(actions, env, available_actions, k, strat='pi', agents=None, property='perfect cover', csv_filename="", net=None, agent_id=0):
    agents_copy = deepcopy(agents)
    if strat != 'pi':
        deterministic_transition = int(strat.split('_')[1])
    else:
        deterministic_transition = 0
    #  Approximate approach
    if deterministic_transition:
        # The last steps use deterministic transitions
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(actions, available_actions, agents, deepcopy(env), agent_id)
        # Get two sets representing the final states
        env.clear_map()
        Sf_a = succ(Ac_a, k, env, agents, agent_id, det_transition=deterministic_transition, net=net)
        env.clear_map()
        Sf_nota = succ(Ac_nota, k, env, agents, agent_id, det_transition=deterministic_transition, net=net)

        # Count in both sets the number of states which respects the property
        return checkPredicate([[agent.get_obs()[1] for agent in agents_copy], actions[agent_id - 1]], Sf_a, Sf_nota, agents, agent_id, property, csv_filename)

    #  Exhaustive approach
    else:
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(actions, available_actions, agents, env, agent_id)
        # Get two sets representing the final states
        env.clear_map()
        Sf_a = succ(Ac_a, k, env, agents, agent_id, net=net) if k else Ac_a

        env.clear_map()
        Sf_nota = succ(Ac_nota, k, env, agents, agent_id, net=net) if k else Ac_nota

        # Count in both sets the number of states which respects the property
        return checkPredicate([[agent.get_obs()[1] for agent in agents_copy], actions[agent_id - 1]], Sf_a, Sf_nota, agents, agent_id, property, csv_filename)


#  Explore trajectories from a set of initial states using only deterministic transitions
#  Input: number of deterministic transition (int), list of current starting states (int list or int list list),
#  environment (DroneCoverageArea), agents (Agent list), agent to explain (int), policy (DQN) and device (str).
#  Output: list of couple final states and associated proba (tuple list list or tuple list list list)
def succ_detTrLoop_module(det_transition, S, env, agents, agent_id, net, device):
    S_tmp = []
    for i in range(det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        # This part used when deterministic transition > 1, and we compute successors based on all not a actions
        # different from pi(s). It could also be used if we first explore search space with classic 'pi' and
        # then want deterministic transitions (but updates might be necessary)
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    # Extract states, proba
                    states, proba = s
                    if not terminalState(states, agent_id):
                        actions = [agent.chooseAction(net, device=device, obs=states[idx]) for idx, agent in enumerate(agents)]
                        new_states, _ = transition(agents, agent_id, actions, deepcopy(env), det_tr=True)[0]
                        # Succession of probabilities
                        S_tmp_sublist.append((new_states, proba))
                    else:
                        # Add the terminal state
                        S_tmp_sublist.append((states, proba))

                S_tmp.append(S_tmp_sublist)
            S = S_tmp
            S_tmp = []

        else:
            # Used for generating the list of reachable final states from the current history's action
            for s in S:
                # Extract states, proba
                states, proba = s
                if not terminalState(states, agent_id):
                    actions = [agent.chooseAction(net, device=device, obs=states[idx]) for idx, agent in enumerate(agents)]
                    new_states, _ = transition(agents, agent_id, actions, deepcopy(env), det_tr=True)[0]
                    # Succession of probabilities
                    S_tmp.append((new_states, proba))
                else:
                    # Add the terminal state
                    S_tmp.append((states, proba))

            S = S_tmp
            S_tmp = []

    return S, S_tmp


#  Explore trajectories from a set of initial states
#  Input: set of initial states (int list), trajectories length (int), environment (DroneCoverageArea), agents
#  (Agent list), the agent to explain (int), number of deterministic transitions (int), policy (DQN) and device (str)
#  Output: list of couples final states and associated proba, i.e.  the reachable states at horizon k, starting from
#  each state s in S (tuple list list or tuple list list list)
def succ(S, k, env, agents, agent_id, det_transition=0, net=None, device="cpu"):
    S_tmp = []
    if det_transition > k:
        det_transition = k
    #  ####################################### No Det. Tr. ######################################################  #
    # Explore each transition
    for i in range(k-det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        if len(S[0]) != 2:
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    states, proba = s
                    if not terminalState(states, agent_id):
                        actions = [agent.chooseAction(net, device=device, obs=states[idx]) for idx, agent in enumerate(agents)]
                        for new_states, p in transition(agents, agent_id, actions, deepcopy(env)):
                            # Succession of probabilities
                            S_tmp_sublist.append((new_states, proba * p))
                    else:
                        # Add the terminal state
                        S_tmp_sublist.append((states, proba))
                S_tmp.append(S_tmp_sublist)
            S = S_tmp
            S_tmp = []

        # Used for generating the list of reachable final states from the current action
        else:
            for s in S:
                states, proba = s
                if not terminalState(states, agent_id):
                    actions = [agent.chooseAction(net, device=device, obs=states[idx]) for idx, agent in enumerate(agents)]
                    for new_states, p in transition(agents, agent_id, actions,  deepcopy(env)):
                        # Succession of probabilities
                        S_tmp.append((new_states, proba * p))
                else:
                    # Add the terminal state
                    S_tmp.append((states, proba))

            S = S_tmp
            S_tmp = []

    #  ####################################### Last Det. Tr. ####################################################  #
    #  Perform the last deterministic transitions
    #  These transitions are imposed and deterministic, then their proba is always 1.0, hence we keep unchanged proba
    if det_transition:
        S, S_tmp = succ_detTrLoop_module(det_transition, S, env, agents, agent_id, net, device)
    return S


#  Compute utilities and compare them to decide the action importance of a
#  Input: states-actions list (int list list), lists of final states (int list list and int list list list), agents
#  (Agent list), the agent to explain (int),  predicate to verify (str) and CSV file to store scores (str)
#  Output: action importance (float)
def checkPredicate(coord_a_list, Sf, Sf_not, agents, agent_id, property='perfect cover', csv_filename=""):
    # 'one' or 'all'
    one_drone = property[:3] == 'one'
    property_tmp = property[4:]
    agent = agents[agent_id - 1]
    coord_a_list.append(agent_id)
    # Important action if the probability of reaching a perfect cover is higher when doing action a
    if property_tmp == 'perfect cover':
        # Path probabilities to reach a perfect cover by doing a from s
        Sf_win = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if not crash(agent) and perfect_cover(agent):
                    Sf_win += p
            else:
                if perfect_covers(agents, states):
                    Sf_win += p
        # Path probabilities to reach a perfect cover by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if not crash(agent) and perfect_cover(agent):
                        tmp_sublist_counter += p
                else:
                    if perfect_covers(agents, states):
                        tmp_sublist_counter += p

            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_win, Sf_not_best, Sf_not_average, probas])

    # Important action if the probability of reaching an imperfect cover is higher when doing action a
    elif property_tmp == 'imperfect cover':
        # Path probabilities to reach an imperfect cover by doing a from s
        Sf_lose = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if not crash(agent) and imperfect_cover(agent):
                    Sf_lose += p
            else:
                if imperfect_covers(agents, states):
                    Sf_lose += p

        # Path probabilities to reach an imperfect cover by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if not crash(agent) and imperfect_cover(agent):
                        tmp_sublist_counter += p
                else:
                    if imperfect_covers(agents, states):
                        tmp_sublist_counter += p
            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_lose, Sf_not_best, Sf_not_average, probas])

    # Important action if the crash probability is higher when doing action a
    elif property_tmp == 'crash':
        # Path probabilities to reach a configuration where the drone is crashed by doing a from s
        Sf_crash = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if crash(agent):
                    Sf_crash += p
            else:
                if crashes(agents, states):
                    Sf_crash += p

        # Path probabilities to reach a configuration where the drone is crashed  by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if crash(agent):
                        tmp_sublist_counter += p
                else:
                    if crashes(agents, states):
                        tmp_sublist_counter += p

            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_crash, Sf_not_best, Sf_not_average, probas])

    # Important action if the probability of obtaining a max reward is higher when doing action a
    elif property_tmp == 'max reward':
        # Path probabilities to get the maximum reward by doing a from s
        Sf_max_reward = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if max_reward(agent):
                    Sf_max_reward += p
            else:
                if all_max_reward(agents, states):
                    Sf_max_reward += p

        # Path probabilities to get the maximum reward by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if max_reward(agent):
                        tmp_sublist_counter += p
                else:
                    if all_max_reward(agents, states):
                        tmp_sublist_counter += p

            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_max_reward, Sf_not_best, Sf_not_average, probas])

    # Important action if the probability of reaching a specific region of the map is higher when doing action a
    elif property_tmp[:-2] == 'region' or property_tmp == 'regions':
        if one_drone:
            region_idx = int(property[-1])
        # Path probabilities to get the maximum reward by doing a from s
        Sf_region = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if not crash(agent) and region(agent, region_idx):
                    Sf_region += p
            else:
                if regions(agents, states):
                    Sf_region += p

        # Path probabilities to get the maximum reward by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if not crash(agent) and region(agent, region_idx):
                        tmp_sublist_counter += p

                else:
                    if regions(agents, states):
                        tmp_sublist_counter += p
            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_region, Sf_not_best, Sf_not_average, probas])

    # Important action if the probability of having no drones in its neighbourhood is higher when doing action a
    elif property_tmp == 'no drones':
        # Path probabilities to get the maximum reward by doing a from s
        Sf_no_drones = 0.0
        for (states, p) in Sf:
            if one_drone:
                agent.set_obs(states[agent_id - 1])
                if not crash(agent) and no_drones(agent):
                    Sf_no_drones += p
            else:
                if all_no_drones(agents, states):
                    Sf_no_drones += p

        # Path probabilities to get the maximum reward by not doing a from s
        probas = []
        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (states, p) in sublist:
                if one_drone:
                    agent.set_obs(states[agent_id - 1])
                    if not crash(agent) and no_drones(agent):
                        tmp_sublist_counter += p
                else:
                    if all_no_drones(agents, states):
                        tmp_sublist_counter += p

            probas.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        Sf_not_best = max(probas)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, coord_a_list, [Sf_no_drones, Sf_not_best, Sf_not_average, probas])


#  Store all info about an action importance in a CSV file and compute importance score
#  Input: CSV file (str), state-action (int list), list of info to store in the CSV file (int list list).
#  Output: an action importance score (float)
def storeInfoCsv(csv_filename, coord_a_list, info_list):

    Sf_goal, Sf_not_best, Sf_not_average, tmp_counter = info_list

    # --------------------------- Display info -------------------------------------
    coordinates, action, agent_id = coord_a_list
    '''
    print(
        "By doing action {} from those drones coordinates {} ({}: coordinates of the agent to explain), "
        "the predicate is respected in {}% \n"
        "By doing the best contrastive action, it is respected in {}% \n"
        "By not doing the action {}, the average respect of the predicate is {}% \n".format(action, coordinates, coordinates[agent_id - 1], Sf_goal * 100, Sf_not_best * 100, action, Sf_not_average * 100))
    '''
    #  ------------------------ Store in CSV ----------------------------------------
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        # Action importance of history's action
        row = ['{}-{}-{}'.format(coordinates, coordinates[agent_id - 1], action), 'Action : {}'.format(action), Sf_goal]
        writer.writerow(row)
        # Action importance of the best different action
        row = ['', 'Other actions: Best', Sf_not_best]
        writer.writerow(row)
        # Average action importance of different actions
        writer.writerow(['', 'Other actions: Average', tmp_counter, Sf_not_average])

        writer.writerow('')  # blank line

    return Sf_goal - Sf_not_average


#  Render action importance (HXp)
#  Input: list of tuples (actions - agent - timestep) (list), environment (DroneCoverageArea) and agent id (int)
#  Output: None
def render_actionImportance(hxp_actions, env, agent_id):
    for actions, agents, i in hxp_actions:
        if agent_id:
            print("Timestep {} --- Action from drone {}:  {}".format(i, agent_id, actions))
        env.clear_map()
        env.set_lastactions(actions)
        for agent in agents:
            env.set_initPos(agent, agent.get_obs()[1])
        env.render(agents)
    print("------ End of action importance explanation ------")
    return


#  Compute the set of reachable states from 's' by doing 'a' and the set of reachable states from 's' by doing
#  a different action from 'a'.
#  Input: agents' actions (int list), available actions for agent_id (int list), agents (Agent list), environment
#  (DroneCoverageArea) and the agent id (int)
#  Output: set of reachable states from 's' doing 'a' (int list list) and set of reachable states from 's'
#  doing a different action from 'a' (int list list list)
def get_Ac(actions, available_actions, agents, env, agent_id):
    Ac_a = []
    Ac_nota = []
    Ac_nota_tmp = []
    agent_id_action = actions[agent_id - 1]
    # HXp for one agent
    if agent_id:
        for a in available_actions:
            actions[agent_id - 1] = a
            for new_states, p in transition(agents, agent_id, actions, env):
                if a == agent_id_action:
                    Ac_a.append((new_states, p))
                else:
                    Ac_nota_tmp.append((new_states, p))

            if a != agent_id_action:
                Ac_nota.append(Ac_nota_tmp)
                Ac_nota_tmp = []

    return Ac_a, Ac_nota


#  Get all possible transitions from agents' states and actions. It's similar to a simple step function but
#  the transitions for the other agents than agent_id are fixed (most probable transition).
#  This function handles deterministic transition from the agent_id.
#  Input: agents (Agent list), the agent id (int), agents actions (int list), environment (DroneCoverageArea) and use of
#  a deterministic transitions for all agents (bool)
#  Output: list of new states 's'' and associated proba (tuple list)
def transition(agents, agent_id, actions, env, det_tr=False):
    states = get_states(agents)
    new_positions = []
    transitions = []
    new_states = []
    #  Update map and positions
    for i in range(len(actions)):
        env.map[states[i][1][0]][states[i][1][1]] = 1
        new_position = env.inc(states[i][1][0], states[i][1][1], actions[i])
        new_positions.append(new_position)
    #  Due to transition probabilities, new positions change
    if not env.windless:
        if det_tr:
            agents_copy = deepcopy(agents)
            new_positions_copy = deepcopy(new_positions)  # not useful
            env_copy = deepcopy(env)
            #  Most probable transition and associated probability
            #  Transition is done for each drone
            transition, p = argmax(env.P), max(env.P)

            for i in range(len(new_positions_copy)):
                #  Change position if actions are not opposite and not 'stop' or if action is 'stop'
                if actions[i] != 4 and not (actions[i] - 2 == transition or transition - 2 == actions[i]):
                    new_positions_copy[i] = env.inc(new_positions_copy[i][0], new_positions_copy[i][1], transition)

            #  Update self.map, agents.dead and dones
            env_copy.collisions(agents_copy, new_positions_copy)
            #  Update observations of agents
            for i in range(len(agents_copy)):
                new_states.append(
                    [agents_copy[i].view(new_positions_copy[i], optional_map=env_copy.map), new_positions_copy[i]])

            transitions.append((new_states, p))

        else:
            for idx, p in enumerate(env.P):
                agents_copy = deepcopy(agents)
                new_positions_copy = deepcopy(new_positions)
                env_copy = deepcopy(env)

                for i in range(len(new_positions_copy)):
                    #  Most probable transition for other agents or the agent to explain in case of deterministic transition
                    if i + 1 != agent_id:
                        transition = argmax(env.P)
                    #  Agent to explain
                    else:
                        transition = idx

                    #  Change position if actions are not opposite and not 'stop' or if action is 'stop'
                    if actions[i] != 4 and not (actions[i] - 2 == transition or transition - 2 == actions[i]):
                        new_positions_copy[i] = env.inc(new_positions_copy[i][0], new_positions_copy[i][1], transition)
                #  Update self.map, agents.dead and dones
                env_copy.collisions(agents_copy, new_positions_copy)
                #  Update observations of agents
                for i in range(len(agents_copy)):
                    new_states.append([agents_copy[i].view(new_positions_copy[i], optional_map=env_copy.map), new_positions_copy[i]])

                transitions.append((new_states, p))
                new_states = []

    return transitions


#  Get available actions
#  Input: the environment (DroneAreaCoverage)
#  Output: list of actions (int list)
def get_availableActions(env):
    return [i for i in range(env.action_space.n)]


#  Boolean function which verifies one of those predicates:
#  'The agent's state is terminal'/'Each agent's state is terminal'
#  Input: agents' states (int list list list) and the agent id (int)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def terminalState(states, agent_id):
    # HXp for a specific drone
    if agent_id:
        state = states[agent_id - 1]
        return state[0][len(state[0])//2][len(state[0][0])//2] != 2
    # HXp for all drones simultaneously
    else:
        bool_condition = [state[len(state[0])//2][len(state[0][0])//2] != 2 for state in states]
        return all(bool_condition)


#  Write the head of the Table in the Csv file
#  Input: exploration strategy (str), predicate to verify (str) and CSV file(str)
#  Output: None
def firstLineCsvTable(strat, property, csv_filename):
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([strat.upper(), '', property, 'Proportion'])
    return


#  Get agents' states
#  Input: agents (Agent list)
#  Output: list of states (int list list list)
def get_states(agents):
    states = []
    #  Get agents states
    for agent in agents:
        states.append(agent.get_obs())
    return states


#  Boolean function which verifies this predicate: 'The agent has an imperfect cover'
#  Input: agent (Agent)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def imperfect_cover(agent):
    view = agent.get_obs()[0]
    #  Get only the wave range matrix
    max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
    index_range = (agent.view_range - agent.wave_range) // 2
    sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
    #  Another drone in range or a tree in coverage area zone
    return sum([sub_list.count(1) for sub_list in sub_view]) != max_cells_highlighted


#  Boolean function which verifies this predicate: 'Each agent has an imperfect cover'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def imperfect_covers(agents, states=[]):
    # If at least one drone has a perfect cover, the configuration is ignored
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not crash(agent) and perfect_cover(agent):
            return False
    return True


#  Boolean function which verifies this predicate: 'The agent has a perfect cover'
#  Input: agent (Agent)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def perfect_cover(agent):
    view = agent.get_obs()[0]
    #  Get only the wave range matrix
    max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
    index_range = (agent.view_range - agent.wave_range) // 2
    sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
    #  Another drone in range or a tree in coverage area zone
    return sum([sub_list.count(1) for sub_list in sub_view]) == max_cells_highlighted


#  Boolean function which verifies this predicate: 'Each agent has a perfect cover'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def perfect_covers(agents, states=[]):
    # If at least one drone has an imperfect cover, the configuration is ignored
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not crash(agent) and imperfect_cover(agent):
            return False

    return True


#  Boolean function which verifies this predicate: 'The agent is crashed'
#  Input: agent (Agent)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def crash(agent):
    view = agent.get_obs()[0]
    return view[len(view)//2][len(view[0])//2] != 2


#  Boolean function which verifies this predicate: 'Each agent is crashed'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def crashes(agents, states=[]):
    # If at least one drone is not crashed, the configuration is ignored
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not crash(agent):
            return False
    return True


#  Boolean function which verifies this predicate: 'The agent has no drones in its neighborhood'
#  Input: agent (Agent)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def no_drones(agent):
    view = agent.get_obs()[0]
    return sum([sub_list.count(2) for sub_list in view]) == 1


#  Boolean function which verifies this predicate: 'Each agent has no drones in its neighborhood'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def all_no_drones(agents, states=[]):
    # If at least one drone is in another drone's neighborhood max reward, the configuration is ignored
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not crash(agent) and not no_drones(agent):
            return False
    return True


#  Boolean function which verifies this predicate: 'The agent has the maximum reward'
#  Input: agent (Agent)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def max_reward(agent):
    return not crash(agent) and perfect_cover(agent) and no_drones(agent)


#  Boolean function which verifies this predicate: 'Each agent has the maximum reward'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def all_max_reward(agents, states=[]):
    # If at least one drone is not max reward, the configuration is ignored
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not max_reward(agent):
            return False
    return True


#  Boolean function which verifies this predicate: 'The agent is in region idx'
#  Input: agent (Agent) and the region id (int)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def region(agent, idx):
    x, y = agent.get_obs()[1]
    limit = agent.env.nRow//2
    #  Map divided into four parts
    if x < limit and y < limit:
        region = 1
    elif x >= limit > y:
        region = 3
    elif x < limit <= y:
        region = 2
    else:
        region = 4
    return idx == region


#  Boolean function which verifies this predicate: 'In each region, there is only one drone'
#  Input: agents (Agent list) and agents' states (int list list list)
#  Output: a boolean that represents the respect or not of the predicate (bool)
def regions(agents, states=[]):
    regions = []
    for idx, agent in enumerate(agents):
        if states:
            agent.set_obs(states[idx])
        if not crash(agent):
            for r in [1, 2, 3, 4]:
                if region(agent, r):
                    if r in regions:
                        return False
                    else:
                        regions.append(r)

    return True


#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))


#  Verify if the last state from a proposed history respects a predicate
#  Input: agents (Agent list), id of an agent (int) and the predicate to check (str)
#  Output:  a boolean that represents the respect or not of the predicate (bool)
def valid_history(agents, idx_agent, predicate):
    # 'one' or 'all'
    one_drone = predicate[:3] == 'one'
    if one_drone:
        agent = agents[idx_agent - 1]
    property_tmp = predicate[4:]
    # Perfect cover predicate
    if property_tmp == 'perfect cover':
        if one_drone:
            return not crash(agent) and perfect_cover(agent)
        else:
            return perfect_covers(agents)
    # Imperfect cover predicate
    elif property_tmp == 'imperfect cover':
        if one_drone:
            return not crash(agent) and imperfect_cover(agent)
        else:
            return imperfect_covers(agents)
    # Max reward predicate
    elif property_tmp == 'max reward':
        if one_drone:
            return max_reward(agent)
        else:
            return all_max_reward(agents)
    # No drones predicate
    elif property_tmp == 'no drones':
        if one_drone:
            return not crash(agent) and no_drones(agent)
        else:
            return all_no_drones(agents)
    # Crash predicate
    elif property_tmp == 'crash':
        if one_drone:
            return crash(agent)
        else:
            return crashes(agents)
    # Region predicate
    elif property_tmp[:-2] == 'region' or property_tmp == 'regions':
        if one_drone:
            region_id = int(property_tmp[-1])
            return not crash(agent) and region(agent, region_id)
        else:
            return regions(agents)


