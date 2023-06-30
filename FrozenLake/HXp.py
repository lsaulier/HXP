import random
from copy import deepcopy
import time

import numpy as np
import csv


#  Main function which manages user queries about respect of a predicate in a given history
#  Input: the history (int list), environment (MyFrozenLake), strategy exploration (str) the agent (Agent),
#  number of action to highlight (int), CSV file to store scores (str)
#  Output: None
def HXp(history, env, strat='pi', agent=None, n=2, csv_filename=""):
    answer = False
    good_answers = ["yes", "y"]
    history = list(history.queue)
    print("History : {}".format(history))
    while not answer:

        question = "Do you want an HXp?"
        action_HXp = input(question)

        # Provide an HXp
        if action_HXp in good_answers:

            property_question = "Which predicate do you want to verify? (goal/holes/specific_state/specific_part) \n"
            property = input(property_question)
            # Goal
            if property == 'goal':
                HXp_actions, _, _ = HXp_action(history, env, strat, agent, n=n, csv_filename=csv_filename)

            elif property in ['holes', 'specific_state']:
                avoid_reach_question = "Reach (R) or Avoid (A)? "
                avoid_reach = input(avoid_reach_question)
                # Holes
                if property == 'holes':
                    if avoid_reach in ['R', 'r']:
                        HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, n=n,
                                                    csv_filename=csv_filename)
                    else:
                        HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, property_info=1, n=n,
                                                    csv_filename=csv_filename)
                # Specific state
                else:
                    state_question = "Specify the state : "
                    specific_state = int(input(state_question))
                    if avoid_reach in ['R', 'r']:
                        property = 'avoid specific_state'
                        HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, property_info=specific_state,
                                                    n=n, csv_filename=csv_filename)
                    else:
                        HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, property_info=specific_state,
                                                    n=n, csv_filename=csv_filename)
            # Specific part
            elif property == 'specific_part':
                region_question = "Specify the region (expected format: [a, b, c, d]) : "
                region = [int(s) for s in input(region_question)[1:-1].split(', ')]
                HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, property_info=region, n=n,
                                            csv_filename=csv_filename)
            else:
                print('None property was selected!')
                HXp_actions = 0
            # Render important actions
            if HXp_actions:
                render_actionImportance(HXp_actions, env)

        answer = True

    return


#  Similar to the main function without the user interaction. Different (approximate) HXP are computed
#  Input: the history (int list), environment (MyFrozenLake), strategies to use (str), the agent
#  (Agent), CSV file to store scores (str), predicate to verify (str), additional
#  information for the predicate i.e. a specific state (int) and number of action to highlight (int)
#  Output: None
def HXpSimilarityMetric(history, env, strats, agent, csv_filename='trash.csv', property=None, property_info=None, n=2):
    history = list(history.queue)
    print("History : {}".format(history))
    print("----------------------------")
    scores = []
    times = []
    for strat in strats:
        _, importance_scores, HXp_time = HXp_action(history, env, strat, agent, property, property_info=property_info, n=n
                                 , csv_filename=csv_filename)
        times.append(HXp_time)
        scores.append(importance_scores)

    return scores, times


#  Similar to the main function without the user interaction. One (approximate) HXP is computed
#  Input: the history (int list), environment (MyFrozenLake), the type of strategy (str), the agent
#  (Agent), CSV file to store scores (str), predicate to verify (str), additional
#  information for the predicate i.e. a specific state (int) and number of action to highlight (int)
#  Output: None
def HXpMetric(history, env, strat, agent, csv_filename, property, property_info=None, n=2):

    history = list(history.queue)
    print("History : {}".format(history))
    HXp_actions, _, _ = HXp_action(history, env, strat, agent, property, property_info=property_info, n=n,
                                csv_filename=csv_filename)
    # Render important actions
    print("Rendering of important actions using the pi strategy")
    render_actionImportance(HXp_actions, env)

    return


#  Verify the action importance of each action in the history
#  Input: the history (int list), environment (MyFrozenLake), exploration strategy (str), agent (Agent), wind agents
#  (WindAgent list), predicate to verify (str), additional information for the predicate
#  i.e. a specific state (int), number of action to highlight (int) and CSV file to store scores (str)
#  Output: list of important actions, states and time-steps associated (int list list)
def HXp_action(H, env, strat='exh', agent=None, property='goal', property_info=None, n=2, csv_filename=""):
    HXp_actions = []
    score_list = []
    firstLineCsvTable(strat, property, property_info, csv_filename)  # head of table in CSV file
    fixed_k = ((len(H)-1)//2) - 1
    start_time = time.perf_counter()
    for i in range(0, len(H)-1, 2):
        s, a = H[i], H[i+1]
        actions = get_availableActions(s, env)
        score = actionImportance(s, a, env, actions, fixed_k, strat, agent, property, property_info, csv_filename)
        score_list.append([score, s, a, i // 2])

    # Rank the scores to select the n most important actions
    '''
    print("Actual score list : {}".format(score_list))
    '''
    print('Strategy: {}'.format(strat))
    tmp_list = [elm[0] for elm in score_list]
    round_list = [round(item, 3) for item in tmp_list]
    print("Scores : {}".format(round_list))
    top_idx = np.sort(np.argpartition(np.array(tmp_list), -n)[-n:])
    print("Top {} index(es) : {}".format(n, top_idx))
    for i in range(n):
        HXp_actions.append(score_list[top_idx[i]][1:])
    '''
    print("HXp action list: {}".format(HXp_actions))
    '''
    final_time_s = time.perf_counter() - start_time
    return HXp_actions, round_list, final_time_s


#  Define the importance of an action a from a state s
#  Input: starting state and associated action (int), environment (MyFrozenLake), available actions from s (int list),
#  trajectories length (int), exploration strategy (str), the agent (Agent), predicate to verify (str), additional
#  information for the predicate i.e. a specific state (int), CSV file to store scores (str)
#  Output: importance of a from s (bool)
def actionImportance(s, a, env, actions, k, strat='pi', agent=None, property='goal', property_info=None, csv_filename=""):
    if strat != 'pi':
        deterministic_transition = int(strat.split('_')[1])
    else:
        deterministic_transition = 0
    #  Approximate approach
    if deterministic_transition:
        # The last steps use deterministic transitions
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(s, a, actions, env)
        '''
        print("Reachable states from state {} by doing action {}: {}".format(s, a, Ac_a))
        print("Reachable states from other actions : {}".format(Ac_nota))
        '''
        # Get two sets representing the final states
        Sf_a = succ(Ac_a, k, env, agent, det_transition=deterministic_transition)
        Sf_nota = succ(Ac_nota, k, env, agent, det_transition=deterministic_transition)
        # Count in both sets the number of states that respect the property
        return checkPredicate([s, a], Sf_a, Sf_nota, env, property, property_info, csv_filename)
    #  Exhaustive approaches
    else:
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(s, a, actions, env)
        '''
        print("Reachable states from state {} by doing action {}: {}".format(s, a, Ac_a))
        print("Reachable states from other actions : {}".format(Ac_nota))
        '''
        # Get two sets representing the final states
        Sf_a = succ(Ac_a, k, env, agent) if k else Ac_a
        Sf_nota = succ(Ac_nota, k, env, agent) if k else Ac_nota
        # Count in both sets the number of states that respect the property
        return checkPredicate([s, a], Sf_a, Sf_nota, env, property, property_info, csv_filename)


#  Compute the new state reached by doing action from state and using a deterministic transition
#  Input: environment (MyFrozenLake), state and action (int)
#  Output: the new state (int)
def succ_detTr_module(env, state, action):
    proba = env.slip_probas
    #  Equiprobable *most probable* transitions
    if proba.count(max(proba)) > 1:
        m_proba = max(proba)
        idx_mostpr_transition = random.choice([i for i, elm in enumerate(proba) if elm == m_proba])
    #  One most probable transition
    else:
        idx_mostpr_transition = argmax(proba)
    return env.P[state][action][idx_mostpr_transition][1]


#  Explore trajectories from a set of initial states using only deterministic transitions
#  Input: number of deterministic transition (int), list of current starting states (int list or int list list),
#  environment (MyFrozenLake), agent (Agent)
#  Output: list of final states (int list or int list list)
def succ_detTrLoop_module(det_transition, S, env, agent):
    S_tmp = []
    for i in range(det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    # Extract state, proba
                    state, proba = s
                    if not terminalState(env, state):
                        # Pi strategy
                        action, _ = agent.predict(state)
                        new_state = succ_detTr_module(env, state, action)
                        # Succession of probabilities
                        S_tmp_sublist.append((new_state, proba))
                    else:
                        # Add the terminal state
                        S_tmp_sublist.append((state, proba))

                S_tmp.append(S_tmp_sublist)

            S = S_tmp
            S_tmp = []

        else:
            # Used for generating the list of reachable final states from the current history's action
            for s in S:
                # Extract state, proba
                state, proba = s
                #  Policy strategy
                if not terminalState(env, state):
                    action, _ = agent.predict(state)
                    new_state = succ_detTr_module(env, state, action)
                    # Succession of probabilities
                    S_tmp.append((new_state, proba))
                else:
                    # Add the terminal state
                    S_tmp.append((state, proba))

            S = S_tmp
            S_tmp = []

    return S, S_tmp


#  Explore trajectories from a set of initial states
#  Input: set of initial states (int list), trajectories length (int), environment (MyFrozenLake), the agent (Agent) and
#  number of deterministic transition (int)
#  Output: list of final states, i.e.  the reachable states at horizon k, starting from each state s in S (int list)
def succ(S, k, env, agent=None, det_transition=0):
    S_tmp = []
    if det_transition > k:
        det_transition = k

    #  ####################################### No Det. Tr. ######################################################  #
    # Explore each transition with 'pi'
    for _ in range(k-det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    # Extract state, proba
                    state, proba = s

                    if not terminalState(env, state):
                        # Pi strategy
                        action, _ = agent.predict(state)
                        for p, new_s, _, _ in env.P[state][action]:
                            # Succession of probabilities
                            S_tmp_sublist.append((new_s, proba * p))
                    else:
                        # Add the terminal state
                        S_tmp_sublist.append((state, proba))

                S_tmp.append(S_tmp_sublist)

            S = S_tmp
            S_tmp = []

        # Used for generating the list of reachable final states from the current action
        # Used for generating the list of reachable final states from different actions from current one
        else:
            for s in S:
                # Extract state, proba
                state, proba = s
                if not terminalState(env, state):
                    # Policy strategy
                    if not terminalState(env, state):
                        action, _ = agent.predict(state)
                        for p, new_s, _, _ in env.P[state][action]:
                            # Succession of probabilities
                            S_tmp.append((new_s, proba * p))
                else:
                    # Add the terminal state
                    S_tmp.append((state, proba))

            S = S_tmp
            S_tmp = []

    #  ####################################### Last Det. Tr. ####################################################  #
    #  Perform the last deterministic transitions
    #  These transitions are imposed and deterministic, then their proba is always 1.0, hence we keep unchanged probas
    if det_transition:
        S, S_tmp = succ_detTrLoop_module(det_transition, S, env, agent)

    return S


#  ------ This function is problem dependent -----
#  Compute utilities and compare them to decide the action importance of a
#  Input: state-action list (int list), environment (MyFrozenLake), final states of scenarios starting from s for each
#  action (int list or int list list), predicate to verify (str) additional information for the predicate i.e. a
#  specific state (int), CSV file to store scores (str)
#  Output: action importance (float)
def checkPredicate(s_a_list, Sf, Sf_not, env, property='goal', property_info=None, csv_filename=""):
    # Important action if the probability of reaching the goal is higher when doing action a
    if property == 'goal':
        goal_state = env.to_s(env.goal_position[0], env.goal_position[1])
        Sf_goal = sum([p for (s, p) in Sf if compare(s, goal_state, property)])
        tmp_counter = []
        for sublist in Sf_not:
            tmp_sublist_counter = sum([p for (s, p) in sublist if compare(s, goal_state, property)])
            tmp_counter.append(tmp_sublist_counter)
        # If the HXp version is proba, we don't need to create a probability, we already have it
        best = argmax(tmp_counter)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(tmp_counter) / len(tmp_counter)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_goal, Sf_not_best, Sf_not_average, tmp_counter])
    # Important action if the probability of falling into a hole is higher when doing action a
    elif property == 'holes':
        Holes = get_holes(env)
        #  SF Holes init
        Sf_holes = 0 if not property_info else sum([p for (_, p) in Sf])  # specific case: avoid holes predicate
        for h in Holes:
            tmp_counter = sum([p for (s, p) in Sf if compare(s, h, property)])
            if not property_info:
                Sf_holes += tmp_counter
            # Specific use case were we want the action importance to avoid holes
            else:
                Sf_holes -= tmp_counter

        #  SF not
        tmp_counter = []
        for i in range(len(Sf_not)):
            #  Tmp counter init
            add = 0 if not property_info else sum([p for (_, p) in Sf_not[i]])  # specific case: avoid holes predicate
            tmp_counter.append(add)

            for h in Holes:
                tmp_sublist_counter = sum([p for (s, p) in Sf_not[i] if compare(s, h, property)])

                if not property_info:
                    tmp_counter[i] += tmp_sublist_counter
                # Specific use case were we want the action importance to avoid holes
                else:
                    tmp_counter[i] -= tmp_sublist_counter
        best = argmax(tmp_counter)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(tmp_counter) / len(tmp_counter)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_holes, Sf_not_best, Sf_not_average, tmp_counter])

    # Important action if the probability of reaching/avoiding a specific state is higher when doing action a
    elif property in ['specific_state', 'avoid specific_state']:
        specific_state = property_info
        Sf_spec = sum([p for (s, p) in Sf if compare(s, specific_state, property)])
        tmp_counter = []
        for sublist in Sf_not:
            tmp_sublist_counter = sum([p for (s, p) in sublist if compare(s, specific_state, property)])
            tmp_counter.append(tmp_sublist_counter)
        best = argmax(tmp_counter)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(tmp_counter) / len(tmp_counter)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_spec, Sf_not_best, Sf_not_average, tmp_counter])

    # Important action if the probability of reaching a sub-part of the map is higher when doing action a
    elif property == 'specific_part':
        states_subpart = property_info
        Sf_spec = sum([p for (s, p) in Sf if s in states_subpart])
        tmp_counter = []
        for sublist in Sf_not:
            tmp_sublist_counter = sum([p for (s, p) in sublist if s in states_subpart])
            tmp_counter.append(tmp_sublist_counter)
        best = argmax(tmp_counter)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(tmp_counter) / len(tmp_counter)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_spec, Sf_not_best, Sf_not_average, tmp_counter])
    else:
        return


#  Store all info about an action importance in a CSV file.
#  Input: CSV file to store scores (str), state-action (action importance), list of info to store in the Csv file
#  (int list list)
#  Output: an action importance (float)
def storeInfoCsv(csv_filename, s_a_list, info_list):

    Sf_goal, Sf_not_best, Sf_not_average, tmp_counter = info_list
    # --------------------------- Display info -------------------------------------
    s, a = s_a_list
    '''
    print(
            "By doing action {} from state {}, the predicate is respected in {}% \n"
            "By doing the best contrastive action, it is respected in {}% \n"
            "By not doing action {}, the average respect of the predicate is {}% \n".format(a, s, Sf_goal * 100, Sf_not_best * 100, a, Sf_not_average * 100))
    '''
    #  ------------------------ Store in CSV ----------------------------------------
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        # Action importance of history's action
        writer.writerow(['{}-{}'.format(s, a), 'Action : {}'.format(a), Sf_goal])
        # Action importance of the best different action
        writer.writerow(['', 'Other actions: Best', Sf_not_best])
        # Average action importance of different actions
        writer.writerow(['', 'Other actions: Average', tmp_counter, Sf_not_average])
        writer.writerow('')  # blank line

    return Sf_goal - Sf_not_average


#  Render action HXp
#  Input: list of important actions (int list list), environment (MyFrozenLake)
#  Output: None
def render_actionImportance(HXp_actions, env):
    env_copy = deepcopy(env)
    for s,a,i in HXp_actions:
        print("Timestep {}".format(i))
        env_copy.setObs(s)
        env_copy.render()
        print("    ({})".format(["Left", "Down", "Right", "Up"][a]))
    print("------ End of action importance explanation ------")
    return


#  Compute the set of reachable states from s by doing a and the set of reachable states from s by doing a different
#  action from a.
#  Input: state and associate action according to the history H (int), list of available actions from s (int list), the
#  environment (MyFrozenLake)
#  Output: set of reachable states from s doing a (int list) and set of reachable states from s doing a different
#  action from a (int list)
def get_Ac(s,a, actions, env):
    Ac_a = []
    Ac_nota = []
    Ac_nota_tmp = []
    for action in actions:
        for p, new_s, _, _ in env.P[s][action]:
            if action == a:
                Ac_a.append((new_s, p))
            else:
                Ac_nota_tmp.append((new_s, p))
        if action != a:
            Ac_nota.append(Ac_nota_tmp)
            Ac_nota_tmp = []

    return Ac_a, Ac_nota


#  Get feasible actions from a state s (similar return from each state s in the Frozen Lake problem)
#  Input: state (int) and environment (MyFrozenLake)
#  Output: action list (int list)
def get_availableActions(s, env):
    return list(env.P[s].keys())


#  Extract the list of holes in the map
#  Input: environment (MyFrozenLake)
#  Output: list of holes (int list)
def get_holes(env):
    Holes = []
    for i in range(len(env.desc)):
        for j in range(len(env.desc[0])):
            if bytes(env.desc[i, j]) in b"H":
                Holes.append(env.to_s(i, j))
    return Holes


#  Check if a state is terminal
#  Input: environment (MyFrozenLake) and the state to test (int)
#  Output: the state is a terminal one (bool)
def terminalState(env, state):
    row, col = state // len(env.desc), state % env.nCol
    return bytes(env.desc[row, col]) in b"GH"


#  Write the head of the Table in the Csv file
#  Input: exploration strategy (str), predicate to verify (str), additional information for the predicate i.e. a
#  specific state (int) and CSV file(str)
#  Output: None
def firstLineCsvTable(strat, property, property_info, csv_filename):
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        if property in ['specific_state', 'avoid specific state']:
            writer.writerow([strat.upper(), '', 'state : {}'.format(property_info), 'Proportion'])
        elif property == 'specific_part':
            writer.writerow([strat.upper(), '', 'states : {}'.format(property_info), 'Proportion'])
        else:
            writer.writerow([strat.upper(), '', property, 'Proportion'])
    return


#  Compare two states
#  Input: states (int), predicate (str)
#  Output: result of comparison (bool)
def compare(s0, s1, property):
    if property in ['specific_state', 'goal', 'holes']:
        return s0 == s1
    else:
        return s0 != s1


#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))


#  Verify if the last state from a proposed history respects a predicate
#  Input: state (int), done (bool), reward (float), predicate (str) and additional information (int)
#  Output:  a boolean that represents the respect or not of the predicate (bool)
def valid_history(obs, done, reward, predicate, predicate_info):
    if predicate == 'goal':
        return done and reward
    elif predicate == 'holes':
        if predicate_info is None:
            return done and not reward
        else:
            return not done
    elif predicate == 'specific_state':
        return obs == predicate_info
    elif predicate == 'avoid specific_state':
        return obs != predicate_info
    elif predicate == 'specific_part':
        return obs in predicate_info
    return
