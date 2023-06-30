
from copy import deepcopy
import time

import numpy as np
import csv


#  Main function which manages user queries about action importance
#  Input: the history (int list list list), environment (Connect4), strategy exploration (str), player 1 (Agent),
#  player 2 (Agent), CSV file to store scores (str), the policy (DQN) and number of action to highlight (int)
#  Output: None
def HXp(history, env, strat='pi', player1=None, player2=None, csv_filename="", net=None, n=2):
    answer = False
    good_answers = ["yes", "y"]
    history = list(history.queue)
    print(history)
    while not answer:

        question = "Do you want an HXp?"
        action_HXp = input(question)
        # Provide an action HXp
        if action_HXp in good_answers:
            player2.random = False
            property_question = "Which predicate do you want to test? (win/lose/control_midcolumn/3inarow/counter_3inarow) \n"
            property = input(property_question)
            HXp_actions, _, _ = HXp_action(history, env, strat, player1, player2, property, csv_filename=csv_filename, net=net, n=n)
            # Render important actions
            if HXp_actions:
                render_actionImportance(HXp_actions, env)

            player2.random = True

        answer = True

    return


#  Similar to the main function without the user interaction. HXp is computed for each strategy
#  Input: the history (int list), environment (Connect4), strategies (str list), player 1 and player 2 (Agent), CSV file
#  to store scores (str), predicate to verify (str), the policy (DQN) and number of action to highlight (int)
#  Output: importance scores (float list)
def HXpSimilarityMetric(history, env, strats, player1, player2, csv_filename='trash.csv', property=None, net=None, n=2):
    history = list(history.queue)
    player2.random = False
    scores = []
    times = []
    for strat in strats:
        env_copy = deepcopy(env)
        _, importance_scores, HXp_time = HXp_action(history, env_copy, strat, player1, player2, property, csv_filename=csv_filename, net=net, n=n)

        times.append(HXp_time)
        scores.append(importance_scores)

    return scores, times


#  Similar to the main function without the user interaction
#  Input: the history (int list list list), environment (Connect4), strategy (str), player 1 and 2 (Agent), CSV file to
#  store scores (str), predicate to verify (str), the policy (DQN) and number of action to highlight (int)
#  Output: importance scores (float list)
def HXpMetric(history, env, strat, player1, player2, csv_filename, property, net=None, n=2):

    history = list(history.queue)
    player2.random = False  # useful for approximate HXp
    HXp_actions, scores, _ = HXp_action(history, env, strat, player1, player2, property, csv_filename=csv_filename, net=net, n=n)
    # Render important actions
    print("Rendering of important actions using the pi strategy")
    render_actionImportance(HXp_actions, env)
    return scores


#  Compute the action importance of each action in the history
#  Input: the history (int list), environment (Connect4), exploration strategy (str), agents (Agent), predicate to
#  verify (str), CSV file to store scores (str), the policy (DQN) and number of action to highlight (int)
#  Output: list of important actions, states and time-steps associated (int list list) and action importance scores
#  (float list list)
def HXp_action(H, env, strat='pi', player1=None, player2=None, property='win', csv_filename="", net=None, n=2):
    HXp_actions = []
    score_list = []
    firstLineCsvTable(strat, property, csv_filename)  # head of table in CSV file
    fixed_k = ((len(H)-1)//2) - 1
    env_copy = deepcopy(env)
    start_time = time.perf_counter()
    for i in range(0, len(H)-1, 2):
        s, a = H[i], H[i+1]
        env_copy.board = s
        actions = get_availableActions(s, env_copy)
        score = actionImportance(s, a, env_copy, actions, fixed_k, strat, player1, player2, property, csv_filename, net)
        score_list.append([score, s, a, i // 2])

    # Rank the scores to select the n most important actions
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
#  Input: starting state (int list list), associated action (int), environment (Connect4), available actions from s
#  (int list), trajectories length (int), exploration strategy (str), agents (Agent), predicate to verify (str), CSV
#  file to store scores (str) and the policy (DQN)
#  Output: importance of 'a' from 's' (float)
def actionImportance(s, a, env, actions, k, strat='pi', player1=None, player2=None, property='win', csv_filename="",
                     net=None):
    if strat != 'pi':
        deterministic_transition = int(strat.split('_')[1])
    else:
        deterministic_transition = 0
    #  Approximate approach
    if deterministic_transition:
        #  The last steps use deterministic transitions
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(s, a, actions, player1, player2, env)
        # Get two sets representing the final states
        Sf_a = succ(Ac_a, k, env, player1, player2, det_transition=deterministic_transition, net=net)
        Sf_nota = succ(Ac_nota, k, env, player1, player2, det_transition=deterministic_transition, net=net)
        # Count in both sets the number of states which respects the property
        return checkPredicate([s, a], Sf_a, Sf_nota, env, player1, property, csv_filename)
    #  Exhaustive approach
    else:
        # Get two initial sets
        Ac_a, Ac_nota = get_Ac(s, a, actions, player1, player2, env)
        # Get two sets representing the final states
        Sf_a = succ(Ac_a, k, env, player1, player2, net=net) if k else Ac_a
        Sf_nota = succ(Ac_nota, k, env, player1, player2, net=net) if k else Ac_nota
        # Count in both sets the number of states which respects the property
        return checkPredicate([s, a], Sf_a, Sf_nota, env, player1, property, csv_filename)


#  Compute the new state reached by doing action from state and using a deterministic transition
#  Input: agents (Agent), action (int), environment (Connect4), state (int list list) and policy (DQN)
#  Output: the new state (int list list)
def succ_detTr_module(agents, action, env, state, net=None):
    #  Probable deterministic transition
    #  Use the agent's knowledge to determine the best transition
    env.board = state
    new_state = env.step(agents, action, net=net, det_transition=True)
    return new_state


#  Explore trajectories from a set of initial states using only deterministic transitions
#  Input: number of deterministic transition (int), list of current starting states (int list list or int list list
#  list), environment (Connect4), agents (Agent), and the policy (DQN)
#  Output: list of final states (int list list or int list list list)
def succ_detTrLoop_module(det_transition, S, env, player1, player2, net):
    S_tmp = []
    for i in range(det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action/transition, different from the current action/transition
        # This part used when deterministic transition > 1, and we compute successors based on all not a actions
        # different from pi(s). It could also be used if we first explore search space with classic 'pi' and
        # then want deterministic transitions (but updates might be necessary)
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    # Deals with different type of HXp
                    state, proba = s
                    if not terminalState(state, env):
                        action = player1.choose_action(state, net=net)
                        new_state = transition(state, action, player1, player2, env, net=net)
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
                # Deals with different type of HXp
                state, proba = s
                #  Exhaustive and Policy strategy
                if not terminalState(state, env):
                    action = player1.choose_action(state, net=net)
                    new_state = transition(state, action, player1, player2, env, net=net)
                    # Succession of probabilities
                    S_tmp.append((new_state, proba))
                else:
                    # Add the terminal state
                    S_tmp.append((state, proba))
            S = S_tmp
            S_tmp = []

    return S, S_tmp


#  Explore trajectories from a set of initial states
#  Input: set of initial states (int list), trajectories length (int), environment (Connect4),  agents (Agent), number
#  of deterministic transitions (int) and the policy (DQN)
#  Output: list of final states, i.e.  the reachable states at horizon 'k', starting from each state 's' in S (int list)
def succ(S, k, env, player1=None, player2=None, det_transition=0, net=None):
    S_tmp = []
    if det_transition > k:
        det_transition = k
    #  ####################################### No Det. Tr. ######################################################  #
    # Explore each transition
    for i in range(k-det_transition):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    # Deals with different type of HXp
                    state, proba = s
                    if not terminalState(state, env):
                        action = player1.choose_action(state, net=net)
                        for new_s, p in transition(state, action, player1, player2, env):
                            # Succession of probabilities
                            S_tmp_sublist.append((new_s, proba * p))
                    else:
                        # Add the terminal state
                        S_tmp_sublist.append((state, proba))
                S_tmp.append(S_tmp_sublist)
            S = S_tmp
            S_tmp = []

        # Used for generating the list of reachable final states from the current action
        else:
            for s in S:
                # Deals with different type of HXp
                state, proba = s
                if not terminalState(state, env):
                    action = player1.choose_action(state, net=net)
                    for new_s, p in transition(state, action, player1, player2, env):
                        # Succession of probabilities
                        S_tmp.append((new_s, proba * p))
                else:
                    # Add the terminal state
                    S_tmp.append((state, proba))
            S = S_tmp
            S_tmp = []

    #  ####################################### Last Det. Tr. ####################################################  #
    #  Perform the last deterministic transitions
    #  These transitions are imposed and deterministic, then their proba is always 1.0, hence we keep unchanged proba
    if det_transition:
        S, S_tmp = succ_detTrLoop_module(det_transition, S, env, player1, player2, net)

    return S


#  Compute scores and compare them to decide the action importance of 'a'
#  Input: state-action list (int list), lists of final states (int list list and int list list list), environment 
#  (Connect4), agent (Agent), predicate to verify (str) and CSV file to store scores (str)
#  Output: action importance (float)
def checkPredicate(s_a_list, Sf, Sf_not, env, agent, property='win', csv_filename=""):
    # Important action if the probability of reaching a winning state is higher when doing action a
    if property == 'win':
        # Path probabilities to reach a winning state by doing a from s
        cpt = 0
        Sf_win = 0.0
        for (s, p) in Sf:
            env.board = s
            Sf_win += p if env.win(agent.token) else 0.0
            cpt += 1 if env.win(agent.token) else 0
        # Path probabilities to reach a winning state by not doing a from s
        proba = []

        for sublist in Sf_not:
            cpt = 0
            tmp_sublist_counter = 0.0
            for (s, p) in sublist:
                env.board = s
                tmp_sublist_counter += p if env.win(agent.token) else 0.0
                cpt += 1 if env.win(agent.token) else 0
            proba.append(tmp_sublist_counter)
        # Extract probability associated to the best 'not a' action
        # Avoid one specific case, no other actions are available
        if not proba:
            proba = [0.0]
        Sf_not_best = max(proba)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(proba) / len(proba)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_win, Sf_not_best, Sf_not_average, proba])
    # Important action if the probability of reaching a losing state is higher when doing action a
    elif property == 'lose':
        # Path probabilities to reach a losing state by doing a from s
        Sf_lose = 0.0
        for (s, p) in Sf:
            env.board = s
            Sf_lose += p if env.win(agent.token*(-1)) else 0.0
        # Path probabilities to reach a losing state by not doing a from s
        proba = []

        for sublist in Sf_not:
            tmp_sublist_counter = 0.0
            for (s, p) in sublist:
                env.board = s
                tmp_sublist_counter += p if env.win(agent.token*(-1)) else 0.0
            proba.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        # Avoid one specific case, no other actions are available
        if not proba:
            proba = [0.0]
        Sf_not_best = max(proba)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(proba) / len(proba)

        return storeInfoCsv(csv_filename, s_a_list, [Sf_lose, Sf_not_best, Sf_not_average, proba])

    elif property in ['3inarow', 'counter_3inarow']:
        # Count number of '3 in a row' from the initial board
        s, _ = s_a_list
        agent_token = agent.token if property == '3inarow' else agent.token * (-1)
        s_nb = count3inarow(s, agent_token)
        # Path probabilities to reach a state by doing a from s
        cpt = 0
        Sf_3inarow = 0.0
        for (s, p) in Sf:
            condition = count3inarow(s, agent_token) > s_nb if property == '3inarow' else count3inarow(s, agent_token) == s_nb
            if condition:
                Sf_3inarow += p
                cpt += 1
        # Path probabilities to reach a winning state by not doing a from s
        proba = []

        for sublist in Sf_not:
            cpt = 0
            tmp_sublist_counter = 0.0
            for (s, p) in sublist:
                env.board = s
                condition = count3inarow(s, agent_token) > s_nb if property == '3inarow' else count3inarow(s, agent_token) == s_nb
                if condition:
                    tmp_sublist_counter += p
                    cpt += 1
            proba.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        # Avoid one specific case, no other actions are available
        if not proba:
            proba = [0.0]
        Sf_not_best = max(proba)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(proba) / len(proba)
        return storeInfoCsv(csv_filename, s_a_list, [Sf_3inarow, Sf_not_best, Sf_not_average, proba])

    elif property == 'control_midcolumn':
        # Count number of tokens in the mid-column of the initial board
        s, _ = s_a_list
        # To control mid-column, we need more yellow tokens than red ones
        s_nb = [row[len(s)//2] for row in s].count(agent.token) - [row[len(s)//2] for row in s].count(agent.token * (-1))
        # Path probabilities to reach a state by doing a from s
        cpt = 0
        Sf_midcol = 0.0
        for (s, p) in Sf:
            if [row[len(s)//2] for row in s].count(agent.token) - [row[len(s)//2] for row in s].count(agent.token * (-1)) > s_nb:
                Sf_midcol += p
                cpt += 1
        # Path probabilities to reach a winning state by not doing a from s
        proba = []

        for sublist in Sf_not:
            cpt = 0
            tmp_sublist_counter = 0.0
            for (s, p) in sublist:
                if [row[len(s)//2] for row in s].count(agent.token) - [row[len(s)//2] for row in s].count(agent.token * (-1)) > s_nb:
                    tmp_sublist_counter += p
                    cpt += 1
            proba.append(tmp_sublist_counter)

        # Extract probability associated to the best 'not a' action
        # Avoid one specific case, no other actions are available
        if not proba:
            proba = [0.0]
        Sf_not_best = max(proba)
        # Extract the average probability associated to all 'not a' actions
        Sf_not_average = sum(proba) / len(proba)
        return storeInfoCsv(csv_filename, s_a_list, [Sf_midcol, Sf_not_best, Sf_not_average, proba])

    else:
        return


#  Store data about an action importance in a CSV file.
#  Input: CSV file to store scores (str), state-action (int list list), list of data to store (int list list)
#  Output: an action importance score (float)
def storeInfoCsv(csv_filename, s_a_list, info_list):

    Sf_goal, Sf_not_best, Sf_not_average, tmp_counter = info_list

    # --------------------------- Display info -------------------------------------
    s, a = s_a_list
    '''
    print(
        "By doing action {} from state {}, the predicate is respected in {}% \n"
        "By doing the best contrastive action, it is respected in {}% \n"
        "By not doing the action {}, the average respect of the predicate is {}% \n".format(a, s, Sf_goal * 100, Sf_not_best * 100, a, Sf_not_average * 100))
    '''
    #  ------------------------ Store in CSV ----------------------------------------
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        # Action importance of history's action
        row = ['{}-{}'.format(s, a), 'Action : {}'.format(a), Sf_goal]
        writer.writerow(row)
        # Action importance of the best different action
        row = ['', 'Other actions: Best', Sf_not_best]
        writer.writerow(row)
        # Average action importance of different actions
        writer.writerow(['', 'Other actions: Average', tmp_counter, Sf_not_average])

        writer.writerow('')  # blank line

    return Sf_goal - Sf_not_average


#  Render action HXp
#  Input: list of state-action-timestep (tuple list list), environment (Connect4)
#  Output: None
def render_actionImportance(HXp_actions, env):
    env_copy = deepcopy(env)
    for s, a, i in HXp_actions:
        print("Timestep {} --- Action {}".format(i, a))
        env_copy.board = s
        env_copy.render()
    print("------ End of action importance explanation ------")
    return


#  Compute the set of reachable states from 's' by doing 'a' and the set of reachable states from 's' by doing 'a''
#  different action from 'a'.
#  Input: state (int list list) and associate action (int), list of available actions from 's' (int list), agents
#  (Agent), the environment (Connect4)
#  Output: set of reachable states from s doing a (int list) and set of reachable states from s doing a different
#  action from a (int list list) with their associated proba
def get_Ac(s, a, actions, player1, player2, env):
    Ac_a = []
    Ac_nota = []
    Ac_nota_tmp = []
    for action in actions:
        for new_s, p in transition(deepcopy(s), action, player1, player2, env):
            if action == a:
                Ac_a.append((new_s, p))
            else:
                Ac_nota_tmp.append((new_s, p))
        if action != a:
            Ac_nota.append(Ac_nota_tmp)
            Ac_nota_tmp = []

    return Ac_a, Ac_nota


#  Get transition(s) from agent's state and action
#  Input: state (int list list), action (int), players (Agent), environment (Connect4), policy (DQN)
#  Output: list of new state(s) 's'' and associated proba (tuple list)
def transition(s, a, player1, player2, env, net=None):
    # Temporary state
    s = env.update_state(s, player1.token, a)

    #  Avoid useless player2 transitions when player1 already won
    if env.win(player1.token, copy_board=s):
        if net is not None:
            return s
        else:
            return [(s, 1.0)]
    #  Deterministic transition
    if net is not None:
        p2_action = player2.choose_action(env.inverse_board(s), net)
        s = env.update_state(s, player2.token, p2_action)
        return s

    # Each new state with the associated probability (1/available_actions)
    transitions = get_availableActions(s, env)
    new_s_proba = []
    for tr in transitions:
        new_s = deepcopy(s)
        new_s = env.update_state(new_s, player2.token, tr)
        new_s_proba.append((new_s, 1/len(transitions)))
    return new_s_proba


#  Get from a state the set of legal actions
#  Input: a board (int list list) and the environment (Connect4)
#  Output: list of legal actions (int list)
def get_availableActions(state, env):
    return [i for i in range(env.cols) if state[0][i] == 0]


#  Write the head of the Table in the Csv file
#  Input: exploration strategy (str), predicate to verify (str) and CSV file (str)
#  Output: None
def firstLineCsvTable(strat, property, csv_filename):
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([strat.upper(), '', property, 'Proportion'])

    return


#  Check if a state is terminal or not
#  Input: a board (int list list) and the environment (Connect4)
#  Output: done (bool)
def terminalState(state, env):
    env.board = state
    # win/lose/draw conditions
    return env.win(1) or env.win(-1) or not sum(line.count(0) for line in env.board)


#  Count number of 3 in a row from a board for 1 player
#  Input: board (int list list) and token of the agent (int)
#  Reward: number of 3 in a row (int)
def count3inarow(board, token):
    cols = len(board[0])
    rows = len(board)
    cpt = 0
    # Check horizontal locations for win
    for c in range(cols - 2):
        for r in range(rows):
            if board[r][c] == token and board[r][c + 1] == token and board[r][c + 2] == token:
                cpt += 1

    # Check vertical locations for win
    for c in range(cols):
        for r in range(rows - 2):
            if board[r][c] == token and board[r + 1][c] == token and board[r + 2][c] == token:
                cpt += 1

    # Check positively sloped diagonals
    for c in range(cols - 2):
        for r in range(rows - 2):
            if board[r][c] == token and board[r + 1][c + 1] == token and board[r + 2][c + 2] == token:
                cpt += 1

    # Check negatively sloped diagonals
    for c in range(cols - 2):
        for r in range(3, rows):
            if board[r][c] == token and board[r - 1][c + 1] == token and board[r - 2][c + 2] == token:
                cpt += 1

    return cpt


#  Verify if the last state from a proposed history respects a predicate
#  Input: agent (Agent), environment (Connect4), last state (int list), terminal or not (bool), associated reward
#  (float), first state (int list) and the predicate to check (str)
#  Output:  a boolean that represents the respect or not of the predicate (bool)
def valid_history(agent, env, state, done, reward, first_state, predicate):
    if predicate == 'win':
        return done and reward == env.rewards['win']
    elif predicate == 'lose':
        return done and reward == env.rewards['lose']
    elif predicate == 'control_midcolumn':
        fs_nb = [row[len(first_state) // 2] for row in first_state].count(agent.token) - [row[len(first_state) // 2] for row in first_state].count(agent.token * (-1))
        s_nb = [row[len(state) // 2] for row in state].count(agent.token) - [row[len(state) // 2] for row in state].count(agent.token * (-1))
        return s_nb > fs_nb
    elif predicate in ['3inarow', 'counter_3inarow']:
        agent_token = agent.token if predicate == '3inarow' else agent.token * (-1)
        fs_nb = count3inarow(first_state, agent_token)
        s_nb = count3inarow(state, agent_token)
        if predicate == '3inarow':
            return s_nb > fs_nb
        else:
            return s_nb == fs_nb
    return



