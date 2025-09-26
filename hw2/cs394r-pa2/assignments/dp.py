import gymnasium as gym
from typing import Tuple

import numpy as np
from interfaces.policy import Policy

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def value_prediction(
    env: gym.Env, 
    pi: Policy,
    initV: np.array,
    theta: float,
    gamma: float
) -> Tuple[np.array, np.array]:
    """
    Runs the value prediction algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 75, "Value Prediction"
    
    Parameters:
        env (gym.Env): environment with model information, i.e. you know transition dynamics and reward function
        pi (Policy): The policy to evaluate (behavior policy)
        initV (np.ndarray): Initial V(s); numpy array shape of [nS,]
        theta (float): The exit criteria
    Returns:
        tuple: A tuple containing:
            - V (np.ndarray): V_pi function; numpy array shape of [nS]
            - Q (np.ndarray): Q_pi function; numpy array shape of [nS,nA]
    """
    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    # Hint: To get the action probability, use pi.action_prob(state,action)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    #####################

    P = env.P
    """Transition Dynamics;  env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]"""
    states = env.observation_space.n
    """Number of states"""
    actions = env.action_space.n
    """Number of actions"""
    V = initV
    """The V(s) function to estimate"""
    Q = np.zeros((states, actions))
    """The Q(s, a) function to estimate"""

    delta = theta
    while delta >= theta: # iterate until convergence
        delta = 0 # reset delta for this iteration
        V_new = np.copy(V)  # 
        
        for s in range(states):
            old_value = V[s] # save the previous value for convergence check
            V_new[s] = 0 # initialize states to 0 for updates
            
            for a in range(actions):
                action_prob = pi.action_prob(s, a) # get the probability of taking action a in state s under policy pi
                for prob, next_state, reward, _ in P[s][a]:
                    V_new[s] += action_prob * prob * (reward + gamma * V[next_state]) # weighted bellman update for V(s)

            delta = max(delta, abs(old_value - V_new[s])) # check to see if change in state values is below threshold theta

        V = V_new  # update all states

    # find Q
    for s in range(states):
        for a in range(actions):
            Q[s, a] = 0
            for prob, next_state, reward, _ in P[s][a]:
                Q[s, a] += prob * (reward + gamma * V[next_state])

    return V, Q
        
def value_iteration(env: gym.Env, initV: np.ndarray, theta: float, gamma: float) -> Tuple[np.array, Policy]:
    """
    Parameters:
        env (EnvWithModel): environment with model information, i.e. you know transition dynamics and reward function
        initV (np.ndarray): initial V(s); numpy array shape of [nS,]
        theta (float): exit criteria

    Returns:
        tuple: A tuple containing:
            - value (np.ndarray): optimal value function; shape of [nS]
            - policy (GreedyQPolicy): optimal deterministic policy
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    # Hint: Try updating the Q function in the `pi` policy object
    #####################
    nS: int = env.observation_space.n
    """Number of states"""
    nA: int = env.action_space.n
    """Number of actions"""
    V: np.ndarray = initV
    """Initial V values"""
    Q: np.ndarray = np.zeros((nS, nA))
    """Initial Q values"""
    pi: Policy_DeterministicGreedy = Policy_DeterministicGreedy(Q)
    """Initial policy, you will need to update this policy after each iteration"""
    P: np.ndarray = env.P
    """Transition Dynamics;  env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]"""

    delta = theta
    while delta >= theta: # iterate until convergence
        delta = 0 # reset delta for this iteration
        for s in range(nS): 
            v = V[s] # save the previous value for convergence check
            for a in range(nA):
                Q[s, a] = 0 # initialize Q(s,a) to 0 for updates
                for prob, next_state, reward, _ in P[s][a]:
                    Q[s, a] += prob * (reward + gamma * V[next_state]) # weighted bellman update
            V[s] = max(Q[s, :]) # take the max over all actions
            delta = max(delta, abs(v - V[s])) # check to see if change in state values is below threshold theta
        


    return V, Policy_DeterministicGreedy(Q)
