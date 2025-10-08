import numpy as np
import gymnasium as gym
from typing import Iterable, Tuple

from interfaces.policy import Policy

def off_policy_mc_prediction_weighted_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *weighted* importance sampling. 

    The algorithm can be found in Sutton & Barto 2nd edition p. 110.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]

    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The discount factor."""
    Q: np.ndarray = initQ
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA))
    """The importance sampling ratios."""

    ## TODO:
    # Implement the off-policy Monte-Carlo prediction algorithm using WEIGHTED importance sampling.
    # Hints:
    #   - Sutton & Barto 2nd edition p. 110
    #   -  Be sure to carefully follow the algorithm.
    #   -  Every-visit implementation is fine.
    #   -  Look at `reversed()` to iterate over a trajectory in reverse order.
    #   -  You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.


    # for each episode
    for ep in trajs:
        G = 0.0
        W = 1.0

        # keep track of visited state-action pairs in this episode for first-visit MC
        visited = []

        # for each step in the episode, iterate backwards
        for (s, a, r, _) in reversed(ep):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.append((s, a))
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a)
            if W == 0:
                break

    return Q

def off_policy_mc_prediction_ordinary_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array,
    gamma:float = 1.0
) -> np.array:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *ordinary* importance sampling. 

    The algorithm with weighted importance sampling can be found in Sutton & Barto 2nd edition p. 110.
    You will need to make a small adjustment for ordinary importance sampling.

    Carefully look at page 109.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]
        
    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The number of actions in the environment."""
    Q: np.ndarray = initQ
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA))
    """The importance sampling ratios."""

    ## TODO:
    # Implement the off-policy Monte-Carlo prediction algorithm using ORDINARY importance sampling.
    # Hints:
    #   - Sutton & Barto 2nd edition p. 110 for the main algorithm.
    #   -  You will need to make a small adjustment for ordinary importance sampling. Carefully look at page 109.
    #        Consider how the C update might be different.
    #   -  Be sure to carefully follow the algorithm.
    #   -  Every-visit implementation is fine.
    #   -  Look at `reversed()` to iterate over a trajectory in reverse order.
    #   -  You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.


    # for each episode
    for ep in trajs:
        G = 0.0
        W = 1.0

        # for each step in the episode, iterate backwards
        for (s, a, r, _) in reversed(ep):
            G = gamma * G + r

            # update for every time we visit this state-action pair, not just the first visit
            N += 1.0
            Q[s, a] += (1/N) * ((W*G) - Q[s, a])
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a)
            if W == 0:
                break

    return Q
