from environment import env, np

# Exploitation
def greedy_action(state, Q):

  action=np.argmax(Q[state, :])

  return action

# Exploration
def e_greedy_action(state, Q, epsilon):
  
  if np.random.rand() < epsilon:
      action = env.action_space.sample()
  else:
      action = np.argmax(Q[state, :])

  return action