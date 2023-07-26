from environment import np, env, n_episodes, gamma
from action_function import e_greedy_action

# Double Q-Learning
Q_A =np.zeros([env.observation_space.n, env.action_space.n])
Q_B =np.zeros([env.observation_space.n, env.action_space.n])

Reward_list = []
score_D_Q=list()

current_epsilon=1.0
k = 1.0
lr = 1e-2

for i in range(n_episodes):

    state = env.reset()
    done = False
    G=0

    # Implement HERE ----------------------------------------------------------
    while not done:
      current_epsilon = 1.0 / (k * 0.01)
      Q = (Q_A + Q_B) / 2
      action = e_greedy_action(state, Q, current_epsilon)
      new_state, reward, done, _ = env.step(action)
      
      if np.random.rand() < 0.5:
        # UPDATE(A)
        td_target = reward + gamma * Q_B[new_state, :].max() * (1 - done)
        Q_A[state, action] += lr * (td_target - Q_A[state, action])
      else:
        # UPDATE(B)
        td_target = reward + gamma * Q_A[new_state, :].max() * (1 - done)
        Q_B[state, action] += lr * (td_target - Q_B[state, action])

      state = new_state
      G += reward

    # -------------- Do not modify anything outside ---------------------------

    k = k + 1.0
    Reward_list.append(G)


    if (i+1) % 1000 == 0:
      print("Current score: {:.3f}".format(sum(Reward_list) / i))
      score_D_Q.append(sum(Reward_list)/i)


env.close()