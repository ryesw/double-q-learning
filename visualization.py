from environment import plt, np
from visual_function import plot_q_values_map
from Double_Q_Learning import Q_A, Q_B, env, score_D_Q

plot_q_values_map(Q_A + Q_B, env, 4)

plt.plot(np.arange(len(score_D_Q)), score_D_Q)
plt.xlabel("Episodes (X1000)")
plt.ylabel("Success Rate")
plt.legend(["Double Q-Learning: {:.3f}".format(max(score_D_Q))])
plt.show()