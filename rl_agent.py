import random
import numpy as np
from collections import deque
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import scipy.stats as st

class RLAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.model = None
        self.MIN_VAL = 99.5
        self.MAX_VAL = 100.5
        self.D = int(((self.MAX_VAL - self.MIN_VAL) * 10) * 3) + 1  # times 3 for the 3 possible current_position values
        self.Qmus_estimates_mu = np.zeros((3, self.D))
        self.Qmus_estimates_sd = np.ones((3, self.D))

        self.Qsds_estimates_mu = np.ones((3, self.D)) * -2.
        self.Qsds_estimates_sd = np.ones((3, self.D)) * 10.

    def reset_memory(self):
        self.memory = deque(maxlen=20000)

    def remember(self, sar):
        self.memory.append(sar)

    def getMaxQ(self, state):

        if self.model is None:
            return 0.

        # In the Q-table we return the Q value of the best action (including flat, if both long and short are negative),
        # for the current state.
        action_slice = self.Qmus_estimates[:, self.state_value_to_index(state)]
        maxQ = np.max(action_slice, axis=0)

        return maxQ

    def act(self, state, use_explo=True):

        if self.model is None:
            return random.randrange(self.action_size)

        def calculate_VPI(mus, sds):

            def gain(i, i2, x):
                gains = []
                for j in range(len(mus)):
                    if j == i:
                        # special case: this is the best action
                        g = mus[i2] - np.minimum(x, mus[i2])
                    else:
                        g = np.maximum(x, mus[i]) - mus[i]

                    gains.append(g)

                gains = np.reshape(np.array(gains), [-1, len(x)]).transpose()
                return gains

            SAMPLE_SIZE = 1000
            Q_LOW = -1.
            Q_HIGH = 1.
            x = np.random.uniform(Q_LOW, Q_HIGH, SAMPLE_SIZE)
            x = np.reshape(x, [-1, 1])

            dist = st.norm(mus, np.exp(sds))

            probs = dist.pdf(x)

            best_action_idx = np.argmax(mus)

            tmp_mus = np.copy(mus)
            tmp_mus[best_action_idx] = -9999.
            second_best_action_idx = np.argmax(tmp_mus)

            gains = gain(best_action_idx, second_best_action_idx, x)

            return np.mean(gains * probs, axis=0)

        state_idx = self.state_value_to_index(state)
        state_mus = self.Qmus_estimates[:, state_idx]
        state_sds = self.Qsds_estimates[:, state_idx]

        if use_explo:
            VPI_per_action = calculate_VPI(state_mus, state_sds)

            action_scores = VPI_per_action + state_mus

            idx_selected_action = np.argmax(action_scores)

            return idx_selected_action
        else:
            return np.argmax(state_mus)

    def state_value_to_index(self, s):
        # map spread value to an index...
        range = (self.MAX_VAL - self.MIN_VAL) * 10
        idx = (s[0] - self.MIN_VAL) * 10

        # round to hard boundaries, in case we receive state values that are beyond the pre-determined MIN and MAX
        if idx < 0.:
            idx = 0
        elif idx > range:
            idx = range

        # and now adjust to the fact that there are 3 possible positions in the state vector
        pos_idx = s[1] + 1
        return int((pos_idx * range) + idx)

    def replay(self):
        mem = np.array(self.memory)

        states = mem[:, :self.state_size]
        actions = np.reshape(mem[:, self.state_size], [-1, 1])
        rewards = mem[:, -1]

        full_tensor = []
        s_short = []
        r_short = []
        s_long = []
        r_long = []
        for t in range(len(states)):

            idx = self.state_value_to_index(states[t])

            full_tensor.append(np.array([actions[t], idx, rewards[t]]))

            if actions[t] == 0:
                s_short.append(idx)
                r_short.append(rewards[t])
            elif actions[t] == 2:
                s_long.append(idx)
                r_long.append(rewards[t])

        plt.scatter(s_short, r_short, color='red')
        plt.scatter(s_long, r_long, color='green')
        plt.title("State vs Reward scatter plot")
        plt.show()

        # qvalues = [N x 3]
        # 1 - action index
        # 2 - state index
        # 3 - reward
        qvalues = np.array(full_tensor)

        with pm.Model() as self.model:

            def likelihood(Qmus, Qsds):

                def _logp(value):
                    idx0 = tt.cast(value[:, 0], dtype='int8')
                    idx1 = tt.cast(value[:, 1], dtype='int8')
                    return pm.Normal.dist(mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1])).logp(value[:, 2])

                return _logp

            Qmus = pm.Normal('Qmus', mu=self.Qmus_estimates_mu, sd=self.Qmus_estimates_sd, shape=[3, self.D])
            Qsds = pm.Normal('Qsds', mu=self.Qsds_estimates_mu, sd=self.Qsds_estimates_sd, shape=[3, self.D])

            pm.DensityDist('Qtable',
                           likelihood(Qmus, Qsds),
                           observed=qvalues)

            mean_field = pm.fit(n=5000, method='advi', obj_optimizer=pm.adam(learning_rate=0.1))
            self.trace = mean_field.sample(5000)

        self.Qmus_estimates = np.mean(self.trace['Qmus'], axis=0)
        self.Qsds_estimates = np.median(self.trace['Qsds'], axis=0)

        self.Qmus_estimates_mu = self.Qmus_estimates
        self.Qmus_estimates_sd = np.std(self.trace['Qmus'], axis=0)

        self.Qsds_estimates_mu = self.Qmus_estimates
        self.Qsds_estimates_sd = np.std(self.trace['Qsds'], axis=0)

        self.reset_memory()
        fig, axarr = plt.subplots(1, 2)

        axarr[0].plot(self.Qmus_estimates[0], color='red')
        axarr[0].plot(self.Qmus_estimates[2], color='green')
        axarr[0].set_title("E[Q-values] for Short/Long action")

        axarr[1].plot(self.Qsds_estimates[0], color='red')
        axarr[1].plot(self.Qsds_estimates[2], color='green')
        axarr[1].set_title("SD[Q-values] for Short/Long action")

        plt.show()
