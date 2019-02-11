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
        self.memory = deque(maxlen=500000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.5
        self.model = None
        self.MIN_VAL = 99.5
        self.MAX_VAL = 100.5
        self.D = int(((self.MAX_VAL - self.MIN_VAL) * 100) + 1)

    def remember(self, sar):
        self.memory.append(sar)

    def getMaxQ(self, state):

        if self.model is None:
            return 0.

        # In the Q-table we return the Q value of the best action (including flat, if both long and short are negative),
        # for the current state.
        action_slice = self.Qmus_estimates[:, self.state_value_to_index(state)]
        maxQ = np.max(action_slice, axis=0)

        if maxQ < 0.:
            return 0.       # we can just go flat, too
        else:
            return maxQ

    def act(self, state, use_explo=True):

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

            dist = st.norm(mus, sds)

            probs = dist.pdf(x)

            best_action_idx = np.argmax(mus)

            tmp_mus = np.copy(mus)
            tmp_mus[best_action_idx] = -9999.
            second_best_action_idx = np.argmax(tmp_mus)

            gains = gain(best_action_idx, second_best_action_idx, x)

            return np.mean(gains * probs, axis=0)

        if use_explo:
           if np.random.rand() < self.epsilon or self.model is None:
               return random.randrange(self.action_size)

        state_mus = self.Qmus_estimates[:, self.state_value_to_index(state)]
        state_sds = self.Qsds_estimates[:, self.state_value_to_index(state)]

        VPI_per_action = calculate_VPI(state_mus, state_sds)

        action_scores = VPI_per_action + state_mus

        idx_best_action = np.argmax(action_scores)

        return idx_best_action

    def state_value_to_index(self, s):
        # map spread value to an index...
        idx = (s[0] - self.MIN_VAL) * 100

        if idx < 0.:
            return 0
        elif idx > self.D - 1:
            return self.D - 1
        else:
            return int(idx)

    def replay(self):

        mem = np.array(self.memory)

        states = mem[:, :self.state_size]
        actions = np.reshape(mem[:, self.state_size], [-1, 1])
        rewards = mem[:, -1]

        full_tensor = []
        s_short = []
        r_short = []
        for t in range(len(states)):

            idx = self.state_value_to_index(states[t])

            if actions[t] == 0:
                full_tensor.append(np.array([0, idx, rewards[t]]))
                s_short.append(idx)
                r_short.append(rewards[t])
            elif actions[t] == 2:
                full_tensor.append(np.array([1, idx, rewards[t]]))

        # plt.scatter(s_short, r_short)
        # plt.show()

        # qvalues = [N x 3]
        # 1 - action index (0 - short, 1 - long)
        # 2 - state index
        # 3 - reward
        qvalues = np.array(full_tensor)
        print "full_tensor shape = ", qvalues.shape

        # TODO: re-add current position to state and fix my whole assumption about flat being always 0. It only is
        # when trying to enter... Also re-add the 3-fold effect on states.

        # TODO: why is the first training batch so small compared to the next one?

        # TODO: instead of re-training at every loop on the full dataset, use the previous training iterations, priors and only train incrementally

        with pm.Model() as self.model:

            def likelihood(Qmus, Qsds):

                def _logp(value):
                    idx0 = tt.cast(value[:, 0], dtype='int8')
                    idx1 = tt.cast(value[:, 1], dtype='int8')
                    return pm.Normal.dist(mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1])).logp(value[:, 2])

                return _logp

            # Average training P&L = 4.2
            # Qmus = pm.Normal('Qmus', mu=0., sd=1., shape=[2, self.D])
            # Qsds = pm.Normal('Qsds', mu=0., sd=10., shape=[2, self.D])
            Qmus = pm.Normal('Qmus', mu=0., sd=1., shape=[2, self.D])
            Qsds = pm.Normal('Qsds', mu=-15., sd=10., shape=[2, self.D])

            pm.DensityDist('Qtable',
                           likelihood(Qmus, Qsds),
                           observed=qvalues)

            mean_field = pm.fit(n=10000, method='advi', obj_optimizer=pm.adam(learning_rate=.25))
            self.trace = mean_field.sample(5000)

            #self.trace = pm.sample(1000, tune=1000)

        self.Qmus_estimates = np.mean(self.trace['Qmus'], axis=0)
        self.Qsds_estimates = np.median(self.trace['Qsds'], axis=0)

        # plt.plot(self.Qmus_estimates[0])
        # plt.title("Q-values for Short action")
        #
        # plt.show()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
