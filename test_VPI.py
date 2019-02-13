import scipy.stats as st
import numpy as np

test_mus = [0.2, 0.8, -0.1]
test_sds = [0.3, 0.0001, 0.5]

def calculate_VPI(mus, sds):

    def gain(i, i2, x):
        print "x shape = ", x.shape
        gains = []
        for j in range(len(mus)):
            if j == i:
                # special case: this is the best action
                g = mus[i2] - np.minimum(x, mus[i2])
                print "g[%s] = %s" % (
                    j,
                    g
                )
            else:
                g = np.maximum(x, mus[i]) - mus[i]
                print "g[%s] = %s" % (
                    j,
                    g
                )

            gains.append(g)

        gains = np.reshape(np.array(gains), [-1, len(x)]).transpose()
        print "gains shape = ", gains.shape

        return gains

    SAMPLE_SIZE = 1000
    Q_LOW = -1.
    Q_HIGH = 1.
    x = np.random.uniform(Q_LOW, Q_HIGH, SAMPLE_SIZE)
    x = np.reshape(x, [-1, 1])

    dist = st.norm(mus, sds)                    # D(s, a)

    probs = dist.pdf(x)                         # P(D(s, a) = x)

    best_action_idx = np.argmax(mus)            # a1

    tmp_mus = np.copy(mus)
    tmp_mus[best_action_idx] = -9999.
    second_best_action_idx = np.argmax(tmp_mus) # a2

    gains = gain(best_action_idx, second_best_action_idx, x)

    return np.mean(gains * probs, axis=0)

vpis = calculate_VPI(test_mus, test_sds)

print "vpis = ", vpis
