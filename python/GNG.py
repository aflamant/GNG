#! /usr/bin/env python3

import numpy as np
import scipy.spatial


class GNG:

    def __init__(self, n_nodes=50, max_iter=20, lambda_=50, epsilon_b=0.2, epsilon_n=0.005, alpha=0.5,
                 delta=0.995, max_age=20):

        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.alpha = alpha
        self.delta = delta
        self.max_age = max_age

    def fit(self, data):

        data = np.random.permutation(data)

        n_data = data.shape[0]
        n_dim = data.shape[1]

        data_min = data.min(axis=0)
        data_max = data.max(axis=0)

        # Number of nodes, initially 2
        ni = 2

        w = np.zeros((ni, n_dim))

        # Generate two random nodes
        for i in range(ni):
            w[i, :] = np.random.uniform(data_min, data_max)

        # Initialize the error vector
        e = np.zeros(ni)

        # Initialize edges matrix and the age matrix
        c = np.zeros((ni, ni))
        t = np.zeros((ni, ni))

        # MAIN LOOP

        # number of the iteration
        nx = 1

        for it in range(self.max_iter):
            for l in range(n_data):
                nx += 1

                x = data[l, :].reshape(1, -1)

                # Competition and ranking
                d = scipy.spatial.distance.cdist(x, w)
                sort_order = np.argsort(d)
                s1 = sort_order[0, 0]
                s2 = sort_order[0, 1]

                # Aging
                t[s1, :] += 1
                t[:, s1] += 1

                # Adding error
                e[s1] += d[0, s1] ** 2

                # Adaptation
                w[s1] += (self.epsilon_b * (x - w[s1, :])).reshape(2)

                n_s1 = np.where(c[s1, :] == 1)
                for j in n_s1:
                    w[j, :] += self.epsilon_n * (x - w[j, :])

                # Creating the links
                c[s1, s2] = 1
                c[s2, s1] = 1
                t[s1, s2] = 0
                t[s2, s1] = 0

                # Removing links that are too old
                c[t > self.max_age] = 0
                n_neighbors = np.sum(c, axis=0)
                # Removing lonesome nodes
                alone_nodes = np.where(n_neighbors == 0)
                np.delete(c, alone_nodes, 0)
                np.delete(c, alone_nodes, 1)
                np.delete(t, alone_nodes, 0)
                np.delete(t, alone_nodes, 1)
                np.delete(w, alone_nodes, 0)
                np.delete(e, alone_nodes)

                # If it's an iteration with node adding and we're not at maximum, add nodes
                if nx % self.lambda_ == 0 and w.shape[0] < self.n_nodes:
                    q = e.argmax()
                    f = np.multiply(c[:, q], e).argmax()

                    new_node = ((w[q, :] + w[f, :]) / 2).reshape(1,-1)
                    w = np.append(w, new_node, axis=0)

                    # Increase the size of c, t and e with values at zero
                    r = w.shape[0] - 1
                    new_c = np.zeros((r + 1, r + 1))
                    new_c[:-1, :-1] = c
                    c = new_c
                    new_t = np.zeros((r + 1, r + 1))
                    new_t[:-1, :-1] = t
                    t = new_t
                    e = np.append(e, 0.0)

                    c[q, f] = 0
                    c[f, q] = 0
                    c[q, r] = 1
                    c[r, q] = 1
                    c[r, f] = 1
                    c[f, r] = 1

                    e[q] = self.alpha * e[q]
                    e[f] = self.alpha * e[f]
                    e[r] = e[q]

                # Decreasing error
                e = self.delta * e

        export = {'nodes': w, 'errors': e, 'edges': c, 'ages': t}

        return export
