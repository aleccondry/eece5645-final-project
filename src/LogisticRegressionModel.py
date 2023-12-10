import numpy as np
import time


class LogisticRegressionModel:
    def __init__(self, num_features, learning_rate=1e-5):
        self.num_features = num_features
        self.alpha = learning_rate
        self.weights = np.zeros(num_features + 1)

    def get_features(self, x):
        return np.append([1], x)

    def get_weights(self):
        return self.weights

    def logisticLoss(self, beta, x, y) -> float:
        return np.log(1. + np.e ** (-y * np.dot(beta, self.get_features(x))))

    def gradLogisticLoss(self, beta, x, y) -> np.array:
        return -1. * y * x / (1. + np.e ** (y * np.dot(beta, x)))

    def totalLoss(self, x_train, y_train, beta, lam) -> float:
        return sum([self.logisticLoss(beta, x, y) for x, y in zip(x_train, y_train)]) + lam * np.dot(beta, beta)

    def gradTotalLoss(self, x_train, y_train, beta, lam) -> np.array:
        gradLogisticLosses = np.sum(np.array([self.gradLogisticLoss(beta, self.get_features(x), y)
                                              for x, y
                                              in zip(x_train, y_train)]), axis=0)
        regularization = 2 * lam * beta
        return gradLogisticLosses + regularization

    def lineSearch(self, fun, x, grad, a=0.2, b=0.6):
        t = 1e-5
        fx = fun(x)
        gradNormSq = np.dot(grad, grad)
        while fun(x - t * grad) > fx - a * t * gradNormSq:
            t = b * t
        return t

    def basicMetrics(self, x_data, y_data, beta):
        pairs = ((int(np.sign(np.dot(beta, self.get_features(x)))), int(y)) for (x, y) in zip(x_data, y_data))
        new_pairs = [(pred_label, pred_label * true_label) for (pred_label, true_label) in pairs]

        TP = 1. * new_pairs.count((1, 1)) + 1
        FP = 1. * new_pairs.count((1, -1)) + 1
        TN = 1. * new_pairs.count((-1, 1)) + 1
        FN = 1. * new_pairs.count((-1, -1)) + 1
        P = TP + FP
        N = TN + FN
        return P, N, TP, FP, TN, FN

    def metrics(self, P, N, TP, FP, TN, FN):
        acc = (TP + TN) / (P + N)
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        return acc, pre, rec

    def get_metrics(self, x_data, y_data, beta):
        P, N, TP, FP, TN, FN = self.basicMetrics(x_data, y_data, beta)
        return self.metrics(P, N, TP, FP, TN, FN)

    def train(self, x_train, y_train, x_test, y_test, lam=1, eps=1e-3, max_iter=1000):
        k = 1
        grad_norm = 2 * eps
        start = time.time()
        losses, grad_norms, metrics_t, metrics_v = ([] for _ in range(4))
        while k < max_iter and grad_norm > eps:
            grad = self.gradTotalLoss(x_train, y_train, self.weights, lam)
            fun = lambda x: self.totalLoss(x_train, y_train, x, lam)
            gamma = 1e-7
            # gamma = self.lineSearch(fun, self.weights, grad)
            self.weights -= gamma * grad
            obj = fun(self.weights)
            grad_norm = np.sqrt(np.dot(grad, grad))

            # Evaluate training and test metrics
            acc_t, pre_t, rec_t = self.get_metrics(x_train, y_train, self.weights)
            acc_v, pre_v, rec_v = self.get_metrics(x_test, y_test, self.weights)

            # Store data on current iteration
            losses.append(obj)
            grad_norms.append(grad_norm)
            metrics_t.append((acc_t, pre_t, rec_t))
            metrics_v.append((acc_v, pre_v, rec_v))

            # Print results from current iteration
            print(f'{k}: t={"{:.5f}".format(time.time() - start)}\t'
                  f'L(β_k)={"{:.5f}".format(obj)}\t'
                  f'||∇L(β_k)||_2={"{:.5f}".format(grad_norm)}\t'
                  f'γ={"{:.5E}".format(gamma)}\t'
                  f'train acc={"{:.5}".format(acc_t)}\t'
                  f'test acc={"{:.5}".format(acc_v)}')
            k += 1
        return time.time() - start, k, losses, grad_norms, metrics_t, metrics_v, self.weights

