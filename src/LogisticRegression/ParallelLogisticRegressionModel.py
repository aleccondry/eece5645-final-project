import numpy as np
import time
import pyspark
from LogisticRegressionModel import LogisticRegressionModel


class ParallelLogisticRegressionModel(LogisticRegressionModel):
    def __init__(self, num_features, learning_rate=1e-7):
        super().__init__(num_features, learning_rate)
        self.weights = np.zeros(num_features + 1)

    def totalLossRDD(self, data: pyspark.RDD, beta: np.array, lam) -> float:
        return data.map(lambda inp: self.logisticLoss(beta, inp[0], inp[1]))\
                   .reduce(lambda x, y: x + y) + lam * np.dot(beta, beta)

    def gradTotalLossRDD(self, data: pyspark.RDD, beta: np.array, lam) -> np.array:
        gradLogLoss = data.map(lambda inp: self.gradLogisticLoss(beta, self.get_features(inp[0]), inp[1]))\
                             .reduce(lambda x, y: x + y)
        regularization = 2 * lam * beta
        return gradLogLoss + regularization

    def lineSearch(self, fun, x, grad, a=0.2, b=0.6):
        t = 1e-5
        fx = fun(x)
        gradNormSq = np.dot(grad, grad)
        while fun(x - t * grad) > fx - a * t * gradNormSq:
            t = b * t
        return t

    def basicMetricsRDD(self, data, beta):
        pairsRDD = data.map(lambda inp: (int(np.sign(np.dot(beta, self.get_features(inp[0])))), int(inp[1])))
        new_pairs = pairsRDD.map(lambda inp: (inp[0], inp[0] * inp[1])).collect()

        TP = 1. * new_pairs.count((1, 1)) + 1
        FP = 1. * new_pairs.count((1, -1)) + 1
        TN = 1. * new_pairs.count((-1, 1)) + 1
        FN = 1. * new_pairs.count((-1, -1)) + 1
        P = TP + FP
        N = TN + FN
        return P, N, TP, FP, TN, FN

    def get_metricsRDD(self, data, beta):
        P, N, TP, FP, TN, FN = self.basicMetricsRDD(data, beta)
        return self.metrics(P, N, TP, FP, TN, FN)

    def trainRDD(self, train_data, test_data, lam=1, eps=1e-3, max_iter=1000, N=20):
        k = 1
        grad_norm = 2 * eps
        start = time.time()
        losses, grad_norms, metrics_t, metrics_v = ([] for _ in range(4))
        while k < max_iter and grad_norm > eps:
            grad = self.gradTotalLossRDD(train_data, self.weights, lam)
            fun = lambda x: self.totalLossRDD(train_data, x, lam)
            gamma = 1e-7
            # gamma = self.lineSearch(fun, self.weights, grad)
            self.weights -= gamma * grad
            obj = fun(self.weights)
            grad_norm = np.sqrt(np.dot(grad, grad))
            acc_t, pre_t, rec_t = self.get_metricsRDD(train_data, self.weights)
            acc_v, pre_v, rec_v = self.get_metricsRDD(test_data, self.weights)
            losses.append(obj)
            grad_norms.append(grad_norm)
            metrics_t.append((acc_t, pre_t, rec_t))
            metrics_v.append((acc_v, pre_v, rec_v))
            print(f'{k}: t={"{:.5f}".format(time.time() - start)}\t'
                  f'L(β_k)={"{:.5f}".format(obj)}\t'
                  f'||∇L(β_k)||_2={"{:.5f}".format(grad_norm)}\t'
                  f'γ={"{:.5E}".format(gamma)}\t'
                  f'train acc={"{:.5}".format(acc_t)}\t'
                  f'test acc={"{:.5}".format(acc_v)}')
            k += 1
        return time.time() - start, k, losses, grad_norms, metrics_t, metrics_v, self.weights

