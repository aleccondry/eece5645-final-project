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

    def trainRDD(self, train_data, test_data, lam=1, eps=1e-3, max_iter=1000):
        k = 1
        grad_norm = 2 * eps
        start = time.time()
        losses, grad_norms, metrics_t, metrics_v = ([] for _ in range(4))
        while k < max_iter and grad_norm > eps:
            # Get gradient of loss and update weights
            grad = self.gradTotalLossRDD(train_data, self.weights, lam)
            self.weights -= self.alpha * grad

            # Calculate GradNorm
            fun = lambda x: self.totalLossRDD(train_data, x, lam)
            obj = fun(self.weights)
            grad_norm = np.sqrt(np.dot(grad, grad))

            # Evaluate training and test metrics
            acc_t, pre_t, rec_t = self.get_metricsRDD(train_data, self.weights)
            acc_v, pre_v, rec_v = self.get_metricsRDD(test_data, self.weights)

            # Store data on current iteration
            losses.append(obj)
            grad_norms.append(grad_norm)
            metrics_t.append((acc_t, pre_t, rec_t))
            metrics_v.append((acc_v, pre_v, rec_v))

            # Print results from current iteration
            print(f'{k}: t={"{:.5f}".format(time.time() - start)}\t'
                  f'L(β_k)={"{:.5f}".format(obj)}\t'
                  f'||∇L(β_k)||_2={"{:.5f}".format(grad_norm)}\t'
                  f'γ={"{:.5E}".format(self.alpha)}\t'
                  f'train acc={"{:.5}".format(acc_t)}\t'
                  f'test acc={"{:.5}".format(acc_v)}')
            k += 1
        return time.time() - start, k, losses, grad_norms, metrics_t, metrics_v, self.weights

