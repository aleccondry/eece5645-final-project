import numpy as np
import time


class LogisticRegressionModel:
    """
    A class to perform logistic regression.

    Inputs:
        num_features: int
            The number of features in the given data
        alpha: float
            The learning rate for gradient descent
        weights: np.array
            Beta learned through gradient descent
    """

    def __init__(self, num_features, learning_rate=1e-7):
        self.num_features = num_features
        self.alpha = learning_rate
        self.weights = np.zeros(num_features + 1)

    def get_features(self, x) -> np.array:
        """
        Add a 1 to a the start of a numpy array as a constant bias term

        Inputs:
            x: np.array
                The dataset to add the bias term to

        Returns:
            a new numpy array with 1 appended to the beginning
        """
        return np.append([1], x)

    def logisticLoss(self, beta, x, y) -> float:
        """
        Compute logistic loss for a given beta, set of features, and label using the formula:
            l(β;x,y) = log(1 + exp(-y * <β,x>))

        Inputs:
            beta: np.array
                The weights of the current model
            x: np.array
                The features to compute the logistic loss over
            y: int
                A label for the given features in {-1, +1}

        Returns:
            A float representing the computed logistic loss.
        """
        return np.log(1. + np.e ** (-y * np.dot(beta, self.get_features(x))))

    def gradLogisticLoss(self, beta, x, y) -> np.array:
        """
        Compute gradient of the logistic loss, for a given beta, set of features, and label using the formula:
              ∇l(B;x,y) = -y * x / (1 + exp(y <β,x>))

        Inputs:
            beta: np.array
                The weights of the current model
            x: np.array
                The features to compute the logistic loss over
            y: int
                A label for the given features in {-1, +1}

        Returns:
            A numpy array containing the partial derivatives of the logistic loss for each feature and the bias term.
        """
        return -1. * y * x / (1. + np.e ** (y * np.dot(beta, x)))

    def totalLoss(self, x_train, y_train, beta, lam) -> float:
        """
        Compute the regularized total logistic loss for a given beta, and all features and labels in the data
        using the formula:
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ||β||_2^2

        Inputs:
            beta: np.array
                The weights of the current model
            x: np.array
                All sets of features in the data
            y: int
                All labels in the data
            lam: float
                The regularization factor

        Returns:
            The sum of the logistic loss for all features and labels plus L2 regularization
        """
        return sum([self.logisticLoss(beta, x, y) for x, y in zip(x_train, y_train)]) + lam * np.dot(beta, beta)

    def gradTotalLoss(self, x_train, y_train, beta, lam) -> np.array:
        """
        Compute the gradient of the regularized total logistic loss for a given beta, and all features and labels in
        the data using the formula:
                ∇L(β) = Σ_{(x,y) in data} ∇l(β;x,y) + 2λβ

        Inputs:
            beta: np.array
                The weights of the current model
            x: np.array
                All sets of features in the data
            y: int
                All labels in the data
            lam: float
                The regularization factor

        Returns:
            The gradient of the sum of the logistic loss for all features and labels plus the gradient of L2 regularization
        """
        gradLogisticLosses = np.sum(np.array([self.gradLogisticLoss(beta, self.get_features(x), y)
                                              for x, y
                                              in zip(x_train, y_train)]), axis=0)
        regularization = 2 * lam * beta
        return gradLogisticLosses + regularization

    def basicMetrics(self, x_data, y_data, beta):
        """
        Compute the following quantities:
            Positives: datapoints for which <β,x> > 0
            Negatives: datapoints for which <β,x> <= 0
            True positives:     Number of positives for which the associated label is +1
            False positives:    Number of positives for which the associated label is -1
            True negatives:     Number of negatives for which the associated label is -1
            False negatives:    Number of negatives for which the associated label is +1

        Inputs:
            beta: np.array
                The weights of the current model
            x: np.array
                All sets of features in the data
            y: int
                All labels in the data

        Returns:
            A tuple of the positives, negatives, true positives, false positives, true negatives, false negatives
        """
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
        """
        Compute the accuracy, precision, and recall for a given 5-tuple of (positives, negatives, true positives,
        false positives, true negatives, false negatives)

        Inputs:
            P: int
                Number of positives
            N: int
                NUmber of negatives
            TP: int
                Number of true positives
            FP: int
                Number of false positives
            TN: int
                Number of true negatives
            FN: int
                Number of false negatives

        Returns:
            A tuple containing the accuracy, precision, and recall for the given inputs
        """
        acc = (TP + TN) / (P + N)
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        return acc, pre, rec

    def get_metrics(self, x_data, y_data, beta):
        """
        Compute the accuracy, precision, and recall for a given dataset and weights
        """
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
            self.weights -= self.alpha * grad
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
                  f'γ={"{:.5E}".format(self.alpha)}\t'
                  f'train acc={"{:.5}".format(acc_t)}\t'
                  f'test acc={"{:.5}".format(acc_v)}')
            k += 1
        return time.time() - start, k, losses, grad_norms, metrics_t, metrics_v, self.weights
