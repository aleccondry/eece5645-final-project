import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from ParallelLogisticRegressionModel import ParallelLogisticRegressionModel
import random
from pyspark import SparkContext


def write_beta(output, beta_to_save, feature_names):
    with open(output, 'w') as f:
        for val, label in zip(beta_to_save, feature_names):
            f.write('(%s,%f)\n' % (label, val))


def read_features_pd(input_file: str) -> pd.DataFrame:
    data = pd.read_csv(input_file)
    data.drop(columns=['Diabetes_binary'], inplace=True)
    return data


def read_features_np(input_file: str) -> np.array:
    return np.array(read_features_pd(input_file))


def read_labels_pd(input_file: str) -> pd.DataFrame:
    return pd.read_csv(input_file)['Diabetes_binary']


def read_labels_np(input_file: str) -> np.array:
    return np.array(read_labels_pd(input_file))


def get_all_features(data: pd.DataFrame) -> list:
    return list(data.columns)


def random_oversample(features, labels):
    pos_feats = [x for i, x in enumerate(features) if labels[i] == 1]
    neg_feats = [x for i, x in enumerate(features) if labels[i] == 0]
    num_pos = len(pos_feats)
    num_neg = len(neg_feats)
    num_less = abs(num_pos - num_neg)
    choice_arr = pos_feats if num_pos < num_neg else neg_feats
    choice_label = 1 if num_pos < num_neg else 0
    for i in range(num_less):
        features.append(random.choice(choice_arr))
        labels.append(choice_label)
    return np.array(features), np.array(labels)


# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def parallelize_data(spark_context, features, labels):
    result = []
    for feature, label in zip(features, labels):
        result.append((feature, label))
    return spark_context.parallelize(result)


def create_data_plots(k, data, label=""):
    plt.plot([i for i in range(1, k)], data, label=label)
    plt.xlabel("Number of Iterations")
    plt.ylabel(label)
    plt.title(f'{label} vs Number of Iterations')
    plt.savefig(f'../results_logreg/{label}_fig_parallelized.png')


def create_metric_plots(k, metrics, label=""):
    iters = [x for x in range(1, k)]
    accs = [x[0] for x in metrics]
    pres = [x[1] for x in metrics]
    recs = [x[2] for x in metrics]
    plt.plot(iters, accs, label="Accuracy")
    plt.plot(iters, pres, label="Precision")
    plt.plot(iters, recs, label="Recall")
    plt.legend(loc='upper right')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Metric Value")
    plt.title(f'{label} Metrics vs Number of Iterations')
    plt.savefig(f'../results_logreg/metrics_fig_parallelized.png')


if __name__ == "__main__":
    # Set up argument parser and associated flags
    parser = argparse.ArgumentParser(description="Logistic Regression.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/Clean_data.csv', help='Input file containing all features and labels, used to train a logistic model')
    parser.add_argument('--beta', default='../results_logreg/beta', help='File where beta is stored')
    parser.add_argument('--split', type=float, default=0.8, help='Test/Training split. Percentage of data to be used for training')
    parser.add_argument('--lam', type=float, default=0.0, help="Regularization parameter λ")
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    parser.add_argument('--N', type=int, default=20, help='Level of parallelism/number of partitions')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true', help="Print Spark warning/info messages.")
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false', help="Suppress Spark warning/info messages.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Sparse Logistic Regression')

    if not args.verbose:
        sc.setLogLevel("ERROR")

    data_path = args.data
    if data_path is None:
        print("Please provide a valid data path using the --data flag")
        exit(0)

    print("Reading data from", data_path)
    features_np = read_features_np(data_path)
    features_pd = read_features_pd(data_path)
    labels_np = read_labels_np(data_path)
    labels_pd = read_labels_pd(data_path)
    print('Read', len(labels_np), 'data points with', len(features_np[0]), 'features in total.')

    print("Balancing dataset using random oversampling...")
    features_np, labels_np = random_oversample(list(features_np), list(labels_np))
    features_np, labels_np = unison_shuffled_copies(features_np, labels_np)
    print("Oversampling complete")

    feature_labels = get_all_features(features_pd)
    model = ParallelLogisticRegressionModel(num_features=len(feature_labels))
    TRAIN_SPLIT = args.split
    END_IDX = int(len(labels_np) * TRAIN_SPLIT)

    labels_np[labels_np == 0] = -1
    x_train = features_np[0:END_IDX]
    y_train = labels_np[0:END_IDX]
    x_test = features_np[END_IDX:]
    y_test = labels_np[END_IDX:]
    assert ((len(x_train) + len(x_test)) == (len(y_train) + len(y_test)) == len(features_np))

    train_data = parallelize_data(sc, x_train, y_train)
    test_data = parallelize_data(sc, x_test, y_test)

    print('Training on data from', args.data,
          'with: λ = %f, ε = %f, max iter = %d:' % (args.lam, args.eps, args.max_iter))
    t, k, losses, grad_norms, metrics_t, metrics_v, beta = model.trainRDD(train_data, test_data,
                                                                          lam=args.lam,
                                                                          eps=args.eps,
                                                                          max_iter=args.max_iter,
                                                                          N=args.N)
    print('Unparallelized logistic regression ran for', k, 'iterations. Converged', grad_norms[-1] < args.eps)
    print("Saving trained β in", args.beta)
    write_beta(args.beta, beta, feature_labels)

    print("Creating plots of results_logreg...")
    create_data_plots(k, losses, label="Loss")
    create_data_plots(k, grad_norms, label='GradNorm')
    create_metric_plots(k, metrics_t, label="Train")
    create_metric_plots(k, metrics_v, label="Test")
