import argparse
from pyspark import SparkContext
from ParallelLogisticRegressionModel import ParallelLogisticRegressionModel
from LogRegUtils import *


def parallelize_data(spark_context, features, labels, N):
    """
    Parallelize a given dataset into an RDD for use with pyspark commands

    Inputs:
        spark_context: SparkContext
            A spark object used to call the parallelize function
        features: np.array
            The features to parallelize
        labels: np.array
            The labels to parallelize
        N: int
            The number of partitions to split the data into
    Returns:
        An RDD with each element in the form (feature, label) with N partitions
    """
    result = []
    for feature, label in zip(features, labels):
        result.append((feature, label))
    return spark_context.parallelize(result, numSlices=N)


if __name__ == "__main__":
    # Set up argument parser and associated flags
    parser = argparse.ArgumentParser(description="Logistic Regression.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../../data/Clean_data.csv', help='Input file containing all features and labels, used to train a logistic model')
    parser.add_argument('--beta', default='./results/beta', help='File where beta is stored')
    parser.add_argument('--split', type=float, default=0.8, help='Test/Training split. Percentage of data to be used for training')
    parser.add_argument('--lam', type=float, default=0.0, help="Regularization parameter λ")
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--N', type=int, default=20, help='Level of parallelism/number of partitions')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--alpha', type=float, default=1e-7, help='Learning rate controlling the step size of gradient descent')

    # Set up verbosity of pyspark
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

    # Create model
    feature_labels = get_all_features(features_pd)
    model = ParallelLogisticRegressionModel(num_features=len(feature_labels), learning_rate=args.alpha)

    # Create training split
    TRAIN_SPLIT = args.split
    END_IDX = int(len(labels_np) * TRAIN_SPLIT)
    labels_np[labels_np == 0] = -1
    x_train = features_np[0:END_IDX]
    y_train = labels_np[0:END_IDX]
    x_test = features_np[END_IDX:]
    y_test = labels_np[END_IDX:]
    assert ((len(x_train) + len(x_test)) == (len(y_train) + len(y_test)) == len(features_np))

    # Parallelize training and testing data with N partitions
    train_data = parallelize_data(sc, x_train, y_train, args.N)
    test_data = parallelize_data(sc, x_test, y_test, args.N)

    print('Training on data from', args.data, 'with: λ = %f, ε = %f, max iter = %d:' % (args.lam, args.eps, args.max_iter))
    t, k, losses, grad_norms, metrics_t, metrics_v, beta = model.trainRDD(train_data, test_data,
                                                                          lam=args.lam,
                                                                          eps=args.eps,
                                                                          max_iter=args.max_iter)
    print('Unparallelized logistic regression ran for', k, 'iterations. Converged', grad_norms[-1] < args.eps)
    print("Saving trained β in", args.beta)
    write_beta(args.beta, beta, feature_labels)

    print("Creating plots of results...")
    create_data_plots(k, losses, label="Loss")
    create_data_plots(k, grad_norms, label='GradNorm')
    create_metric_plots(k, metrics_t, label="Train")
    create_metric_plots(k, metrics_v, label="Test")
