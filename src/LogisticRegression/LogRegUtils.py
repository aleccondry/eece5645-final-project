import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


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


def create_data_plots(k, data, label=""):
    plt.plot([i for i in range(1, k)], data, label=label)
    plt.xlabel("Number of Iterations")
    plt.ylabel(label)
    plt.title(f'{label} vs Number of Iterations')
    plt.savefig(f'./results/{label}_fig_unparallelized.png')


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
    plt.savefig(f'./results/metrics_fig_unparallelized.png')
