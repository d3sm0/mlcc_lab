import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def MCE(y, y_hat, bound=0):
    """
    Compute the multi class miss classification error as 1/n sum(y_hat != y)
    :param y_hat: predicted y
    :param y: true y
    :param bound: confidence bound
    :return: error estimate
    """
    y_hat = (y_hat >= bound)
    y = (y >= bound)
    return np.mean(y_hat != y)


def accuracy(y_hat, y, bound=0):
    """
    Compute the accuracy as 1-MCE
    """
    return 1 - MCE(y=y, y_hat=y_hat, bound=bound)


def l2_loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def hinge_loss(y, y_hat):
    """
    Compute hinge loss as mean( sign(y) != sign(y_hat))
    :param y_hat: predicted y
    :param y: true y
    :return: error estimate
    """
    return np.mean(np.sign(y) != np.sign(y_hat))


def flip_labels(y, proportion=.2, shuffle=True):
    """
    Change the label from 0,1 to -1, 1.
    :param y: true label
    :param p: proportion of the dataset to be returned
    :param shuffle: shuffle the labels if True
    :return: label vector
    """
    nr_points = y.shape[0]
    y_p = y.copy()
    if shuffle:
        subset = np.random.permutation(nr_points)[:int(nr_points * proportion)]
    else:
        subset = np.arange(0, int(nr_points * proportion))

    # idxs = np.random.permutation(n)[:int(n * p)]
    y_p[subset] = -1 * y_p[subset]
    return y_p


def kernel_matrix(x1, x2, gamma=.1, kernel='l2'):
    """
    Compute the kernel matrix. In the linear case is X X'
    :param x1: N x D matrix
    :param x2: N x D matrix
    :param gamma: kernel parameter
    :param kernel: l2, gaussian, polynomial
    :return: D x D matrix
    """
    if kernel == 'l2':
        K = l2(x1=x1, x2=x2, p=2)
    elif kernel == 'linear':
        K = x1.dot(x2.T)
    elif kernel == 'gaussian':
        D = l2(x1=x1, x2=x2, p=2)
        K = np.exp(-D / (2 * gamma ** 2))
    elif kernel == 'polynomial':
        K = (1 + x1.dot(x2.T)) ** gamma
    else:
        print('No kernel found')
        raise NotImplementedError()
    return K


def kNN(x, y, x_new, k=4):
    """
    Compute k-nearest neighbors algorithm
    :param x: N x D feature vector
    :param y: N x 1 labels
    :param x_new: N X D datapoints
    :param k: N. of neighbors to consider
    :return: N x 1 predicted y
    """
    D = l2(x1=x, x2=x_new)
    ys = []
    for d in D.T:
        idxs = np.argsort(d)[:k]
        y_hat = np.sum(y[idxs]) / k
        ys.append(np.sign(y_hat))
    return np.array(ys)


def mix_Gaussian(mu, sigma, n=100):
    """
    Generate data from a mixture of gaussian distribution
    :param mu: P x D matrix
    :param sigma: D x 1 vector
    :param n: N. of samples
    :return: X, Y
    """
    p, d = mu.shape
    X = []
    Y = []
    for idx in range(p):
        x = np.random.multivariate_normal(mean=mu[idx], cov=sigma[idx] * np.identity(d), size=n)
        y = np.repeat(idx, n)
        X.append(x)
        Y.append(y)
    return np.vstack(X), np.array(Y).flatten()


def PCA(x, k, should_normalize=False):
    """
    Compute Principal Component Analysis using an estimate of the covariance matrix built on x
    :param x: N X D feature vector
    :param k: N. of eigen values to consider
    :param should_normalize: if data should be normalized
    :return: N x D matrix, eigenvalues, eigenvectors
    """
    x_pca = x.copy()
    if should_normalize:
        x_pca -= x.mean(axis=0)
    # computing an estimate of the covariance matrix
    cov = np.cov(x, rowvar=False)
    S, V = np.linalg.eigh(cov)
    idx = np.argsort(S)[::-1][:k]
    # the index of the largest direction
    V = V[:, idx]
    S = S[idx]
    U = np.dot(x_pca, V)
    # Notice this would be as an SVD application where: U, S, V
    return U, S, V


def kernel_train(x, y, kernel='linear', _gamma=0.1, _lambda=1.):
    """
    Compute the exact solution least square matrix in the kernel setting
    :param x: N x D feature vector
    :param y: N x 1 label
    :param kernel: kernel can be 'linear', 'polynomial', 'gaussian'
    :param _gamma: kernel parameter
    :param _lambda: regularizer
    :return: N parameters
    """
    n = x.shape[0]
    K = kernel_matrix(x, x, _gamma, kernel)
    assert K.shape[0] == K.shape[1]
    c = np.linalg.solve(K + _lambda * n * np.eye(n), y)
    return c


def kernel_test(x, x_new, c, _gamma=1., kernel='linear'):
    """
    Compute the prediction with the trained kernel
    :param x: training input
    :param x_new: new data
    :param c: model weights
    :param _gamma: kernel parameter
    :param kernel: kernel type
    :return: predicted values
    """
    Ktest = kernel_matrix(x_new, x, _gamma, kernel)
    return Ktest.dot(c)


def separating_kernel(x, x_new, y, c, kernel='linear', _gamma=1., step=.01, save_fig=True):
    """
    The function classifies points evenly sampled in a visualization area,
    according to the classifier Regularized Least Square.
    :param x: N x D input data
    :param x_new: N x D test data
    :param y: N x 1 labels
    :param c: pre trained coefficients
    :param kernel: kernel type
    :param _gamma: kernel parameter
    :param step: distance between grid points
    :param save_fig: if true, it saves the plot
    :return:
    """
    m0, m1 = np.min(x_new, axis=0)
    M0, M1 = np.max(x_new, axis=0)
    x_axis = np.arange(m0, M0, step)
    y_axis = np.arange(m1, M1, step)

    xx, yy = np.meshgrid(x_axis, y_axis)
    x_grid = np.column_stack((xx.flatten(), yy.flatten()))
    y_grid = kernel_test(x=x, x_new=x_grid, c=c, kernel=kernel, _gamma=_gamma)
    # fig, ax = plt.subplots()
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y)  # 'filled')
    for x0, x1, txt in zip(x[0], x[1], y):
        plt.annotate(txt, (x0, x1))
    plt.contour(xx, yy, np.reshape(y_grid, (xx.shape[0], yy.shape[1])))
    if save_fig:
        plt.savefig('Kernel_decision_boundary')
    else:
        plt.show()


def separating_kNN(x, y, step=.01, save_fig=True):
    """
    The function classifies points evenly sampled in a visualization area,
    according to the classifier kNNClassify
    :param x: N x D Data matrix
    :param y: 1D vector with labels
    :param step: distance between grid points
    :param save_fig: if true, save plot
    :return:
    """
    m1, m2 = np.min(x, axis=0)
    M1, M2 = np.max(x, axis=0)

    x_axis = np.arange(m1, M1, step)
    y_axis = np.arange(m2, M2, step)

    xx, yy = np.meshgrid(x_axis, y_axis)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    y_grid = kNN(x=x, y=y, x_new=grid, k=4)

    plt.contour(xx, yy, y_grid.reshape((xx.shape[0], yy.shape[1])))
    if save_fig:
        plt.savefig('kNN_decision_boundary')
    else:
        plt.show()


def l2(x1, x2, p=2):
    """
    Compute square matrix
    :param x1: N x D matrix
    :param x2: N x D matrix
    :param p: norm_ parameter
    :return: D x D squared matrix
    """
    D = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** p, axis=-1)
    return D


def load_mat(file_path):
    """
    Loads a Matlab .mat file ignoring
    any variable including the "_" character
    :param file_path: path of the file .mat
    :return: Dataset as dictionary
    """
    from scipy.io import loadmat
    data = loadmat(file_path)
    tmp = {}
    try:
        for k in data.keys():
            if '_' not in k:
                tmp[k] = data[k]
    except Exception as e:
        print('Something went wrong loading {}'.format(file_path), e)
        raise e
    return tmp


def load_pickle(file_path='moons_dataset.pkl'):
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)


def save_pickle(data, file_path):
    os.makedirs(file_path[:file_path.rindex(os.path.sep)], exist_ok=True)
    with open(file_path, 'wb') as fout:
        pickle.dump(data, fout)
    print('File saved')


def two_moons(n, proportion=0.2):
    """
    Load two moons dataset if available. Otherwise create a new dataset
    :param n: number of samples
    :param proportion: of
    :return: dataset as dictionary
    """
    dataset = load_pickle(file_path='moons_dataset.pkl')
    if dataset['x_tr'].shape[0] < n:
        from sklearn.datasets import make_moons
        x, y = make_moons(n_samples=n * (1 + proportion))
        dataset = train_test_split(data=dict(x=x, y=y), proportion=proportion, shuffle=True)
    dataset['y_tr'] = flip_labels(dataset['y_tr'], proportion=proportion)
    dataset['y_ts'] = flip_labels(dataset['y_ts'], proportion=proportion)
    return dataset


def create_grid_points(x0, x1, step=.1):
    """
    Compute a grid of points up to 3 dimension
    :param x0: lower bound
    :param x1: upper bound
    :param step: grid density
    :return: X x Y grid
    """
    if np.ndim(x0) <= 1:
        grid = np.arange(x0, x1, step)
    elif np.ndim(x0) == 2:
        xx, yy = np.mgrid[x0[0]:x1[0]:step, x0[1]:x1[1]:step]
        grid = np.column_stack((xx.flatten(), yy.flatten()))
    elif np.ndim(x0) == 3:
        xx, yy, zz = np.mgrid[x0[0]:x1[0]:step, x0[1]:x1[1]:step, x0[2]:x1[2]:step]
        grid = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    else:
        grid = None
        print('Too many dimensions')
    return grid


def train_test_split(data, proportion, shuffle=True):
    """
    Splits into training and test set and optionally shuffle
    :param data: dataset dictionary
    :param proportion: proportion of splitting
    :returns: a dictionary containing the split dataset
    """
    n = data['x'].shape[0]
    n_folds = int(np.ceil(n * (1 - proportion)))
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)

    new_data = {}
    for k, v in data.items():
        key = k + '_tr'
        new_data[key] = v[idxs[:n_folds]]
        key = k + '_ts'
        new_data[key] = v[idxs[n_folds:]]
    return new_data


def holdout_cv_kNN(data, proportion=.2, k_list=(2,), reps=10):
    """
    Perform a hold out cross validation procedure on the available training data
    and reports the best k.
    :param data: dataset
    :param proportion: proportion of the dataset
    :param k_list: list of candidates ks
    :param reps: number of repettion for each fold
    :return best_k, train stats
    """
    assert len(k_list) > 1
    train_stats = {
        'tr_error': np.zeros(len(k_list)),
        'ts_error': np.zeros(len(k_list))
    }

    y_m = (np.max(data['y']) + np.min(data['y'])) / 2

    for idx, k in enumerate(k_list):
        for step in range(reps):
            dataset = train_test_split(data=data, proportion=proportion)
            y_hat = kNN(dataset['x_tr'], dataset['y_tr'], dataset['x_tr'], k=k)
            error = MCE(y_hat, dataset['y_tr'], bound=y_m)
            train_stats['tr_error'][idx] += error
            y_hat = kNN(dataset['x_tr'], dataset['y_tr'], dataset['x_ts'], k=k)
            error = MCE(y_hat, dataset['y_ts'], bound=y_m)
            train_stats['ts_error'][idx] += error

    best_k = np.argmin(train_stats['ts_error'])
    train_stats['ts_error'] = train_stats['ts_error'][best_k]
    train_stats['tr_error'] = train_stats['tr_error'][best_k]
    return best_k, train_stats


def holdout_cv_OMP(data, proportion=.2, init_iter=50, reps=10):
    """
    Perform holdout cross validation for OMP
    :param data: Dataset as dictionary
    :param proportion: proportion of the dataset for validation
    :param init_iter: list of candidate iterations
    :param reps: repetition for each fold
    :return: best_iter, train_stats
    """

    train_stats = {
        'tr_error': np.zeros((init_iter, reps)),
        'ts_error': np.zeros((init_iter, reps))
    }

    for iter_idx in range(init_iter):
        for rep_idx in range(reps):
            dataset = train_test_split(data=data, proportion=proportion)
            w, _, _ = OMP(x=dataset['x_tr'], y=dataset['y_tr'], n_iter=iter_idx)
            train_stats['tr_error'][iter_idx, rep_idx] = hinge_loss(y=dataset['x_tr'].dot(w), y_hat=dataset['y_tr'])
            train_stats['ts_error'][iter_idx, rep_idx] = hinge_loss(y=dataset['x_ts'].dot(w), y_hat=dataset['y_ts'])

    tr_median, tr_std = np.median(train_stats['tr_error'], axis=1), train_stats['tr_error'].std(axis=1)
    ts_median, ts_std = np.median(train_stats['ts_error'], axis=1), train_stats['ts_error'].std(axis=1)
    best_iter = np.where(ts_median <= np.min(ts_median))[0]
    return best_iter, {
        'tr_median': np.mean(tr_median),
        'tr_std': np.mean(tr_std),
        'ts_median': np.mean(ts_median),
        'ts_std': np.mean(ts_std)
    }


def holdout_cv_kernel(data, proportion=.2, kernel='linear', _lambdas=(.1, .2), _gammas=(.1, .2), reps=10):
    """
    Perform holdout CV for regularized kernel method
    :param data: Dataset dictionary
    :param proportion: proportion of the train test split
    :param kernel: Kernel type
    :param _lambdas: list of candidate penalization factor
    :param _gammas: list of parameters for the kernel
    :param reps: repetition for each fold
    :return: (best_lambda, best_sigma), train_stats
    """
    train_stats = {
        'tr_error_median': np.zeros((len(_lambdas), len(_gammas))),
        'tr_error_sd': np.zeros((len(_lambdas), len(_gammas))),
        'ts_error_median': np.zeros((len(_lambdas), len(_gammas))),
        'ts_error_sd': np.zeros((len(_lambdas), len(_gammas)))
    }

    y_m = (np.max(data['y']) + np.min(data['y'])) / 2
    for lambda_idx, lambda_val in enumerate(_lambdas):
        for gamma_idx, _gamma in enumerate(_gammas):
            training_error, test_error = np.zeros(reps), np.zeros(reps)
            for rep in range(reps):
                dataset = train_test_split(data=data, proportion=proportion, shuffle=True)
                c = kernel_train(x=dataset['x_tr'], y=dataset['y_tr'], kernel=kernel, _gamma=_gamma, _lambda=lambda_val)
                y_hat = kernel_test(x=dataset['x_tr'], x_new=dataset['x_tr'], c=c, _gamma=_gamma, kernel=kernel)
                training_error += MCE(y_hat, y_hat=dataset['y_tr'], bound=y_m)
                y_hat = kernel_test(x=dataset['x_tr'], x_new=dataset['x_ts'], c=c, _gamma=_gamma, kernel=kernel)
                test_error += MCE(y_hat, y_hat=dataset['y_ts'], bound=y_m)
            train_stats['tr_error_median'][lambda_idx, gamma_idx] = np.median(training_error)
            train_stats['tr_error_sd'][lambda_idx, gamma_idx] = np.std(training_error)

            train_stats['ts_error_median'][lambda_idx, gamma_idx] = np.median(test_error)
            train_stats['ts_error_sd'][lambda_idx, gamma_idx] = np.std(test_error)

    lambda_idx, gamma_idx = np.where(train_stats['ts_error_median'] <= np.min(train_stats['ts_error_median']))
    return (_lambdas[lambda_idx[0]], _gammas[gamma_idx[0]]), {k: v.mean() for k, v in train_stats.items()}


def OMP(x, y, n_iter=50):
    """
    Orthogonal matching pursuit
    :param x: N x D feature vector
    :param y: labels
    :param n_iter: number of iterations
    :return: weight matrix, residuals and indices
    """
    N, D = x.shape
    residual = y.copy()
    # Initialization of residual, coefficient vector and index set I
    # r = Y
    w = np.zeros((D, 1))
    indices = []
    assert np.all(np.isreal(x))

    for i in range(n_iter):
        # Select the column of X which most "explains" the residual
        a_max = np.dot(residual.T, x[:, column]) ** 2 / np.dot(x[:, column].T, x[:, column])
        best_column = 0

        for column in range(1, D):
            a_tmp = np.dot(residual.T, x[:, column]) ** 2 / np.dot(x[:, column].T, x[:, column])
            if a_tmp > a_max:
                a_max = a_tmp
                best_column = column

        # zz = np.dot(residual.T, x) ** 2 / np.dot(x.T, x)
        # am = np.argmax(zz)
        # print(am, best_column)

        # Add the index to the set of indexes
        if np.sum([np.where(index == best_column) for index in indices]) == 0:
            indices.append(best_column)

        # Compute the M matrix
        M_I = np.zeros((D, D))
        # for column in indices:
        #     M_I[column, column] = 1
        M_I[indices, indices] = 1

        A = M_I.dot(x.T).dot(x).dot(M_I)
        B = M_I.dot(x.T).dot(y)
        # Update w
        w = np.linalg.pinv(A).dot(B)

        # Update the residual
        residual = y - x.dot(w)

    return w, residual, indices