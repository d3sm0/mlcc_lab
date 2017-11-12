from pys.utils import *

dataset = load_pickle('moons_dataset.pkl')

y_hat = kNN(x=dataset['x_tr'], y=dataset['y_tr'], x_new=dataset['x_ts'], k=4)
train_error = MCE(y=dataset['y_ts'], y_hat=y_hat)
print('Train Error CV {}'.format(train_error))

best_k, train_stats = holdout_cv_kNN(data=dict(x=dataset['x_tr'], y = dataset['y_tr']), proportion=.2, k_list=list(range(1, 10)))
print(train_stats)

c = kernel_train(x=dataset['x_tr'], y=dataset['y_tr'], _gamma=.1, kernel='linear')
y_hat = kernel_test(x=dataset['x_tr'], x_new=dataset['x_ts'], c=c, kernel='linear', )
train_error = MCE(y=dataset['y_ts'], y_hat=y_hat)
print('Train Error CV {}'.format(train_error))

best_params, train_stats = holdout_cv_kernel(data=dict(x=dataset['x_tr'], y=dataset['y_tr']), proportion=.2, kernel='gaussian',
                                             _lambdas=np.arange(0.1, 1, .1), _gammas=np.arange(0.1, 1, .1))
print(train_stats)

w, r, i = OMP(x=dataset['x_tr'], y=dataset['y_tr'], n_iter=100)
train_error = hinge_loss(y=dataset['y_tr'], y_hat=dataset['x_tr'].dot(w))

print('Train Error CV {}'.format(train_error))

best_iter, train_stats = holdout_cv_OMP(data=dict(x = dataset['x_tr'], y = dataset['y_tr']), proportion=.2, init_iter=100)
print(train_stats)

x, y = mix_Gaussian(mu = np.array([[0,0],[0,1],[1,1],[1,0]]),sigma=np.array([0.3,0.3,0.3,0.3]))
u,s,v = PCA(x=x, k=2,should_normalize=True )
dataset = train_test_split(data = dict(x=x, y=y), proportion=.2, shuffle=True)
dataset['y_tr'] = flip_labels(y = dataset['y_tr'], proportion=.2)
dataset['y_ts'] = flip_labels(y=dataset['y_ts'], proportion=.2)

separating_kNN(x = dataset['x_tr'], y = dataset['y_tr'], step = .1, save_fig=True)

c = kernel_train(x=dataset['x_tr'], y=dataset['y_tr'], _gamma=.1, kernel='gaussian')
separating_kernel(x = dataset['x_tr'], x_new=dataset['x_ts'], y= dataset['y_tr'], c= c, kernel='gaussian', _gamma = .1, step = .1, save_fig=True)