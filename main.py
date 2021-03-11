import numpy as np
import numpy.matlib
import scipy.io
import scipy.signal
import sklearn
from sklearn.decomposition import PCA
from nnsg import nnsg

base = scipy.io.loadmat('data/YaleB_32x32.mat')
features = base['fea']
labels = base['gnd']

parameter_alpha = [0.001]
parameter_lambda = [0.5]
parameter_beta = [0.01]
number_of_repetitions = 30
select_number = 5


number_of_components = 60
pca_model = PCA(n_components=number_of_components, svd_solver='full')
features = pca_model.fit_transform(features)
nn_class, num_class = np.unique(labels, return_counts=True)


accuracy = []
for i in range(number_of_repetitions):
    train_features = np.zeros((select_number*len(nn_class), number_of_components))
    train_labels = np.zeros((select_number*len(nn_class)))
    # test_features = np.zeros((features.shape[0]-train_features.shape[0], number_of_components))
    # test_labels = np.zeros((features.shape[0]-train_features.shape[0]))
    for j in range(len(nn_class)):
        idx = np.where(labels == nn_class[j])
        idx = idx[0]
        rand_idx = np.random.permutation(num_class[j])
        train_features[j*select_number:(j + 1)*select_number, :] = features[idx[rand_idx[0:select_number]], :]
        train_labels[j*select_number:(j + 1)*select_number] = numpy.reshape(labels[idx[rand_idx[0:select_number]]], select_number)
        if j == 0:
            test_features = features[idx[rand_idx[select_number:]], :]
            test_labels = labels[idx[rand_idx[select_number:]]]
        else:
            test_features_new = features[idx[rand_idx[select_number:]], :]
            test_labels_new = labels[idx[rand_idx[select_number:]]]
            test_features = np.concatenate((test_features, test_features_new))
            test_labels = np.concatenate((test_labels, test_labels_new))

    train_features = sklearn.preprocessing.normalize(train_features)
    test_features = sklearn.preprocessing.normalize(test_features)

    complete_matrix = np.concatenate((train_features, test_features))

    # best_acurracy = 0
    for alpha1 in parameter_alpha:
        for lambda1 in parameter_lambda:
            for beta1 in parameter_beta:
                W, F, S, obj = nnsg(train_labels, np.transpose(complete_matrix), alpha1, lambda1, beta1)
                predicted_labels = F[len(train_labels):, :]
                id_2 = np.argsort(predicted_labels)
                id_3 = id_2[:, -1] + 1
                current_accuracy = sklearn.metrics.accuracy_score(test_labels, id_3)
                # if current_accuracy > best_acurracy:
                   # best_acurracy = current_accuracy
                   # best_p1 = p1
                   # best_p2 = p2
                   # best_p3 = p3
                accuracy.append(current_accuracy)
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)

np.savetxt('Accuracy.csv', accuracy, delimiter=',')