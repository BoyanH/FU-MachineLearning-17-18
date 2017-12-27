import numpy as np


class AdaBoost:
    @staticmethod
    def boost_classifiers(classifier_pool, predictions, labels, k, fp_tolerate=0.7):
        data_size = len(labels)
        classifiers_count = len(classifier_pool)
        pos_size = len(labels[np.where(labels == 1)])
        neg_size = data_size - pos_size
        cw = []  # weights of chosen classifiers
        # initialize data set weights

        # here we take a slightly different approach from the general ada boost one
        # we have an fp_tolerate argument which can be set between 0 and 1
        # 0 fully ignores false positives and only tries to correct false negatives, 1 is opposite case
        # using this argument we can decide whether the error is equally separated between fp and fn
        data_weight = np.vectorize(lambda x: fp_tolerate / pos_size if x == 1 else (1 - fp_tolerate) / neg_size)(labels)
        classifiers = []

        # calculate error vector for each classifier
        # e.g. if a classifiers mistakes only 2. and 4. sample, its vector would be [0,1,0,1,0...,0]
        # that way, we can easily multiply the data-set weight vector by the error vector later on
        # to get e_i
        classifiers_e_vectors = np.zeros((classifiers_count, data_size), dtype=np.float64)
        for i in range(classifiers_count):
            wrong_idx = predictions[i] != labels
            # expected_pos = np.array(labels) == np.array([1] * len(labels))
            # false_negatives = np.logical_and(wrong_idx, expected_pos)
            # false_positives = np.invert(false_negatives)
            # wrongs = np.where(wrong_idx)

            classifiers_e_vectors[i][wrong_idx] = 1

        for j in range(k):
            e = classifiers_e_vectors.dot(data_weight) / data_weight.sum()
            idx_best = e.argmin()
            classifiers.append(classifier_pool[idx_best])

            e_i = e[idx_best]
            new_cw = np.log((1 - e_i) / (e_i + np.nextafter(0, 1)))
            cw.append(new_cw)

            signs = (classifiers_e_vectors[idx_best] - 0.5) * 2
            dw_update = np.vectorize(lambda s: np.exp(s * new_cw))(signs)
            data_weight = data_weight * dw_update

            classifier_pool = classifier_pool[:idx_best] + classifier_pool[idx_best + 1:]
            predictions = predictions[:idx_best] + predictions[idx_best + 1:]
            classifiers_e_vectors = np.delete(classifiers_e_vectors, idx_best, 0)

        return classifiers, np.array(cw)
