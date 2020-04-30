# 
# 

# TODO

# Setting up
# 
# def nudge_dataset(X, Y):
    # """
    # This produces a dataset 5 times bigger than the original one,
    # by moving the 8x8 images in X around by 1px to left, right, down, up
    # """
    # direction_vectors = [
        # [[0, 1, 0],
        #  [0, 0, 0],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [1, 0, 0],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [0, 0, 1],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [0, 0, 0],
        #  [0, 1, 0]]]
# 
    # def shift(x, w):
        # return convolve(x.reshape((9, 9)), mode='constant', weights=w).ravel()
# 
    # X = np.concatenate([X] +
                    #    [np.apply_along_axis(shift, 1, X, vector)
                        # for vector in direction_vectors])
    # Y = np.concatenate([Y for _ in range(5)], axis=0)
    # return X, Y
# 

# Models we will use
# logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000,
                                        #    multi_class='multinomial')
# rbm = BernoulliRBM(random_state=0, verbose=True)
# 
# rbm_features_classifier = Pipeline(
    # steps=[('rbm', rbm), ('logistic', logistic)])
# 

