from scipy.spatial.distance import euclidean
import numpy as np

# Define the KNN class with two empty methods - fit and predict
class KNN():
    
    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train

    def predict(self, X_test, k=3):
        preds = []
        # Iterate through each item in X_test
        for i in X_test:
            # Get distances between i and each item in X_train
            dists = self._get_distances(i)
            k_nearest = self._get_k_nearest(dists, k)
            predicted_label = self._get_label_prediction(k_nearest)
            preds.append(predicted_label)
        return preds

    def _get_distances(self, x):
        distances = []
        for i, v in enumerate(self.X_train):
            dist_to_i = euclidean(x, v)
            distances.append((i, dist_to_i))
        return distances

    def _get_k_nearest(self, dists, k):
        sorted_dists = sorted(dists, key=lambda x: x[1])
        return sorted_dists[:k]

    def _get_label_prediction(self, k_nearest):
        labels = [self.y_train[i] for i, _ in k_nearest]
        counts = np.bincount(labels)
        return np.argmax(counts)



# Import the necessary functions
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# iris = load_iris()
# data = iris.data
# target = iris.target

# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

# # Instantiate and fit KNN
# knn = KNN()
# knn.fit(X_train,y_train)

# # Generate predictions
# preds = knn.predict(X_test)

# print("Testing Accuracy: {}".format(accuracy_score(y_test, preds)))
# # Expected Output: Testing Accuracy: 0.9736842105263158






from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score,roc_auc_score

clf = KNeighborsClassifier()

clf.fit(X_train,y_train)
y_hat_test = pd.Series(clf.predict(X_test),index=y_test.index)
y_hat_train= pd.Series(clf.predict(X_train), index=y_train.index)
type(y_hat_test)

res = [['Metric/Function','Value']]

for func in [accuracy_score,roc_auc_score,f1_score,precision_score]:
    
    try: 
        score = func(y_test, y_hat_test)
        print(func.__name__+ '=' + str(score))
        res.append([func.__name__,score])
    except:
        import pdb 
        pdb.set_trace()
        
        res.append([func.__name__,'Error'])
        
res

res_df = pd.DataFrame(res[1:],columns=res[0])
res_df

preds = pd.concat([y_hat_train,y_hat_test])
preds.name ='preds'
preds

# Making funcs for gridsearch
def make_model(X_train, y_traibn, model_kws={}):
    clf = KNeighborsClassifier(**model_kws)
    

    clf.fit(X_train,y_train)

    y_hat_test = pd.Series(clf.predict(X_test),index=y_test.index)
    y_hat_train= pd.Series(clf.predict(X_train), index=y_train.index)

    res = [['Metric/Function','Value',"model_kws"]]

    for func in [accuracy_score,roc_auc_score,f1_score,precision_score]:

        try: 
            score = func(y_test, y_hat_test)
#             print(func.__name__+ '=' + str(score))
            res.append([func.__name__,score,model_kws])
        except:

            res.append([func.__name__,'Error',model_kws])


    res_df = pd.DataFrame(res[1:],columns=res[0])
#     model_dict=dict(results = res_df, model=clf)
    
    
    return res_df


# df_search = pd.DataFrame()
# for k in [3,5,7]:
#     mod_df = make_model(X_train, y_train, model_kws ={'n_neighbors':k})
#     df_search = pd.concat([df_search, mod_df],axis=0)

# df_search['k'] = df_search['model_kws'].apply(lambda x: x['n_neighbors'])
# df_search

# df_search.sort_values(['Metric/Function','Value'],ascending=False)



