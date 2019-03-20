# The GMM model was developed on an ipython notebook #

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import torch

import pickle
flag = True

pickle_file = "anime.pkl"
pickle_file = open(pickle_file, 'rb')
try:
    for i in range(1000):
        imgarray = pickle.load(pickle_file)
        imgarray = imgarray.flatten()
        imgarray = imgarray.reshape([1, imgarray.shape[0]])
        if flag:
            anime = imgarray
            flag = False
        else:
            anime = np.vstack((anime, imgarray))
except EOFError:
    pass

plt.imshow(anime[0].reshape(64, 64), cmap='Greys_r')

# The plot digits method was used to plot the digits on the Ipython notebook

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(64, 64),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(64, 64), cmap='Greys_r')

plot_digits(anime)
print (type(anime))

# use a straightforward PCA, asking it to preserve 99% of the variance in the projected data
from sklearn.decomposition import PCA

pca = PCA(0.99, whiten=True)
print (np.random.shuffle(anime))
data = pca.fit_transform(anime)
print(data.shape)

plot_digits(anime)

# Use of AIC to get a gauge of number of GMM components to use
from sklearn.mixture import GaussianMixture

n_components = np.arange(50, 210, 10)
models = []
for n in n_components:
    models.append(GaussianMixture(n, covariance_type='full', random_state=0))
aics = []
for model in models:
    print (str(model))
    aics.append(model.fit(data).aic(data))
plt.plot(n_components, aics);


# Use of 110 as components number
gmm = GaussianMixture(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

# Generate new data
data_new = gmm.sample(100)
data_new[0].shape

# use the inverse transform of the PCA object to construct the new digits
digits_new = pca.inverse_transform(data_new[0])
plot_digits(digits_new)
