# The GMM model was developed on an ipython notebook #

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from sklearn.datasets import load_digits

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='mnist')
print (type(mnist))

print(mnist.data[:1000,:].data.shape)

digits = mnist
plt.imshow(mnist.data[0].reshape(28, 28), cmap='binary')
digits.target[0]

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(28, 28),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(28, 28), cmap='binary')
        im.set_clim(0, 16)

plot_digits(digits.data[50000:,:])

# use a straightforward PCA, asking it to preserve 99% of the variance in the projected data
from sklearn.decomposition import PCA

pca = PCA(0.99, whiten=True)
print (np.random.shuffle(digits.data[:1000,:]))
data = pca.fit_transform(digits.data[:1000,:])

plot_digits(digits.data[:1000,:])

# Use of AIC to get a gauge of number of GMM components to use
from sklearn.mixture import GaussianMixture

n_components = np.arange(50, 210, 10)
models = []
for n in n_components:
    print (n)
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
