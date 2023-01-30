import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

    
def plot_multi_vectors_as_images(image, h, w, cols, rows):
    plt.figure(figsize=(1.5 * cols, 2.2 * rows))
    plt.subplots_adjust(0, 0, 1.5, 1.5)
    for k in range(rows*cols):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(image[k].reshape((h, w)), cmap=plt.cm.gray)
        plt.yticks(())
        plt.xticks(())
    plt.tight_layout()
    plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h*w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w

def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
              of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """

    # Compute the SVD of the zero-mean data
    u, s, vt = np.linalg.svd(X)
    # Select the top k eigenvectors
    U = vt[:k, :]
    S = np.reshape((s**2)[:k], (k,1))
    return U, S


def exercise_b(X):
    U, S = PCA(X, k=10)
    plot_multi_vectors_as_images(U[:U.shape[0], :], h, w, 5, 2)


def exercise_c(X):
    ks = np.array([1, 5, 10, 30, 50, 100])
    l2_norm = np.zeros(ks.shape)
    for index in range(len(ks)):
    #for k in k_range:
        U, S = PCA(X, ks[index])
        X_hat = X.dot(np.transpose(U)).dot(U)
        random_rows = np.random.randint(0, X_hat.shape[0], 5)
        l2_norm[index] = np.sum(np.linalg.norm(X[random_rows, :] - X_hat[random_rows, :], axis=1))
        plot_multi_vectors_as_images(np.concatenate((X[random_rows, :], X_hat[random_rows, :]), axis=0), h, w, 5, 2)
    plt.plot(ks, l2_norm)
    plt.xlabel('k')
    plt.ylabel('ℓ2')
    plt.title('ℓ2 distances as a function of k')
    plt.show()


selected_images, h, w = get_pictures_by_name(name='Donald Rumsfeld')
X = np.array(selected_images)[:, :, 0]
# Zero mean the data
X = X - np.mean(X, axis=0)
exercise_b(X)
exercise_c(X)    