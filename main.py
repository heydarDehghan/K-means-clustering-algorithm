from services import *

from classification import KMeans


def compress_image(means, index, img):
    # recovering the compressed image by
    # assigning each pixel to its corresponding centroid.
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],
                                       img.shape[2]))

    # plotting the compressed image.
    plt.imshow(recovered)
    plt.show()

    # saving the compressed image.
    # misc.imsave('compressed_' + str(clusters) +
    #             '_colors.png', recovered)


if __name__ == '__main__':
    data = load_data('datasets/kmeans/bird.tiff', array=False, show_data=True)
    k_number = 16
    k_means_model = KMeans(data, k_number, KMeansType.PLUS_PLUS_KMEANS)
    k_means_model.fit()

    compress_image(k_means_model.centroids, k_means_model.labels, data)
