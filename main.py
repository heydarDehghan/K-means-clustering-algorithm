from services import *

from classification import KMeans

if __name__ == '__main__':
    data = load_data('datasets/kmeans/bird.tiff', array=True, show_data=False)
    k_number = 16
    k_means_model = KMeans(data, k_number, KMeansType.PLUS_PLUS_KMEANS)
    k_means_model.fit()
