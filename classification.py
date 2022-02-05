import collections
import numpy as np
from services import KMeansType
from services import Data


class KMeans:

    def __init__(self, data, k_number=16, kmeans_type: KMeansType = KMeansType.PLUS_PLUS_KMEANS):
        self.data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
        self.k_number = k_number
        self.kmeans_type = kmeans_type
        self.WCSS_history = []
        self.current_wcss = None
        self.labels = np.zeros((self.data.shape[0], 1))
        self.centroids = np.zeros((self.k_number, self.data.shape[1]))
        self.init_centroid()

    def calc_dist(self, row_index):
        main_list_dist = []
        for point_main in self.centroids:
            list_dict = [(index, np.sum(np.square(np.subtract(point, point_main)))) for index, point in enumerate(self.data)]
            list_dict.sort(key=lambda x: x[1])
            for min_dist_point in list_dict:
                if min_dist_point not in main_list_dist and min_dist_point[1] != 0:
                    main_list_dist.append(min_dist_point)
                    break

        main_list_dist.sort(key=lambda x: x[1], reverse=True)
        for max_dist_point in main_list_dist:
            v = self.centroids[:, :] == self.data[max_dist_point[0]]
            if not np.any(np.all(v, axis=1)):
                self.centroids[row_index] = self.data[max_dist_point[0]]
                break

    def init_centroid_plus_plus(self):
        first_coordinate = self.data[np.random.choice(self.data.shape[0])]

        # centroid_points = np.zeros((self.k_number, self.data.shape[1],  self.data.shape[-1]))
        self.centroids[0] = first_coordinate
        for x in range(1, self.k_number):
            self.calc_dist(x)

        print(self.centroids)

    def init_centroid(self):
        if self.kmeans_type == KMeansType.PLUS_PLUS_KMEANS:
            self.init_centroid_plus_plus()

    def update_centroid(self):
        for index in range(self.k_number):
            points_with_label_k = self.data[self.labels == index]
            x = np.mean(points_with_label_k, axis=0)
            self.centroids[index] = np.mean(points_with_label_k, axis=0)

    def calculate_wcss(self):
        class_variance_list = []
        for index, center in enumerate(self.centroids):
            points_with_label_k = self.data[self.labels == index]
            sum_distance_class_k = np.sum([np.sum(np.subtract(center, point) ** 2) for point in points_with_label_k])
            class_variance_list.append(sum_distance_class_k)

        self.current_wcss = np.sum(class_variance_list)

    def predict_label(self):
        labels_array = np.zeros((self.data.shape[0], self.k_number))
        for index, point in enumerate(self.data):
            labels_array[index] = np.array([np.sum(np.subtract(center_k, point) ** 2) for center_k in self.centroids])
        self.labels = np.argmin(labels_array, axis=1)

    def fit(self):
        self.predict_label()
        self.update_centroid()
        for x in range(100):
            self.calculate_wcss()
            if len(self.WCSS_history) > 0 and self.WCSS_history[-1] - self.current_wcss == 0 :
                break
            self.WCSS_history.append(self.current_wcss)
            self.predict_label()
            self.update_centroid()

            print(f'subtract of wcss in iterationn {x} is :{np.abs(self.WCSS_history[-1] - self.current_wcss)}')

        print(np.abs(self.WCSS_history[-1] - self.current_wcss))


