import argparse
from cluster import create_dataset, load_data, preprocess_patches, dir_path
from clustering_application import load_data as load_data_measurements

from tslearn.preprocessing import TimeSeriesResampler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

RESAMPLING_DIVISOR = 1
N_NEIGHBORS = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='knn-classifier for sub-ROI patches')
    parser.add_argument('--norm', action='store_true', help='normalize time series')
    parser.add_argument('--train_path', type=dir_path, required=True, help='path to the training data to be processed')
    parser.add_argument('--test_path', type=dir_path, required=True, help='path to the test data to be processed')
    args = parser.parse_args()

    create_dataset(args.norm, args.train_path)
    x_train, y_train = load_data()
    x_train = preprocess_patches(x_train)

    create_dataset(args.norm, args.test_path)
    x_test, y_test, measurement_ids = load_data_measurements()
    x_test = preprocess_patches(x_test)

    print("original TS size:", len(x_train[0]))
    # # resample time series so that they reach the target size (sz - size of output TS)
    # #   -> we need to reduce the length of the TS (due to runtime, memory)
    x_train = TimeSeriesResampler(sz=len(x_train[0]) // RESAMPLING_DIVISOR).fit_transform(x_train)
    print("after down sampling:", len(x_train[0]))

    print("K-nearest neighbors")
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=N_NEIGHBORS,
                                         weights="distance"
                                         )
    fitted_knn = knn.fit(x_train, y_train)
    y_pred = fitted_knn.predict(x_test)

    print("Ground truth: ", y_test)
    print("Prediction: ", y_pred)

    assert len(y_test) == len(y_pred)
    accuracy = (y_test == y_pred).sum() / len(y_test)

    print("Accuracy: ", accuracy)
    print("-------------------------------------------------------------------------")
    print("Classification for each measurement: ")

    classification_per_measurement_id = {}

    for i in range(len(y_test)):
        if measurement_ids[i] in classification_per_measurement_id:
            classification_per_measurement_id[measurement_ids[i]][0].append(y_pred[i])
            classification_per_measurement_id[measurement_ids[i]][1].append(y_test[i])

        else:
            classification_per_measurement_id[measurement_ids[i]] = [
                [y_pred[i]], [y_test[i]]]

    for key, value in classification_per_measurement_id.items():
        print("Measurement ", key)
        print("Prediction: ", value[0])
        print("Ground Truth: ", value[1])
