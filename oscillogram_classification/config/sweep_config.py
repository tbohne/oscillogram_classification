sweep_config = {
    "batch_size": {
        "values": [2, 16]
    },
    "learning_rate": {
        "values": [0.01, 0.0001]
    },
    "optimizer": {
        "value": "keras.optimizers.Adam"
    },
    "epochs": {
        "value": 3
    },
    "model": {
        "values": ["FCN", "ResNet"]
    },
    "loss_function": {
        "value": "sparse_categorical_crossentropy"
    },
    "accuracy_metric": {
        "value": "sparse_categorical_accuracy"
    }
}
