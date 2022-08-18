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
  "models": {
    "values": ["FCN", "ResNet"]
  }
}
