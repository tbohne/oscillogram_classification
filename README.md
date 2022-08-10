# oscillogram_classification

Neural network based anomaly detection for vehicle components using oscilloscope recordings.

Example of the time series data to be considered (voltage over time).
![](img/plot.png)

The task comes down to binary time series classification.

## CNN Architecture

<img src="img/model.png" width="300">

## Positive and Negative Sample for each Component

### Battery:
<img src="img/example.png" width="420">

## Training and Validation Loss

### Mini-Batch Gradient Descent
<img src="img/mini_batch_gd.png" width="420">

### Stochastic Gradient Descent
<img src="img/stochastic_gd.png" width="420">

### Grad-CAM Example
![](img/heatmap.png)
