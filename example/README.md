This folder contains an example of training an LSTM model to get good convergence. The trained model has three sequential LSTM cells each with 20 units. The training process is detailed below. 
## Training methodolgy
Tunable hyperparameters:
|parameter|value|
|---------|-----|
|output period|500 us|
|training length|400 samples|
|learning rate|0.0005|
|batch size|8|
|epochs|75|
### Preprocessing
Preprocessing consists of:
1. Resampling the roller signal to the desired output frequency and the accelerometer signal to sixteen times the output frequency. Then we reshape the accelerometer signal to the output frequency with sixteen features at each timestep.
2. Normalizing both signals by dividing by the standard deviation.
### Model training
The dataset is too large to train on the entire thing at once. Instead, we divide the signal into sequences of 400 samples and train on each sequence. At 500 us sampling and 400 samples per sequence, the dataset is divided into 212 sequences. We fit the model with an Adam optimizer with learning rate 0.0005 and a batch size of 8. I found the small batch size helpful because it increases the amount of weight updates per epoch. For the given LSTM model, it converged in about 75 epochs and final SNR was 23.06 dB.
