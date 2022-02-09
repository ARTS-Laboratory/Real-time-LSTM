import imageio
import os
import matplotlib.pyplot as plt
import tempfile as tmp
from tempfile import mkstemp


frames = 16;
savfile = "sample_rate_FP.gif"
duration = 1

files = [mkstemp(suffix='.png') for i in range(frames)]

import numpy
from numpy import loadtxt
y_true = loadtxt("y_test.csv", delimiter=',')[0]
t_true = loadtxt("t_test.csv", delimiter=',')[0]
timers = [2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
snrs = [10.3950, 11.8899, 11.8448, 8.6821, 10.0200, 11.2186, 12.0897, 10.2673, \
        10.0235, 12.0209, 10.7480, 11.1486, 9.9606, 11.4403, 11.2550, 1.742]
for i in range(frames):
    y_pred = loadtxt("./input_rate_outputs/" + str(timers[i]) + "ms.csv", delimiter=',')
    
    plt.figure(figsize=(6,3))
    plt.title("LSTM prediction of pin location")
    plt.plot(t_true, y_pred[:-1], label = "predicted pin location")
    plt.plot(t_true, y_true, label = "actual pin location",alpha=.8)
    plt.xlabel("time")
    plt.ylabel("y out")
    plt.ylim((-1, 3))
    plt.legend(loc=1)
    plt.text(1,2.6, str(timers[i]) + " ms")
    plt.text(15, 2.6, "SNR = " + str(snrs[i]))
    plt.tight_layout() 
    plt.savefig(files[i][1], dpi=800)


with imageio.get_writer(savfile, mode='I', duration=duration) as writer:
    for file in files:
        print(file[1])
        image = imageio.imread(file[1])
        writer.append_data(image)
    writer.close()