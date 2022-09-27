import numpy as np
import matplotlib.pyplot as plt
import json
"""
make a plot from the results
"""
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')


output_period_scheme = [1000, 750, 500, 400, 300, 200, 100] # in us
units_scheme = [40, 35, 30, 25, 20, 18, 15, 12, 10, 8]
cells_scheme = [1, 2, 3]

output_period = output_period_scheme[3]
units = units_scheme[6]
cells = cells_scheme[2]

string_name = "%dus%dcells%dunits"%(output_period, cells, units)

pred = np.load("./prediction results/" + string_name + ".npy").flatten()
dt = 400*10**(-6)
t = np.array([i*dt for i in range(pred.size)]);

f = open('data_6_with_FFT.json')
data = json.load(f)
f.close()
ref_t = np.array(data['measured_pin_location_tt'])
ref = np.array(data['measured_pin_location'])

ref = ref[ref_t > 1.5]
ref_t = ref_t[ref_t > 1.5] - 1.5
ref = np.interp(t, ref_t, ref)
plt.figure(figsize=(6.5, 2.5))
plt.plot(t, pred, label = "prediction", linewidth=1, color=cc[1], linestyle='--')
plt.plot(t, ref, label = "reference", color=cc[0], linewidth=1)
plt.xlabel("time (s)")
plt.ylabel("roller location (m)")
plt.ylim((0.045, .19))
plt.xlim((0,t[-1]))
plt.legend(loc=2, fancybox=False, framealpha=1)
plt.grid()
plt.tight_layout()
# plt.savefig("PredVsRef.svg")
plt.savefig("PredVsRef.png", dpi=800)