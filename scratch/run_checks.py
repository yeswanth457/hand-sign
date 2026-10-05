import numpy as np

s = np.load('models/debug_browser_no_sequence.npy')
print('debug_browser_no_sequence shape:', s.shape, 'mean Hy:', s[:,1].mean(), 'mean Ry:', s[:,5].mean())
