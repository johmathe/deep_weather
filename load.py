"""Load noaa data into hd5 X/Y dataset for training."""

# TODO(johmathe): make this less horrible.

import datetime
import h5py
import numpy as np
import os
import pygrib


ANL_DATA = './weather_data/'    # was : '/spindisk/weather_data/gfs_anl/'
ANL_LEVEL = 10
TRAIN_SAMPLES = 200
VAL_SAMPLES = 50

grib_list = []

d1 = datetime.date(2010, 1, 11)
d2 = datetime.date(2015, 12, 30)

delta = d2 - d1

for i in range(delta.days + 1):
    date = d1 + datetime.timedelta(days=i)
    for hour in [0, 6, 12, 18]:
        f = (
            ANL_DATA + 'gfsanl_4_%04d%02d%02d_%02d00_%03d.grb2'
            % (date.year, date.month, date.day, hour, 0))
        grib_list.append(f)

full_dataset = []
for x, y in zip(grib_list[:-2], grib_list[1:]):
    if os.path.exists(x) and os.path.exists(y):
        full_dataset.append((x, y))

print 'full dataset size : %i' % len(full_dataset)

training_size = int(len(full_dataset) * 0.9)
val_size = len(full_dataset) - training_size
train_paths = full_dataset[:training_size]
val_paths = full_dataset[training_size:]
print 'num of triaining samples %d' % len(train_paths)

# make sure we don't go off index
if val_size < VAL_SAMPLES:
    VAL_SAMPLES = val_size
if training_size < TRAIN_SAMPLES:
    TRAIN_SAMPLES = training_size

print 'VAL_SAMPLES: %d' % VAL_SAMPLES
print 'TRAIN SAMPLES : %d' % TRAIN_SAMPLES

X_train = np.zeros((TRAIN_SAMPLES, 1, 361, 720), dtype=np.float)
Y_train = np.zeros(TRAIN_SAMPLES, dtype=np.float)
for i, (x_path, y_path) in enumerate(train_paths[:TRAIN_SAMPLES-1]):
    print 'reading training file id %d' % i
    X_grb = pygrib.open(x_path)
    X_slice = X_grb.select(name='Temperature')[ANL_LEVEL]['values']
    X_train[i][0] = X_slice
    Y_grb = pygrib.open(y_path)
    Y_slice = Y_grb.select(name='Temperature')[ANL_LEVEL]['values']
    Y_train[i] = Y_slice[180, 360]

X_val = np.zeros((VAL_SAMPLES, 1, 361, 720), dtype=np.float)
Y_val = np.zeros(VAL_SAMPLES, dtype=np.float)
for i, (x_path, y_path) in enumerate(val_paths[:VAL_SAMPLES-1]):
    print 'reading validation file id %d' % i
    X_grb = pygrib.open(x_path)
    X_slice = X_grb.select(name='Temperature')[ANL_LEVEL]['values']
    X_val[i][0] = X_slice
    Y_grb = pygrib.open(y_path)
    Y_slice = Y_grb.select(name='Temperature')[ANL_LEVEL]['values']
    Y_val[i] = Y_slice[180, 360]


dest = './weather.hdf5'  # was '/home/johmathe/weather.hdf5'
full_path = os.getcwd() + dest[1:]
print 'writing dataset to disk at %s...' % full_path
with h5py.File(dest, 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('X_val', data=X_val)
    f.create_dataset('Y_train', data=Y_train)
    f.create_dataset('Y_val', data=Y_val)
print 'done.'
