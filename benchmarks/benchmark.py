#!/usr/env/bin python3

import time

from redpitaya.overlay.mercury import mercury as overlay

fpga = overlay()
osc0 = fpga.osc(0, 1.0)

# data rate decimation 
osc0.decimation = 4
# trigger timing [sample periods]
buffer_size = osc0.buffer_size
osc0.trigger_pre  = 0
osc0.trigger_post = buffer_size

# disable hardware trigger sources
osc0.trig_src = 0

# synchronization and trigger sources are the default,
# which is the module itself
osc0.reset()
osc0.start()
osc0.trigger()
# wait for data
while (osc0.status_run()): 
    pass
print ('triggered')

time_start = time.perf_counter()
data = osc0.data(buffer_size)
time_end = time.perf_counter()
duration = time_end - time_start
print(f"osc.data: {duration = :.3g} s")
