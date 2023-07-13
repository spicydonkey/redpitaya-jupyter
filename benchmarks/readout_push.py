import zmq
import numpy as np
import ctypes


# Refactor this
from redpitaya.overlay.mercury import mercury as overlay
# Set up the FPGA
fpga = overlay()

# Set up the oscilloscope
osc0 = fpga.osc(0, 1.0)
# data rate decimation 
osc0.decimation = 4
# trigger timing [sample periods]
osc0.trigger_pre  = 0
osc0.trigger_post = osc0.buffer_size
# disable hardware trigger sources
osc0.trig_src = 0

# synchronization and trigger sources are the default,
# which is the module itself
osc0.reset()
osc0.start()
osc0.trigger()


# Create the ZMQ context
context = zmq.Context()

# Create a PUSH socket
socket = context.socket(zmq.PUSH)

# Bind the socket to a TCP address
socket.bind("tcp://*:5555")

# Initialize your data
buffer = np.arange(2**14, dtype=np.int16)

while True:
    # synchronization and trigger sources are the default,
    # which is the module itself
    osc0.reset()
    osc0.start()
    osc0.trigger()

    print("Sending data...")
    #socket.send(buffer, copy=True)

    buffer_np = np.ctypeslib.as_array(osc0.buffer)
    buffer_copy = np.empty(len(osc0.buffer))
    buffer_copy[:] = buffer_np[:]
    socket.send(buffer_copy, copy=False)
