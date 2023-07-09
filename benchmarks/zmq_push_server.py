import zmq
import numpy as np
import ctypes

# Create the ZMQ context
context = zmq.Context()

# Create a PUSH socket
socket = context.socket(zmq.PUSH)

# Bind the socket to a TCP address
socket.bind("tcp://*:5555")

# Initialize your data
buffer = (ctypes.c_int16 * 2**14)()

# Fill your buffer with data
# ...

while True:
    # Convert buffer to numpy array
    buffer_np = np.ctypeslib.as_array(buffer)
    
    # Create a copy of the array
    buffer_copy = np.empty(len(buffer))
    buffer_copy[:] = buffer_np[:]

    # Send the data
    print("Sending data...")
    socket.send(buffer_copy, copy=False)

