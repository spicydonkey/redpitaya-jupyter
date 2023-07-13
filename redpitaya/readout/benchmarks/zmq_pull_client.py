import zmq
import numpy as np

# Create the ZMQ context
context = zmq.Context()

# Create a PULL socket
socket = context.socket(zmq.PULL)

# Connect to the server's address
socket.connect("tcp://localhost:5555")

while True:
    # Receive the data
    data = socket.recv()

    # Convert the received bytes back into a numpy array
    data = np.frombuffer(data, dtype=np.int16)

    # Now you can process your data
    # ...
    print("Processing data...")
