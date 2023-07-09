
import argparse

import zmq
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ip", type=str, help="ip address of the server", default="localhost")
parser.add_argument("-p", "--port", type=str, help="port of the server", default="5555")
args = parser.parse_args()
print(f"{args = }")

# Create the ZMQ context
context = zmq.Context()

# Create a PULL socket
socket = context.socket(zmq.PULL)

# Connect to the server's address
socket.connect(f"tcp://{args.ip}:{args.port}")

while True:
    # Receive the data
    data = socket.recv()

    # Convert the received bytes back into a numpy array
    data = np.frombuffer(data, dtype=np.int16)

    # Now you can process your data
    # ...
    print("Processing data...")

