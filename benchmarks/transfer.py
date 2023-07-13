
from utils import timer

import numpy as np

import zmq

import zmq
import numpy as np

def request_zmq(socket):
    socket.send(b"Request data")
    message = socket.recv()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, help="ip address of the server", default="localhost")
    parser.add_argument("-p", "--port", type=str, help="port of the server", default="5555")
    # parser.add_argument("-v", "--verbosity", action="count", default=0,
    #                     help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")


    n_iters = 30  # number of iterations to run

    filename = "transfer_results.csv"
    with open(filename, mode="w", encoding="utf-8") as fid:

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{args.ip}:{args.port}")

        for _ in range(n_iters):
            # Set up ZMQ client
            with timer(name="request_zmq", fid=fid):
            # with timer(name="request_zmq", fid=None):
                request_zmq(socket)
