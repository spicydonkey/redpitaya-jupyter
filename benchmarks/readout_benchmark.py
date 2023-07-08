
from utils import timer

import numpy as np

import zmq

import zmq
import numpy as np

def request_zmq(socket):
    socket.send(b"1")
    message = socket.recv(copy=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, help="ip address of the server", default="localhost")
    parser.add_argument("-p", "--port", type=str, help="port of the server", default="5555")
    parser.add_argument("-m", "--mode", type=str, help="mode of the server", 
                        default="protobuf", choices=["protobuf", "raw", ])
    # parser.add_argument("-v", "--verbosity", action="count", default=0,
    #                     help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")

    n_iters = 30  # number of iterations to run

    filename = "readout_results.csv"
    with open(filename, mode="a", encoding="utf-8") as fid:

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{args.ip}:{args.port}")

        for _ in range(n_iters):
            # Set up ZMQ client
            _fid = None  # fid
            with timer(name=args.mode, fid=_fid):
                request_zmq(socket)
