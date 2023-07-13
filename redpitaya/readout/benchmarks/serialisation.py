"""Benchmark data serialisation.
"""

import time

from utils import timer

import numpy as np

import signal_pb2

# suite of serialisation functions
def serialise_numpy_bytes(buffer: list[int]) -> None:
    # Convert list to numpy array and then to bytes
    buffer_np = np.array(buffer, dtype=np.int32)
    buffer_bytes = buffer_np.tobytes()

def serialise_protobuf(buffer: list[int]) -> None:
    signal_msg = signal_pb2.Signal()
    signal_msg.buffer.extend(buffer)
    signal_msg.SerializeToString()

def serialise_protobuf_bytes(buffer: list[int]) -> None:
    signal_msg = signal_pb2.SignalBytes()

    # Convert list to numpy array and then to bytes
    buffer_np = np.array(buffer, dtype=np.int32)
    buffer_bytes = buffer_np.tobytes()

    signal_msg.buffer = buffer_bytes
    # signal_msg.buffer.extend(buffer_bytes)
    signal_msg.SerializeToString()

def main():
    n_iters = 10  # number of iterations to run
    # generate random data
    size = 2**14
    buffer = np.random.randint(-2**31, 2**31, size=size, dtype=np.int32).tolist()

    filename = "serialisation_results.csv"
    with open(filename, mode="w", encoding="utf-8") as fid:
        for _ in range(n_iters):
            with timer(name="calibrate", fid=fid):
                time.sleep(0.1)

            with timer(name="numpy_bytes", fid=fid):
                serialise_numpy_bytes(buffer)

            with timer(name="protobuf", fid=fid):
                serialise_protobuf(buffer)

            with timer(name="protobuf_bytes", fid=fid):
                serialise_protobuf_bytes(buffer)

if __name__ == "__main__":
    main()
