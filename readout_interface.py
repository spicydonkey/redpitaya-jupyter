"""Module for interfacing with the readout system.
"""

import logging
import pickle

import numpy as np
import numpy.typing as npt
import zmq


logger = logging.getLogger(__name__)

class ReadoutInterface:
    def __init__(self, ip: str, port: str="5555") -> None:
        self._ip = ip
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self._ip}:{self._port}")
        logger.info(f"Connected to tcp://{self._ip}:{self._port}")

    def configure(self,
                  decimation: int,
                  trigger_pre: int,
                  trigger_post: int,
                #   trig_src: str,
                  ) -> None:
        """Configure the readout system.
        """
        print("Configuring readout system...")
        data = {
            "decimation": decimation,
            "trigger_pre": trigger_pre,
            "trigger_post": trigger_post,
            # "trig_src": trig_src,
        }
        print(f"{data = }")
        self._socket.send(b"c" + pickle.dumps(data))

        # wait for the response
        message = self._socket.recv()
        print(message)

    def get_raw_buffer(self) -> tuple[npt.NDArray[np.float_], int, float]:
        """Get the raw buffer from the oscilloscope.

        Returns:
            (raw_samples, pointer, timestamp):
                raw_samples: raw samples in buffer
                pointer: index to trigger.
                timestamp: timestamp of the trigger.
        """
        print("get_raw_buffer")
        self._socket.send(b"g")
        # response = self._socket.recv(copy=False)
        # raw_buffer = np.frombuffer(response, dtype=np.float_)

        raw_buffer_bytes, pointer_bytes, timestamp_bytes = self._socket.recv_multipart(copy=True)  # otherwise returned as Frames
        raw_buffer = np.frombuffer(raw_buffer_bytes, dtype=np.float_)
        pointer = int(pointer_bytes.decode("utf-8"))
        timestamp = float(timestamp_bytes.decode("utf-8"))
        return raw_buffer, pointer, timestamp

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

def main(): 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, help="ip address of the server",
                        default="localhost")
    parser.add_argument("-p", "--port", type=str, help="port of the server",
                        default="5555")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")

    readout_interface = ReadoutInterface(ip=args.ip, port=args.port)

    readout_interface.configure(decimation=1, trigger_pre=0, trigger_post=2**14)

    while True:
        raw_buffer, pointer, timestamp = readout_interface.get_raw_buffer()
        print(f"{raw_buffer[:20] = }")
        print(f"{pointer = }")
        print(f"{timestamp = }")

if __name__ == "__main__":
    main()

