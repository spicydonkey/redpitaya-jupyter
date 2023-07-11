"""Module for interfacing with the readout system.
"""

import logging

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
        raise NotImplementedError
        self._socket.send_pyobj({
            "command": "configure",
            "decimation": decimation,
            "trigger_pre": trigger_pre,
            "trigger_post": trigger_post,
            # "trig_src": trig_src,
        })
        message = self._socket.recv_pyobj()
        logger.info(f"Received reply: {message}")

    #def get_raw_buffer(self) -> tuple[npt.NDArray, int, float]:
    def get_raw_buffer(self) -> npt.NDArray[np.float_]:
        """Get the raw buffer from the oscilloscope.

        Returns:
            (raw_samples, trigger, timestamp):
                raw_samples: raw samples in buffer
                trigger: index to trigger.
                timestamp: timestamp of the trigger.
        """
        print("get_raw_buffer")
        self._socket.send(b"1")
        response = self._socket.recv(copy=False)
        raw_buffer = np.frombuffer(response, dtype=np.float_)
        return raw_buffer

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
    # readout_interface.configure(decimation=1, trigger_pre=0, trigger_post=2**14)

    while True:
        raw_buffer = readout_interface.get_raw_buffer()
        print(f"{raw_buffer[:20] = }")

if __name__ == "__main__":
    main()

