"""Module for the readout system server.
"""

import logging

import numpy as np
import zmq

from redpitaya.overlay.mercury import mercury as overlay


logger = logging.getLogger(__name__)

class ReadoutServer:
    """Readout system server.
    """
    def __init__(self, port: str="5555") -> None:
        """Initialize the server.

        Args:
            port: The port to listen on. Defaults to "5555".
        """
        # Set up the FPGA
        self._fpga = overlay()
        # Set up the oscilloscope
        self._osc0 = self._fpga.osc(0, 1.0)
        # data rate decimation 
        self._osc0.decimation = 4
        # trigger timing [sample periods]
        self._osc0.trigger_pre  = 0
        self._osc0.trigger_post = self._osc0.buffer_size
        # disable hardware trigger sources
        self._osc0.trig_src = 0

        # Set up the ZMQ socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{port}")
        print(f"Server started: tcp://*:{port}")

    def configure_capture(self,
                  decimation: int,
                  trigger_pre: int,
                  trigger_post: int,
                #   trig_src: str,
                  ) -> None:
        """Configure the readout system.
        """
        raise NotImplementedError
        self._osc0.decimation = decimation
        self._osc0.trigger_pre = trigger_pre
        self._osc0.trigger_post = trigger_post
        
        self._osc0.trig_src = 0  # disable hardware trigger sources
        # self._osc0.trig_src = self._fpga.trig_src[trig_src]
        # self._osc0.trig_src = self._fpga.trig_src["osc1"]
        # self._osc0.edge = "pos"
        # self._osc0.level = [0.4, 0.5]

    def trigger(self) -> None:
        """Force a trigger.
        """
        # synchronization and trigger sources are the default,
        # which is the module itself
        print("Triggering")
        self._osc0.reset()
        self._osc0.start()
        self._osc0.trigger()

    def run(self):
        """Run the server.
        """
        print("Server running.")
        while True:
            
            message = self._socket.recv()
            # FIXME: currently there is no handling of requests -- the server just 
            # sends the buffer back to the client.
            print(f"Received request: {message}")
            self.trigger()

            # Bus error
            #buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
            #socket.send(buffer_np, copy=True)  # error for even copy=False

            # OK: 3 ms
            #socket.send(np.arange(2**14), copy=False)

            # OK: 2.5 ms
            #socket.send(MOCK_DATA, copy=False)

            # Crashes
            #buffer_np = np.ctypeslib.as_array(self._osc0.buffer).copy()
            #socket.send(buffer_np, copy=True)

            # OK: 5.6 ms
            buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
            buffer_copy = np.empty(len(self._osc0.buffer))  # dtype=np.int16 crashes
            buffer_copy[:] = buffer_np[:]
            self._socket.send(buffer_copy, copy=False)

            print(f"{type(buffer_np[0]) = }")
            print(f"{type(buffer_copy[0]) = }")  # it's in np.float_
            print(f"{buffer_copy[:20] = }")

            # bus error
            #buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
            #socket.send(buffer_np[:], copy=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=str, help="port of the server",
                        default="5555")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")

    readout_server = ReadoutServer(args.port)
    readout_server.run()

if __name__ == "__main__":
    main()

