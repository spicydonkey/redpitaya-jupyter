import time

import numpy as np

import signal_pb2
import zmq

from redpitaya.overlay.mercury import mercury as overlay

MOCK_DATA = np.arange(2**14)

class Readout:
    """Readout system.
    """    

    def __init__(self):
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

        # synchronization and trigger sources are the default,
        # which is the module itself
        self._osc0.reset()
        self._osc0.start()
        self._osc0.trigger()

    def configure(self, 
                  decimation,
                  trigger_pre,
                  trigger_post,
                  trig_src,
                  ) -> None:
        """Configure the readout system.
        """
        self._osc0.decimation = decimation
        self._osc0.trigger_pre = trigger_pre
        self._osc0.trigger_post = trigger_post
        self._osc0.trig_src = trig_src

    def get_buffer(self):
        """Get the raw buffer from the oscilloscope. 

        Returns:
            The raw buffer.
        """        
        return self._osc0.buffer

    def start_rawbuffer_server(self, socket: zmq.Socket):
        print("Rawbuffer server started")
        while True:
            message = socket.recv()
            print(f"Received request: {message}")
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
            buffer_copy = np.empty(len(self._osc0.buffer))
            buffer_copy[:] = buffer_np[:]
            socket.send(buffer_copy, copy=False)

            # bus error
            #buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
            #socket.send(buffer_np[:], copy=False)


    def start_protobuf_server(self, socket: zmq.Socket):
        print("Protobuf server started")
        while True:
            message = socket.recv()
            print(f"Received request: {message}")

            # Create Signal protobuf message
            signal_msg = signal_pb2.Signal()
            signal_msg.buffer.extend(self.get_buffer())
            # Send the reply back to the client
            socket.send(signal_msg.SerializeToString(), copy=False)
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="mode of the server", 
                        default="protobuf", choices=["protobuf", "raw", ])
    # parser.add_argument("-v", "--verbosity", action="count", default=0,
    #                     help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Server started")

    readout = Readout()
    if args.mode == "protobuf":
        readout.start_protobuf_server(socket)
    elif args.mode == "raw":
        readout.start_rawbuffer_server(socket)


if __name__ == "__main__":
    main()
    
