import numpy as np

import signal_pb2
import zmq

from redpitaya.overlay.mercury import mercury as overlay

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

    def start_server(self):
        """Start the server.
        """        
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

        print("Server started")
        while True:
            message = socket.recv()
            print(f"Received request: {message}")
            if message == 0:
                # Get the signal data
                # signal_data = np.array(self.get_buffer(), dtype=np.int32)
                # Send the reply back to the client
                socket.send_pyobj(self._osc0.buffer)

            # # Create Signal protobuf message
            # signal_msg = signal_pb2.Signal()
            # signal_msg.buffer.extend(self.get_buffer())
            # # Send the reply back to the client
            # socket.send(signal_msg.SerializeToString())

            # Get the signal data
            signal_data = np.array(self.get_buffer(), dtype=np.int32)

            # Send the reply back to the client
            socket.send_pyobj(signal_data)

def main():
    readout = Readout()
    readout.start_server()

if __name__ == "__main__":
    main()
    