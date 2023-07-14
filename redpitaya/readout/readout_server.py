"""Module for the readout system server.
"""

import logging
import pickle
import time

import numpy as np
import zmq

from redpitaya.overlay.mercury import mercury as overlay
from . import InputRange, TriggerEdge, InputChannel


logger = logging.getLogger(__name__)

class ReadoutServer:
    """Readout system server.
    """
    def __init__(self,
                 port: str="5555",
                 input_ranges: tuple[InputRange, InputRange]=(InputRange.LV_1, InputRange.LV_1,),
                 signal_channel: InputChannel=InputChannel.CH1,
                 trigger_channel: InputChannel=InputChannel.CH2,
                 trigger_level: tuple[float, float]=(0.5, 0.6),
                 trigger_edge: TriggerEdge=TriggerEdge.NEG,
                 ) -> None:
        """Initialize the Readout system.

        Args:
            port: The port to listen on. Defaults to "5555".
            input_ranges: The input ranges for the two channels. Defaults to (InputRange.LV_1, InputRange.LV_1).
            signal_channel: The signal channel. Defaults to InputChannel.CH1.
            trigger_channel: The trigger channel. Defaults to InputChannel.CH2.
            trigger_level: The trigger level. Defaults to (0.5, 0.6).
            trigger_edge: The trigger edge type. Defaults to TriggerEdge.NEG.
        """
        self.signal_channel = signal_channel

        # Set up the FPGA
        self._fpga = overlay()

        # Instantiate oscilloscope devices
        self._oscilloscopes = [self._fpga.osc(ch, input_ranges[ch].value) for ch in range(2)]

        # Set up the trigger event
        self._oscilloscopes[trigger_channel].level = trigger_level
        self._oscilloscopes[trigger_channel].edge = trigger_edge.value
        for osc in self._oscilloscopes:
            osc.sync_src = self._fpga.sync_src[f"osc{signal_channel.value}"]
            osc.trig_src = self._fpga.trig_src[f"osc{trigger_channel.value}"]

        # Set up the ZMQ socket
        self._context = zmq.Context()
        self._socket: zmq.Socket[zmq.SocketType] = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{port}")
        logger.info("Server started: tcp://*:%s", port)

    def configure(self,
                  decimation: int,
                  trigger_pre: int,
                  trigger_post: int,
                  ) -> None:
        """Configure the readout system.
        """
        for osc in self._oscilloscopes:
            osc.decimation = decimation
            osc.trigger_pre = trigger_pre
            osc.trigger_post = trigger_post

        # # 1. Software trigger
        # self._oscs[self.SIGNAL_CHANNEL].trig_src = 0  # disable hardware trigger sources
        
        # # 2. Internal trigger
        # # trigger source is the level trigger from the same input
        # self._oscs[self.SIGNAL_CHANNEL].trig_src = self._fpga.trig_src["osc0"]
        # self._oscs[self.SIGNAL_CHANNEL].level = [0.4, 0.5]   # trigger level, or pair [neg, pos] for hysteresis (V)
        # self._oscs[self.SIGNAL_CHANNEL].edge  = 'pos'        # trigger edge type: ['neg', 'pos']

        # # 3. External trigger
        # # NOT DOCUMENTED!

    def capture(self, force: bool=False) -> tuple[int, float]:
        """Capture a shot.
        
        Args:
            force: If True, force a trigger. If False, waits for programmed trigger event.
            Defaults to False.
        
        Returns:
            (pointer, timestamp):
                pointer: The index of the trigger.
                timestamp: The timestamp of the trigger.
        """
        logger.info("Triggering...")
        self._oscilloscopes[self.signal_channel].reset()
        self._oscilloscopes[self.signal_channel].start()
        if force:
            self._oscilloscopes[self.signal_channel].trigger()
        while self._oscilloscopes[self.signal_channel].status_run():
            pass
        timestamp = time.time()
        pointer = int(self._oscilloscopes[self.signal_channel].pointer)
        logger.info('Triggered!')
        return (pointer, timestamp)

    def transfer_raw_buffer(self) -> None:
        """Transfer the raw buffer to the client.
        """

        # NOTE: this is overloading the meaning of transfer
        pointer, timestamp = self.capture(force=False)

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

        # bus error
        #buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
        #socket.send(buffer_np[:], copy=False)

        # # OK: 5.6 ms
        # buffer_np = np.ctypeslib.as_array(self._osc0.buffer)
        # buffer_copy = np.empty(len(self._osc0.buffer))  # dtype=np.int16 crashes
        # buffer_copy[:] = buffer_np[:]

        # OK: 5.6 ms
        buffer_np = np.ctypeslib.as_array(self._oscilloscopes[self.signal_channel].buffer)
        # FIXME: OPTIMISATION: we don't need to regenerate the buffer_copy every time, and we know the size.
        buffer_copy = np.empty(len(self._oscilloscopes[self.signal_channel].buffer))  # dtype=np.int16 crashes
        buffer_copy[:] = buffer_np[:]

        logger.debug("type(buffer_np[0]) = %s", type(buffer_np[0]))
        logger.debug("type(buffer_copy[0]) = %s", type(buffer_copy[0]))  # it's in np.float_
        logger.debug("buffer_copy[:20] = %s", buffer_copy[:20])
        logger.debug("pointer = %s", pointer)
        logger.debug("timestamp = %s", timestamp)

        self._socket.send_multipart(
            [
                buffer_copy,
                str(pointer).encode("utf-8"),
                str(timestamp).encode("utf-8"),
            ],
            copy=True,  # otherwise "Frame"
            )

    def run(self):
        """Run the server.
        """
        logger.info("Server running.")
        while True:
             #  Wait for next request from client
            message = self._socket.recv()
            logger.info("Received request: %s", message)

            # first byte of the message will determine the command
            command = message[0]

            # Handle commands
            if command == ord('g'):  # 'g' for 'get'
                self.transfer_raw_buffer()
            elif command == ord('c'):  # 'c' for 'configure'
                # rest_of_message should contain the configuration data
                message_body = message[1:]
                parameters = pickle.loads(message_body)
                logger.debug("parameters = %s", parameters)
                self.configure(**parameters)
                self._socket.send(b"Configuration complete")
            else:
                self._socket.send(b"Unknown command")

def main():
    """Start the server.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=str, help="port of the server",
                        default="5555")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()
    print(f"{args = }")

    # Set up logging based on the verbosity argument.
    # Verbosity levels:
    # - 0: WARNING
    # - 1: INFO
    # - 2+: DEBUG
    if args.verbosity >= 2:
        log_level = logging.DEBUG
    elif args.verbosity == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level)

    readout_server = ReadoutServer(args.port)
    readout_server.run()

if __name__ == "__main__":
    main()

