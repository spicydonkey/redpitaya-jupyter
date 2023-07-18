"""Module for interfacing with the readout system.
"""

import asyncio
import logging
import pickle

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
import zmq.asyncio

logger = logging.getLogger(__name__)

class Result(BaseModel):
    """The Result returned by the Readout system.
    (For the lack of a better name...)
    """

    class Config:  # pylint: disable=too-few-public-methods,missing-class-docstring
        # validate field defaults
        validate_all = True
        # validate on assignment to field
        validate_assignment = True

    waveform: list[float] # npt.NDArray[np.float_] FIXME: numpy arrays please!
    """The captured waveform (V)."""

    timestamp: float
    """The time of acquisition (s)."""

    sampling_rate: float
    """The sampling rate (Samples/s)."""


class ReadoutInterface:
    """Readout system interface.
    """

    def __init__(self, ip: str="localhost", port: str="5555") -> None:  # pylint: disable=invalid-name
        """Initialize the interface.

        Args:
            ip: The IP address of the server. Defaults to "localhost".
            port: The port on the host. Defaults to "5555".
        """
        self._ip = ip
        self._port = port
        self._context = zmq.asyncio.Context()
        self._socket: zmq.asyncio.Socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self._ip}:{self._port}")
        logger.info("Connected to tcp://%s:%s", self._ip, self._port)

        self._decimation = 1  # FIXME: get from hardware.  ALWAYS!
        self.base_sampling_rate = 125000000.0
        self.num_samples = 2**14  # FIXME: get from hw

        # FIXME: It's currently quite hacky -- user needs to "configure" before doing anything.
        # would be good to make a "dummy".
        # self.configure()

    @property
    def scale(self) -> float:
        """The scale factor for the input range.

        input range to scale mapping:
        LV 1.0:    1.0/32767.0
        HV 20.0:   20 * 1.0/32767.0
        """
        return 3.051850947599719e-05

    @property
    def sampling_rate(self) -> float:
        """The sampling rate of the readout system.

        decimation to sampling frequency mapping:
            formula: 125e6/decimation
            where decimation should be a power of 2
        """
        return self.base_sampling_rate / self.decimation

    @property
    def decimation(self) -> int:
        """The decimation factor of the readout system.
        """
        return self._decimation

    def configure(self,
                  decimation: int=1,
                  trigger_pre: int=0,
                  trigger_post: int=2**14,
                  ) -> None:
        """Configure the readout system.

        Args:
            decimation: The decimation factor, as a power of 2. Defaults to 1.
            trigger_pre: The pre-trigger delay. Defaults to 0.
            trigger_post: The post-trigger delay. Defaults to 2**14.
        """
        self._decimation = decimation
        logger.info("Configuring readout system...")
        data = {
            "decimation": decimation,
            "trigger_pre": trigger_pre,
            "trigger_post": trigger_post,
        }
        logger.debug("data = %s", data)
        self._socket.send(b"c" + pickle.dumps(data))

        # wait for the response
        message = self._socket.recv()
        logger.debug("message = %s", message)

    async def get_raw_buffer(self) -> tuple[npt.NDArray[np.float_], int, float]:
        """Get the raw buffer from the oscilloscope.

        Returns:
            (raw_samples, pointer, timestamp):
                raw_samples: raw samples in buffer
                pointer: index to trigger.
                timestamp: timestamp of the trigger.
        """
        self._socket.send(b"g")
        raw_buffer_bytes, pointer_bytes, timestamp_bytes = await self._socket.recv_multipart(copy=True)  # otherwise returned as Frames
        raw_buffer = np.frombuffer(raw_buffer_bytes, dtype=np.float_)
        pointer = int(pointer_bytes.decode("utf-8"))
        timestamp = float(timestamp_bytes.decode("utf-8"))
        return raw_buffer, pointer, timestamp

    @staticmethod
    def raw_buffer_to_waveform(
        raw_buffer: npt.NDArray[np.float_],
        pointer: int,
        scale: float,
        ) -> npt.NDArray[np.float_]:
        """Convert the raw buffer to the captured waveform.

        Args:
            raw_buffer: The raw buffer waveform.
            pointer: The index to trigger.
            scale: The scale factor for the input range.

        Returns:
            The captured waveform (V).
        """
        return scale * np.roll(raw_buffer, -pointer)

    async def get_result(self) -> Result:
        """Get the result from the readout system.

        Returns:
            The Result.
        """
        raw_buffer, pointer, timestamp = await self.get_raw_buffer()
        return Result(
            waveform=self.raw_buffer_to_waveform(raw_buffer, pointer, self.scale).tolist(),
            timestamp=timestamp,
            sampling_rate=self.sampling_rate,
            )

    def start(self) -> None:
        """Start the readout system.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the readout system.
        """
        raise NotImplementedError

async def main():
    """Test of the readout interface.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, help="ip address of the server",
                        default="localhost")
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

    readout_interface = ReadoutInterface(ip=args.ip, port=args.port)
    readout_interface.configure(decimation=1, trigger_pre=0, trigger_post=2**14)

    start = time.time()
    while time.time() - start < 10:
        raw_buffer, pointer, timestamp = await readout_interface.get_raw_buffer()
        print(f"{raw_buffer[:20] = }")
        print(f"{pointer = }")
        print(f"{timestamp = }")

    print("-"*80)
    print("Pass")

if __name__ == "__main__":
    import argparse
    import time

    asyncio.run(main())
