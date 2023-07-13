import numpy as np

import zmq

def start_server():
    """Start the ZMQ server.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Server started")
    while True:
        message = socket.recv()
        # print(f"Received request: {message}")
        socket.send(b"Hello from server")

def main():
    start_server()

if __name__ == "__main__":
    main()
    