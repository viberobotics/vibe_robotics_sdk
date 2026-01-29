import socket
import struct
import json
import numpy as np


class NumpySocket:
    def __init__(self, host="127.0.0.1", port=9000, is_sender=False):
        self.host = host
        self.port = port
        self.is_sender = is_sender
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if is_sender:
            self.sock.connect((host, port))
        else:
            self.sock.bind((host, port))
            self.sock.listen(1)
            print("Waiting for connection...")
            self.conn, _ = self.sock.accept()
            print("Connected!")

    # ---------- internal utils ----------
    def _sendall(self, data: bytes):
        self.sock.sendall(data)

    def _recvall(self, n: int):
        data = b""
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet
        return data

    # ---------- public API ----------
    def send(self, arr: np.ndarray):
        assert self.is_sender, "This instance is not a sender"

        header = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "nbytes": arr.nbytes,
        }
        header_bytes = json.dumps(header).encode("utf-8")

        # send header length (4 bytes)
        self._sendall(struct.pack("!I", len(header_bytes)))
        # send header
        self._sendall(header_bytes)
        # send raw array bytes
        self._sendall(arr.tobytes())

    def recv(self) -> np.ndarray:
        assert not self.is_sender, "This instance is not a receiver"

        # read header length
        header_len = struct.unpack("!I", self._recvall(4))[0]
        # read header
        header = json.loads(self._recvall(header_len).decode("utf-8"))

        shape = tuple(header["shape"])
        dtype = np.dtype(header["dtype"])
        nbytes = header["nbytes"]

        # read raw array bytes
        data = self._recvall(nbytes)
        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        return arr