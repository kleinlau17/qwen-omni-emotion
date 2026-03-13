"""UDP Sender Module

Sends commands to target device via UDP.
"""

import socket
from dataclasses import dataclass
from typing import Optional


@dataclass
class SendResult:
    """Send result"""
    success: bool
    bytes_sent: int = 0
    error: str = ""


class AnimationSender:
    """UDP Sender

    Args:
        host: Target IP address
        port: Target port
        timeout: socket timeout in seconds
    """

    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        """Establish socket connection."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(self.timeout)
        except socket.error as e:
            raise RuntimeError(f"Failed to create socket: {e}") from e

    def send_raw(self, json_str: str) -> SendResult:
        """Send raw JSON string."""
        if self._socket is None:
            return SendResult(success=False, error="Socket not initialized")

        try:
            data = json_str.encode("utf-8")
            self._socket.sendto(data, (self.host, self.port))
            return SendResult(success=True, bytes_sent=len(data))
        except socket.timeout:
            return SendResult(success=False, error="Send timeout")
        except socket.error as e:
            return SendResult(success=False, error=f"Socket error: {e}")
        except Exception as e:
            return SendResult(success=False, error=f"Unknown error: {e}")

    def close(self) -> None:
        """Close socket connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
