"""Network communication between distributed inference nodes.

Handles activation transfer and health checks between PC and Mac nodes
over Thunderbolt 4 or 10GbE connections.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class NetworkStats:
    """Network performance measurements between nodes."""

    latency_ms: float
    bandwidth_gbps: float
    packet_loss_pct: float


def check_connectivity(host: str, port: int, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Check if a remote node is reachable.

    Args:
        host: Target hostname or IP.
        port: Target port number.
        timeout: Connection timeout in seconds.

    Returns:
        True if the node is reachable.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False
