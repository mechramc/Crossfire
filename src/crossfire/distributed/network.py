"""Network communication between distributed inference nodes.

Handles RDMA over Thunderbolt 5 (3us latency via EXO) and fallback
TCP/IP connectivity checks between PC and Mac nodes.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from enum import Enum

DEFAULT_TIMEOUT_SECONDS = 5


class InterconnectType(Enum):
    """Network interconnect type between nodes."""

    THUNDERBOLT_5_RDMA = "tb5_rdma"
    THUNDERBOLT_4 = "tb4"
    ETHERNET_10G = "10gbe"
    ETHERNET_1G = "1gbe"


@dataclass(frozen=True)
class NetworkStats:
    """Network performance measurements between nodes."""

    latency_us: float
    bandwidth_gbps: float
    interconnect: InterconnectType
    rdma_active: bool = False

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_us / 1000.0


# Expected latencies per interconnect (microseconds)
LATENCY_TB5_RDMA_US = 3.0
LATENCY_TB4_TCP_US = 300.0
LATENCY_10GBE_US = 100.0

# Bandwidth per interconnect (Gbps)
BANDWIDTH_TB5_GBPS = 80.0
BANDWIDTH_TB4_GBPS = 40.0
BANDWIDTH_10GBE_GBPS = 10.0


def check_connectivity(host: str, port: int, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Check if a remote node is reachable via TCP.

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
