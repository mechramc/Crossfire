"""Network communication between distributed inference nodes.

All supported interconnects run TCP/IP. USB4 (Thunderbolt IP bridge) is the
performance target; 5GbE is the practical fallback and the link in use
during development. RDMA is not supported in CROSSFIRE-X: composed
TriAttention + TurboQuant compression (6.8x KV reduction) makes the
USB4/5GbE bandwidth envelope sufficient, per crossfire_x_final.docx.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from enum import Enum

DEFAULT_TIMEOUT_SECONDS = 5


class InterconnectType(Enum):
    """TCP/IP interconnect between CROSSFIRE-X nodes.

    All values are TCP/IP transports; they differ only in physical-link
    bandwidth and latency. USB4 is the target for production; 5GbE is the
    documented fallback; 1GbE and WiFi exist for bring-up and discovery.
    """

    USB4 = "usb4"
    ETHERNET_5G = "5gbe"
    ETHERNET_1G = "1gbe"
    WIFI = "wifi"


# Expected latencies per interconnect (microseconds, TCP/IP round-trip)
LATENCY_USB4_US = 300.0
LATENCY_5GBE_US = 500.0
LATENCY_1GBE_US = 600.0
LATENCY_WIFI_US = 2000.0

# Bandwidth per interconnect (Gbps, effective TCP/IP throughput)
BANDWIDTH_USB4_GBPS = 40.0
BANDWIDTH_5GBE_GBPS = 5.0
BANDWIDTH_1GBE_GBPS = 1.0
BANDWIDTH_WIFI_GBPS = 1.0  # WiFi 7 typical effective


@dataclass(frozen=True)
class NetworkStats:
    """Network performance measurements between nodes."""

    latency_us: float
    bandwidth_gbps: float
    interconnect: InterconnectType

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_us / 1000.0


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
