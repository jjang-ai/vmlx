# SPDX-License-Identifier: Apache-2.0
"""Node discovery via Bonjour/mDNS and manual registration.

Advertises this node as a vMLX compute worker and discovers other nodes
on all available network interfaces (Thunderbolt bridge, Ethernet, WiFi,
Tailscale). Auto-detects the fastest available link between nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_vmlx-worker._tcp."
SERVICE_DOMAIN = "local."
DEFAULT_WORKER_PORT = 9100


class LinkType(Enum):
    """Network link type, ordered by expected performance."""
    THUNDERBOLT = "thunderbolt"
    ETHERNET_10G = "ethernet_10g"
    ETHERNET_1G = "ethernet_1g"
    TAILSCALE = "tailscale"
    WIFI = "wifi"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class NodeInfo:
    """Discovered or manually registered compute node."""
    node_id: str
    hostname: str
    address: str
    port: int = DEFAULT_WORKER_PORT
    chip: str = ""
    ram_gb: int = 0
    gpu_cores: int = 0
    available_gb: float = 0.0
    status: str = "discovered"
    link_type: LinkType = LinkType.UNKNOWN
    measured_bandwidth_mbps: float = 0.0
    measured_latency_ms: float = 0.0
    mlx_version: str = ""
    vmlx_version: str = ""
    assigned_layers: Optional[tuple] = None
    last_seen: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_seen > 30.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "address": self.address,
            "port": self.port,
            "chip": self.chip,
            "ram_gb": self.ram_gb,
            "gpu_cores": self.gpu_cores,
            "available_gb": self.available_gb,
            "status": self.status,
            "link_type": self.link_type.value,
            "measured_bandwidth_mbps": self.measured_bandwidth_mbps,
            "measured_latency_ms": self.measured_latency_ms,
            "mlx_version": self.mlx_version,
            "vmlx_version": self.vmlx_version,
            "assigned_layers": list(self.assigned_layers) if self.assigned_layers else None,
        }


def get_local_node_info() -> NodeInfo:
    """Build NodeInfo for this machine."""
    hostname = socket.gethostname().replace(".local", "")
    chip = _detect_chip()
    ram_gb = _detect_ram_gb()

    import mlx.core as mx
    mlx_ver = getattr(mx, "__version__", "unknown")

    try:
        from vmlx_engine import __version__ as vmlx_ver
    except ImportError:
        vmlx_ver = "unknown"

    return NodeInfo(
        node_id=f"{hostname}-{socket.getfqdn()}",
        hostname=hostname,
        address="0.0.0.0",
        port=DEFAULT_WORKER_PORT,
        chip=chip,
        ram_gb=ram_gb,
        gpu_cores=_detect_gpu_cores(),
        available_gb=_detect_available_gb(),
        status="idle",
        mlx_version=mlx_ver,
        vmlx_version=vmlx_ver,
    )


def detect_link_type(interface: str) -> LinkType:
    """Classify a network interface by link type.

    Uses four independent checks and combines them:
      1. Interface name substring match (bridge*, utun*, en*)
      2. Thunderbolt presence via system_profiler SPThunderboltDataType
      3. Wi-Fi check via networksetup -listallhardwareports
      4. Link speed via ifconfig media (10Gbase / 1000base / ...)

    Thunderbolt wins if ANY of these report a Thunderbolt link that
    covers `interface`. This is more robust than name matching alone
    because macOS names Thunderbolt bridges `bridge0`, `bridge1`,
    `en7`, etc. depending on how many Thunderbolt devices are
    connected and in what order.
    """
    iface = interface.lower()

    # Method 1: interface name heuristic
    if "thunderbolt" in iface:
        return LinkType.THUNDERBOLT
    if iface.startswith("utun"):
        return LinkType.TAILSCALE

    # Method 2: system_profiler SPThunderboltDataType — authoritative
    # source for Thunderbolt topology. Checks whether ANY connected TB
    # device has an active bridge, then classifies bridge* as TB.
    if iface.startswith("bridge"):
        if _thunderbolt_has_active_bridge():
            return LinkType.THUNDERBOLT
        # `bridge0` on macOS is also used for AP mode / internet sharing.
        # Without a TB peer it's not a TB link — fall through.

    # en* interfaces — could be Ethernet or Wi-Fi, and could ALSO be
    # a Thunderbolt Ethernet adapter on TB5 (which reports as en0 or
    # en1 in some configurations).
    if iface.startswith("en"):
        if _is_wifi_interface(interface):
            return LinkType.WIFI

        # Method 3: is this en* interface actually a Thunderbolt Ethernet
        # bridge on TB5? Check system_profiler for a "Thunderbolt Network"
        # device whose BSD name matches this interface.
        if _thunderbolt_owns_interface(interface):
            return LinkType.THUNDERBOLT

        # Method 4: link speed from ifconfig media
        speed = _get_link_speed_mbps(interface)
        if speed >= 40000:
            # TB4 = 40 Gbps, TB5 = 80/120 Gbps — ifconfig may report
            # these as "40Gbase" / "80Gbase". Classify as Thunderbolt
            # even if the interface name didn't advertise it.
            return LinkType.THUNDERBOLT
        if speed >= 5000:
            return LinkType.ETHERNET_10G
        return LinkType.ETHERNET_1G
    return LinkType.UNKNOWN


def _thunderbolt_has_active_bridge() -> bool:
    """Return True if system_profiler reports at least one connected
    Thunderbolt device with a networking bridge. This is the
    authoritative way to detect a live Thunderbolt link between two
    Macs — plugging a TB cable without an active peer reports the
    port but no bridge, while an active peer shows a `Link Status:
    0x1` or similar active state.
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPThunderboltDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(result.stdout)
        items = data.get("SPThunderboltDataType", [])
        for item in items:
            # Each TB port is an entry. A port with a peer will have
            # connected devices listed under "_items" and a
            # "device_name_key" showing the remote Mac's name.
            sub = item.get("_items", [])
            if sub:
                return True
            if item.get("device_name_key"):
                return True
    except Exception:
        pass
    return False


def _thunderbolt_owns_interface(interface: str) -> bool:
    """Return True if the given BSD interface name is associated with
    a Thunderbolt Networking device (macOS's "Thunderbolt Bridge" or
    "Thunderbolt Ethernet Slot" presented as en*)."""
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if f"Device: {interface}" in line and i > 0:
                prev = lines[i - 1].lower()
                if "thunderbolt" in prev:
                    return True
    except Exception:
        pass
    return False


def detect_best_route(target_ip: str) -> LinkType:
    """Detect the link type for reaching a specific IP address.

    Uses `route -n get <ip>` to find the outgoing interface, then
    classifies it. If the route lookup fails (e.g., no route), returns
    UNKNOWN so the caller can decide whether to retry or reject.
    """
    try:
        result = subprocess.run(
            ["route", "-n", "get", target_ip],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "interface:" in line:
                iface = line.split("interface:")[-1].strip()
                return detect_link_type(iface)
    except Exception:
        pass
    return LinkType.UNKNOWN


async def measure_bandwidth(address: str, port: int, payload_kb: int = 1024) -> tuple:
    """Measure bandwidth and latency to a remote node.

    Returns (bandwidth_mbps, latency_ms).
    """
    try:
        t0 = time.perf_counter()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(address, port), timeout=5.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        payload = b"\x00" * (payload_kb * 1024)
        header = json.dumps({"type": "bandwidth_probe", "size": len(payload)}).encode()
        writer.write(len(header).to_bytes(4, "big") + header + payload)
        await writer.drain()

        t1 = time.perf_counter()
        resp_hdr_len = int.from_bytes(await reader.readexactly(4), "big")
        await reader.readexactly(resp_hdr_len)
        elapsed = time.perf_counter() - t1

        bandwidth_mbps = (len(payload) * 8 / 1e6) / elapsed if elapsed > 0 else 0

        writer.close()
        await writer.wait_closed()
        return bandwidth_mbps, latency_ms
    except Exception as e:
        logger.warning("Bandwidth probe to %s:%d failed: %s", address, port, e)
        return 0.0, 999.0


# ---------------------------------------------------------------------------
# Bonjour / mDNS
# ---------------------------------------------------------------------------

class BonjourAdvertiser:
    """Advertise this node as a vMLX compute worker via mDNS."""

    def __init__(self, node_info: NodeInfo):
        self.node_info = node_info
        self._proc = None

    async def start(self):
        try:
            from zeroconf import Zeroconf, ServiceInfo
            self._zc = Zeroconf()
            txt = {
                "version": self.node_info.vmlx_version,
                "chip": self.node_info.chip,
                "ram_gb": str(self.node_info.ram_gb),
                "gpu_cores": str(self.node_info.gpu_cores),
                "available_gb": f"{self.node_info.available_gb:.0f}",
                "status": self.node_info.status,
            }
            self._info = ServiceInfo(
                SERVICE_TYPE,
                f"{self.node_info.hostname}.{SERVICE_TYPE}",
                port=self.node_info.port,
                properties=txt,
                server=f"{self.node_info.hostname}.local.",
            )
            self._zc.register_service(self._info)
            logger.info("Bonjour: advertising %s on port %d", self.node_info.hostname, self.node_info.port)
        except ImportError:
            logger.info("zeroconf not installed — using dns-sd CLI for Bonjour")
            await self._start_dns_sd()

    async def _start_dns_sd(self):
        txt_parts = [
            f"version={self.node_info.vmlx_version}",
            f"chip={self.node_info.chip}",
            f"ram_gb={self.node_info.ram_gb}",
            f"gpu_cores={self.node_info.gpu_cores}",
        ]
        cmd = [
            "dns-sd", "-R",
            self.node_info.hostname,
            SERVICE_TYPE,
            SERVICE_DOMAIN,
            str(self.node_info.port),
        ] + txt_parts
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        logger.info("dns-sd: advertising %s on port %d", self.node_info.hostname, self.node_info.port)

    async def stop(self):
        if hasattr(self, "_zc"):
            self._zc.unregister_service(self._info)
            self._zc.close()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()


class BonjourScanner:
    """Scan for vMLX compute workers on the network."""

    def __init__(self, on_found: Optional[Callable[[NodeInfo], None]] = None):
        self.nodes: Dict[str, NodeInfo] = {}
        self.on_found = on_found

    async def scan(self, timeout: float = 5.0) -> List[NodeInfo]:
        try:
            return await self._scan_zeroconf(timeout)
        except ImportError:
            return await self._scan_dns_sd(timeout)

    async def _scan_zeroconf(self, timeout: float) -> List[NodeInfo]:
        from zeroconf import Zeroconf, ServiceBrowser

        zc = Zeroconf()
        found = []

        class Listener:
            def add_service(self_, zc_inst, type_, name):
                info = zc_inst.get_service_info(type_, name)
                if info:
                    node = NodeInfo(
                        node_id=name,
                        hostname=info.server.rstrip(".").replace(".local", ""),
                        address=str(info.parsed_addresses()[0]) if info.parsed_addresses() else "",
                        port=info.port,
                        chip=info.properties.get(b"chip", b"").decode(),
                        ram_gb=int(info.properties.get(b"ram_gb", b"0")),
                        gpu_cores=int(info.properties.get(b"gpu_cores", b"0")),
                        available_gb=float(info.properties.get(b"available_gb", b"0")),
                        vmlx_version=info.properties.get(b"version", b"").decode(),
                        status="discovered",
                    )
                    found.append(node)

            def remove_service(self_, zc_inst, type_, name):
                pass

            def update_service(self_, zc_inst, type_, name):
                pass

        browser = ServiceBrowser(zc, SERVICE_TYPE, Listener())
        await asyncio.sleep(timeout)
        browser.cancel()
        zc.close()

        for node in found:
            node.link_type = detect_best_route(node.address)
            self.nodes[node.node_id] = node
            if self.on_found:
                self.on_found(node)
        return found

    async def _scan_dns_sd(self, timeout: float) -> List[NodeInfo]:
        proc = await asyncio.create_subprocess_exec(
            "dns-sd", "-B", SERVICE_TYPE, SERVICE_DOMAIN,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(timeout)
        proc.terminate()
        stdout, _ = await proc.communicate()

        found = []
        for line in stdout.decode().splitlines():
            if SERVICE_TYPE in line and "Add" in line:
                parts = line.strip().split()
                if len(parts) >= 7:
                    name = parts[-1]
                    node = await self._resolve_dns_sd(name)
                    if node:
                        found.append(node)
                        self.nodes[node.node_id] = node
                        if self.on_found:
                            self.on_found(node)
        return found

    async def _resolve_dns_sd(self, name: str) -> Optional[NodeInfo]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "dns-sd", "-L", name, SERVICE_TYPE, SERVICE_DOMAIN,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.sleep(2.0)
            proc.terminate()
            stdout, _ = await proc.communicate()
            for line in stdout.decode().splitlines():
                if "can be reached at" in line:
                    addr_part = line.split("at")[-1].strip()
                    host, port_str = addr_part.rsplit(":", 1)
                    host = host.rstrip(".")
                    port = int(port_str)
                    try:
                        ip = socket.gethostbyname(host)
                    except socket.gaierror:
                        ip = host
                    return NodeInfo(
                        node_id=name, hostname=name, address=ip, port=port,
                        status="discovered", link_type=detect_best_route(ip),
                    )
        except Exception as e:
            logger.warning("Failed to resolve Bonjour service %s: %s", name, e)
        return None


# ---------------------------------------------------------------------------
# Hardware detection (macOS)
# ---------------------------------------------------------------------------

def _detect_chip() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def _detect_ram_gb() -> int:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip()) // (1024 ** 3)
    except Exception:
        return 0


def _detect_gpu_cores() -> int:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        for gpu in data.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores", "")
            if cores:
                return int(str(cores).replace(",", ""))
    except Exception:
        pass
    return 0


def _detect_available_gb() -> float:
    try:
        import mlx.core as mx
        info = mx.device_info()
        max_mem = info.get("memory_size", 0)
        return max(0, max_mem / (1024 ** 3) - 4)
    except Exception:
        return max(0, _detect_ram_gb() - 4)


def _is_wifi_interface(interface: str) -> bool:
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if f"Device: {interface}" in line and i > 0:
                if "Wi-Fi" in lines[i - 1] or "AirPort" in lines[i - 1]:
                    return True
    except Exception:
        pass
    return False


def _get_link_speed_mbps(interface: str) -> int:
    try:
        result = subprocess.run(
            ["ifconfig", interface],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "media:" in line.lower():
                if "10Gbase" in line:
                    return 10000
                if "1000base" in line:
                    return 1000
                if "100base" in line:
                    return 100
    except Exception:
        pass
    return 1000


# ---------------------------------------------------------------------------
# Multi-method discovery cascade
# ---------------------------------------------------------------------------

UDP_DISCOVERY_PORT = 9101
UDP_MAGIC = b"VMLX_DISCOVER_v1"


class UDPBroadcaster:
    """Broadcast UDP discovery packets on all interfaces."""

    def __init__(self, node_info: NodeInfo):
        self.node_info = node_info
        self._sock = None

    async def broadcast(self):
        """Send a discovery broadcast."""
        payload = json.dumps({
            "magic": UDP_MAGIC.decode(),
            "node_id": self.node_info.node_id,
            "hostname": self.node_info.hostname,
            "port": self.node_info.port,
            "chip": self.node_info.chip,
            "ram_gb": self.node_info.ram_gb,
            "gpu_cores": self.node_info.gpu_cores,
            "available_gb": self.node_info.available_gb,
            "vmlx_version": self.node_info.vmlx_version,
        }).encode()

        import socket as _socket
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_BROADCAST, 1)
        sock.settimeout(1)
        try:
            sock.sendto(payload, ("<broadcast>", UDP_DISCOVERY_PORT))
        finally:
            sock.close()

    async def listen(self, timeout: float = 5.0, on_found: Optional[Callable] = None) -> List[NodeInfo]:
        """Listen for UDP discovery broadcasts."""
        import socket as _socket
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.bind(("", UDP_DISCOVERY_PORT))
        sock.settimeout(0.5)

        found = []
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                data, addr = sock.recvfrom(4096)
                info = json.loads(data)
                if info.get("magic") != UDP_MAGIC.decode():
                    continue
                node = NodeInfo(
                    node_id=info["node_id"],
                    hostname=info.get("hostname", addr[0]),
                    address=addr[0],
                    port=info.get("port", DEFAULT_WORKER_PORT),
                    chip=info.get("chip", ""),
                    ram_gb=info.get("ram_gb", 0),
                    gpu_cores=info.get("gpu_cores", 0),
                    available_gb=info.get("available_gb", 0),
                    vmlx_version=info.get("vmlx_version", ""),
                    status="discovered",
                    link_type=detect_best_route(addr[0]),
                )
                found.append(node)
                if on_found:
                    on_found(node)
            except TimeoutError:
                continue
            except Exception:
                continue
            await asyncio.sleep(0)
        sock.close()
        return found


async def probe_http_identity(address: str, port: int = DEFAULT_WORKER_PORT) -> Optional[NodeInfo]:
    """Probe a known IP for a vMLX worker via HTTP GET /node_id.

    Workers expose two listeners:
      - `port` → binary protocol (length-prefixed JSON + tensor payloads)
      - `port + 1` → HTTP identity endpoint (for this probe)

    This function takes the BINARY port (the one the caller will later
    use for WorkerConnection.connect) and internally probes the HTTP
    port at `port + 1`. The returned NodeInfo carries the binary port
    so downstream code doesn't need to know about the +1 offset.

    Before this fix both `_cached` (manual --worker-nodes) and the
    Tailscale peer probe were passing 9100 and hitting the binary
    server with a raw HTTP request — the binary server would try to
    parse 'GET ' as a 4-byte big-endian header length (1.1GB), fail
    with IncompleteReadError, and silently close the connection.
    Both discovery methods appeared broken because every probe
    returned None.
    """
    http_port = port + 1
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(address, http_port), timeout=3.0,
        )
        request = f"GET /node_id HTTP/1.0\r\nHost: {address}\r\n\r\n"
        writer.write(request.encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=3.0)
        writer.close()
        await writer.wait_closed()

        # Parse HTTP response body
        resp_str = response.decode()
        if "200" not in resp_str.split("\r\n")[0]:
            return None

        body = resp_str.split("\r\n\r\n", 1)[-1].strip()
        info = json.loads(body)

        # NodeInfo.port holds the BINARY port — WorkerConnection.connect
        # uses this value for the real inference TCP connection.
        return NodeInfo(
            node_id=info.get("node_id", f"{address}:{port}"),
            hostname=info.get("hostname", address),
            address=address,
            port=port,
            chip=info.get("chip", ""),
            ram_gb=info.get("ram_gb", 0),
            gpu_cores=info.get("gpu_cores", 0),
            available_gb=info.get("available_gb", 0),
            vmlx_version=info.get("vmlx_version", ""),
            status="discovered",
            link_type=detect_best_route(address),
        )
    except Exception:
        return None


async def discover_tailscale_peers(worker_port: int = DEFAULT_WORKER_PORT) -> List[NodeInfo]:
    """Discover vMLX workers on the Tailscale network.

    Uses `tailscale status --json` to list all peers, then probes
    each for a running vMLX worker.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "tailscale", "status", "--json",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        status = json.loads(stdout)
    except Exception:
        return []

    peers = status.get("Peer", {})
    found = []

    async def _probe(peer_id, peer_info):
        addrs = peer_info.get("TailscaleIPs", [])
        hostname = peer_info.get("HostName", "")
        for addr in addrs:
            if ":" in addr:  # skip IPv6
                continue
            node = await probe_http_identity(addr, worker_port)
            if node:
                node.link_type = LinkType.TAILSCALE
                node.hostname = hostname or node.hostname
                found.append(node)
                return

    tasks = []
    for pid, pinfo in peers.items():
        if pinfo.get("Online", False):
            tasks.append(_probe(pid, pinfo))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return found


async def verify_reachable(address: str, port: int, timeout: float = 3.0) -> bool:
    """Pre-flight TCP reachability check.

    Bonjour can return a stale or intentionally-announced address
    that isn't actually routable from the coordinator (worker moved
    networks, firewall, mDNS cache lag). This is a cheap connect-only
    test that confirms the binary port is answering BEFORE the
    coordinator tries to send a JOIN message and wait for a 10s
    connect timeout inside WorkerConnection.connect.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(address, port), timeout=timeout,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def discover_all(
    timeout: float = 5.0,
    cached_peers: Optional[List[dict]] = None,
    on_found: Optional[Callable[[NodeInfo, str], None]] = None,
    verify: bool = True,
) -> List[NodeInfo]:
    """Run ALL discovery methods in parallel and deduplicate results.

    Args:
        timeout: Max time for each method.
        cached_peers: Previously known nodes [{address, port}] to re-probe.
        on_found: Callback(node, method) called as nodes are found.

    Returns deduplicated list of discovered nodes.
    """
    all_nodes: Dict[str, tuple] = {}  # node_id → (NodeInfo, method)

    def _add(node: NodeInfo, method: str):
        if node.node_id not in all_nodes:
            all_nodes[node.node_id] = (node, method)
            if on_found:
                on_found(node, method)
        else:
            # Keep the one with better link type
            existing_node, existing_method = all_nodes[node.node_id]
            link_priority = {
                LinkType.THUNDERBOLT: 0, LinkType.ETHERNET_10G: 1,
                LinkType.ETHERNET_1G: 2, LinkType.WIFI: 3,
                LinkType.TAILSCALE: 4, LinkType.MANUAL: 5, LinkType.UNKNOWN: 6,
            }
            if link_priority.get(node.link_type, 6) < link_priority.get(existing_node.link_type, 6):
                all_nodes[node.node_id] = (node, method)

    async def _bonjour():
        scanner = BonjourScanner()
        nodes = await scanner.scan(timeout)
        for n in nodes:
            _add(n, "bonjour")

    async def _udp():
        broadcaster = UDPBroadcaster(get_local_node_info())
        await broadcaster.broadcast()
        nodes = await broadcaster.listen(timeout)
        for n in nodes:
            _add(n, "udp_broadcast")

    async def _tailscale():
        nodes = await discover_tailscale_peers()
        for n in nodes:
            _add(n, "tailscale")

    async def _cached():
        if not cached_peers:
            return
        for peer in cached_peers:
            node = await probe_http_identity(
                peer["address"], peer.get("port", DEFAULT_WORKER_PORT),
            )
            if node:
                _add(node, "cached")

    # Run all methods concurrently
    await asyncio.gather(
        _bonjour(), _udp(), _tailscale(), _cached(),
        return_exceptions=True,
    )

    raw_result = [(node, method) for node, method in all_nodes.values()]

    if verify and raw_result:
        # Pre-flight reachability check — drop any discovered node whose
        # advertised address doesn't accept TCP on the binary port. Catches
        # stale mDNS entries, firewall blocks, and NAT'd addresses.
        probes = [
            verify_reachable(node.address, node.port)
            for node, _ in raw_result
        ]
        reachable = await asyncio.gather(*probes, return_exceptions=True)
        verified: List[NodeInfo] = []
        for (node, method), ok in zip(raw_result, reachable):
            if ok is True:
                verified.append(node)
            else:
                logger.info(
                    "Discovery: dropping %s (%s) — %s:%d not reachable",
                    node.hostname, method, node.address, node.port,
                )
        result = verified
    else:
        result = [node for node, _ in raw_result]

    method_counts = {}
    for _, method in raw_result:
        method_counts[method] = method_counts.get(method, 0) + 1
    logger.info(
        "Discovery complete: %d raw / %d reachable (%s)",
        len(raw_result),
        len(result),
        ", ".join(f"{m}:{c}" for m, c in sorted(method_counts.items())),
    )
    return result
