"""Device-test EP coordination; never installed in the wheel."""

from datetime import timedelta
import os
from urllib.parse import urlsplit


def create_ep_group(rank, world_size, endpoint):
    from torch.distributed import TCPStore
    from deep_ep._store import StoreGroup

    address = urlsplit(endpoint)
    # Keep SHMEM, existing launcher coordination and EP-store ports separate.
    port = int(os.environ.get("MEGAMOE_TEST_EP_PORT", address.port + 2))
    if not 1024 <= port <= 65535 or port == address.port:
        raise ValueError("Set MEGAMOE_TEST_EP_PORT to a separate unprivileged TCP port")
    store = TCPStore(address.hostname, port, world_size, rank == 0, timeout=timedelta(seconds=300))
    return StoreGroup(store, rank, world_size, prefix="bf16-validation")
