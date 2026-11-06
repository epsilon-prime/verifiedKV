# SPDX-License-Identifier: Apache-2.0
"""Metadata for OffloadPackageConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class OffloadPackageMetadata(KVConnectorMetadata):
    """Scheduler → Worker metadata for offload/staging operations."""

    # Store (offload GPU → CPU)
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Active decode requests this step
    request_ids: list[str] = field(default_factory=list)

    # Reverse mapping for completion tracking
    store_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class OffloadPackageWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker → Scheduler metadata for completed transfers."""

    completed_store_events: dict[int, int]

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadPackageWorkerMetadata)
        merged_store = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged_store[k] = merged_store.get(k, 0) + v
        return OffloadPackageWorkerMetadata(
            completed_store_events=merged_store,
        )