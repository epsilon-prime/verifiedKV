from .manager import OffloadOrchestrator 

__all__ = [
    "OffloadOrchestrator",
    "OffloadPackageConnector",
    "OffloadPackageScheduler",
    "OffloadPackageWorker",
    "OffloadPackageMetadata",
    "OffloadPackageWorkerMetadata",
]

def __getattr__(name: str):
    if name == "OffloadPackageConnector":
        from .connector import OffloadPackageConnector
        return OffloadPackageConnector
    if name == "OffloadPackageScheduler":
        from .scheduler_connector import OffloadPackageScheduler
        return OffloadPackageScheduler
    if name == "OffloadPackageWorker":
        from .worker_connector import OffloadPackageWorker
        return OffloadPackageWorker
    if name in ("OffloadPackageMetadata", "OffloadPackageWorkerMetadata"):
        from .metadata import OffloadPackageMetadata, OffloadPackageWorkerMetadata
        return {"OffloadPackageMetadata": OffloadPackageMetadata,
                "OffloadPackageWorkerMetadata": OffloadPackageWorkerMetadata}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")