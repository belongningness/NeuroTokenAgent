"""Source adapter for recording-level NOD MEG HDF5 files."""

from pathlib import Path

from agent_pipeline.adapters._timestamp_aligned_meg_h5 import TimestampAlignedMEGH5Adapter


class NodMegH5Adapter(TimestampAlignedMEGH5Adapter):
    DEFAULT_H5_ROOT = Path("/home/luyilin/nod/nod_processed_h5")
    SCHEMA_VERSION = "nod-meg-h5-v2"
    DATASET_ATTR = "NOD_MEG_2025"
    LOCAL_PATH_METADATA_FIELDS = frozenset({
        "coordsystem_json",
        "meg_json",
        "events_json",
        "events_tsv_text",
    })
