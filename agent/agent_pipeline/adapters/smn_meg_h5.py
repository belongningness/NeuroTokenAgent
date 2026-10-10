"""Source adapter for recording-level SMN MEG HDF5 files."""

from pathlib import Path

from agent_pipeline.adapters._timestamp_aligned_meg_h5 import TimestampAlignedMEGH5Adapter


class SmnMegH5Adapter(TimestampAlignedMEGH5Adapter):
    DEFAULT_H5_ROOT = Path("/home/luyilin/smn/smn_h5")
    SCHEMA_VERSION = "smn_meg_h5_v1"
    DATASET_ATTR = "SMN_MEG"
