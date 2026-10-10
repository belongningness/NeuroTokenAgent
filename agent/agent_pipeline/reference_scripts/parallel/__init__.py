"""CUDA parallel orchestration for the NeuroToken demo pipeline.

The package exposes one complete Stage 0–6 entrypoint and one standalone
Stage 1/2 entrypoint. Canonical stage algorithms stay in the parent ``scripts``
directory; this package only adds resource planning, process ownership, strict
barriers, and deterministic result aggregation.
"""
