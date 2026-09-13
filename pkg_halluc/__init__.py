"""pkg_halluc -- Package Hallucination Mitigation: GA / NPO (and their
no-tri-mask "plain" ablations) compared on a shared, unified evaluation
harness.

See README.md for the full pipeline walkthrough; `pkg_halluc.cli` for the
command-line entrypoint.
"""

__version__ = "0.1.0"

from .model_setup import build_model, download_base_model

__all__ = ["build_model", "download_base_model", "__version__"]
