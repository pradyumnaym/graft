"""Vendored OmniEraser inpainting pipeline (used only by `graft.scene`)."""
from .pipeline_flux_control_removal import FluxControlRemovalPipeline

__all__ = ["FluxControlRemovalPipeline"]
