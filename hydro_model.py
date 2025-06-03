"""Hydrologic model API consolidating submodules."""
from __future__ import annotations

from parameters import (
    GlacierParams,
    PermafrostParams,
    SnowParams,
    CanopyParams,
    SoilParams,
    WaterBodyParams,
    GroundwaterParams,
    WetlandParams,
    LandUseParams,
    HydroParams,
    HydroState,
)
from land_use import (
    default_landuse_lookup,
    is_water_landuse,
    encode_landuse_types,
    decode_landuse_types,
    get_landuse_encoding,
    _apply_parameter_overrides,
)
from physics_modules import *  # noqa: F401,F403
from update_modules import *  # noqa: F401,F403
from computation_modules import (
    build_cell_params,
    hydrologic_model,
    build_cell_params_from_landuse,
    validate_cell_params,
    _require_jax,
)

__all__ = [
    "SnowParams", "CanopyParams", "SoilParams", "GroundwaterParams",
    "GlacierParams", "PermafrostParams", "WetlandParams", "WaterBodyParams",
    "LandUseParams", "HydroParams", "HydroState",
    "build_cell_params", "hydrologic_model", "default_landuse_lookup",
    "is_water_landuse", "build_cell_params_from_landuse", "validate_cell_params",
    "get_landuse_encoding", "encode_landuse_types", "decode_landuse_types",
    "_apply_parameter_overrides",
]
