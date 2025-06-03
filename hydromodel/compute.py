"""High level computation utilities for running the hydrologic model."""
from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any, Optional, List

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from .parameters import HydroParams, HydroState
from .land_use import (
    default_landuse_lookup,
    encode_landuse_types,
    get_landuse_encoding,
    _apply_parameter_overrides,
)
from .updates import (
    update_snow_state_jax,
    update_canopy_state_jax_corrected,
    update_soil_state_jax,
    update_groundwater_state_jax,
    update_glacier_state_jax,
    update_permafrost_state_jax,
    update_wetland_state_jax,
    update_water_body_state_jax,
)


def _require_jax() -> None:
    if jnp is None:
        raise ImportError("JAX is required to use hydropy.model. Please install jax.")


def _validate_inputs(precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray) -> None:
    if not (precip.shape == evap.shape == temp.shape):
        raise ValueError(
            f"Input shapes must match. Got precip: {precip.shape}, "
            f"evap: {evap.shape}, temp: {temp.shape}"
        )
    if jnp.any(precip < 0):
        raise ValueError("Precipitation cannot be negative")
    if jnp.any(evap < 0):
        raise ValueError("Potential evapotranspiration cannot be negative")
    if jnp.any(temp < 173.15):
        raise ValueError("Temperature values seem unreasonably low (< -100°C)")
    if jnp.any(temp > 373.15):
        raise ValueError("Temperature values seem unreasonably high (> 100°C)")
    for name, arr in [("precip", precip), ("evap", evap), ("temp", temp)]:
        if jnp.any(jnp.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        if jnp.any(jnp.isinf(arr)):
            raise ValueError(f"{name} contains infinite values")


def build_cell_params(per_cell: Sequence[HydroParams], landuse_types: Sequence[str] | None = None) -> Tuple[HydroParams, Optional[Sequence[str]]]:
    _require_jax()
    if not per_cell:
        raise ValueError("per_cell sequence cannot be empty")
    for i, params in enumerate(per_cell):
        if not isinstance(params, HydroParams):
            raise ValueError(f"Element {i} is not a HydroParams instance: {type(params)}")

    landuse_sequence = None
    if landuse_types is not None:
        if len(landuse_types) != len(per_cell):
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) must match number of cells ({len(per_cell)})"
            )
        valid_landuse_types = set(default_landuse_lookup().keys())
        for i, lu_type in enumerate(landuse_types):
            if lu_type not in valid_landuse_types:
                raise ValueError(
                    f"Unknown landuse type '{lu_type}' at index {i}. Valid types: {sorted(valid_landuse_types)}"
                )
        landuse_sequence = list(landuse_types)

    try:
        stacked_params = tree_map(lambda *xs: jnp.stack(xs), *per_cell)
        n_cells = len(per_cell)
        if hasattr(stacked_params.snow, 'day_frac'):
            if stacked_params.snow.day_frac.shape != (n_cells,):
                raise ValueError(
                    f"Stacking failed: expected shape ({n_cells},), got {stacked_params.snow.day_frac.shape}"
                )
        return stacked_params, landuse_sequence
    except Exception as e:
        raise ValueError(f"Failed to stack cell parameters: {str(e)}") from e


def build_cell_params_from_landuse(landuse_types: Sequence[str], snow_params=None, custom_overrides: Dict[str, Dict[str, Any]] | None = None) -> Tuple[HydroParams, Sequence[str]]:
    _require_jax()
    if not landuse_types:
        raise ValueError("landuse_types sequence cannot be empty")
    lookup = default_landuse_lookup()
    if snow_params is None:
        from .parameters import SnowParams
        snow_params = SnowParams(day_frac=0.35)
    per_cell_params = []
    for i, lu_type in enumerate(landuse_types):
        if lu_type not in lookup:
            valid_types = sorted(lookup.keys())
            raise ValueError(
                f"Unknown landuse type '{lu_type}' at index {i}. Valid types: {valid_types}"
            )
        landuse_params = lookup[lu_type]
        if custom_overrides and lu_type in custom_overrides:
            overrides = custom_overrides[lu_type]
            landuse_params = _apply_parameter_overrides(landuse_params, overrides, lu_type)
        hydro_params = HydroParams(snow=snow_params, landuse=landuse_params)
        per_cell_params.append(hydro_params)
    return build_cell_params(per_cell_params, landuse_types)


def _single_cell_model_fixed(precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams, landuse_code: int = -1) -> jnp.ndarray:
    _require_jax()
    lu = params.landuse

    def step(state: HydroState, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        P, E, T = inputs
        is_watr = jnp.equal(landuse_code, 8)
        is_lake = jnp.equal(landuse_code, 9)
        is_water_body = jnp.logical_or(is_watr, is_lake)

        def water_body_branch():
            state_new, wb = update_water_body_state_jax(state, P, E, lu.water_body)
            runoff = wb["R_water"]
            return state_new, runoff

        wetland_active = lu.wetland.f_wet > 0.0

        def wetland_branch():
            state_new, wet = update_wetland_state_jax(state, P, T, E, lu.wetland)
            runoff = wet["R_sw"] + wet["R_gw"]
            return state_new, runoff

        glacier_active = lu.glacier.DDF_glac > 0.0

        def glacier_branch():
            state_new, glac = update_glacier_state_jax(state, P, T, E, lu.glacier)
            runoff = glac["R_glac"]
            return state_new, runoff

        permafrost_active = lu.permafrost.max_active_depth > 0.0

        def permafrost_branch():
            state_new, perm = update_permafrost_state_jax(state, P, T, E, lu.permafrost)
            runoff = perm["R_perm"]
            return state_new, runoff

        def standard_branch():
            state_snow, snow = update_snow_state_jax(state, P, T, params)
            state_canopy, canopy = update_canopy_state_jax_corrected(state_snow, P, snow, E, params)
            state_soil, soil = update_soil_state_jax(state_canopy, P, T, E, snow, canopy, params)
            state_gw, gw = update_groundwater_state_jax(state_soil, P, E, snow, soil, params)
            runoff = canopy["R_imp"] + soil["R_srf"] + gw["R_sw"] + gw["R_gw"]
            return state_gw, runoff

        return jax.lax.cond(
            is_water_body,
            water_body_branch,
            lambda: jax.lax.cond(
                wetland_active,
                wetland_branch,
                lambda: jax.lax.cond(
                    glacier_active,
                    glacier_branch,
                    lambda: jax.lax.cond(
                        permafrost_active,
                        permafrost_branch,
                        standard_branch,
                    )
                )
            )
        )

    init = HydroState(
        S_sn=0.0,
        S_snlq=0.0,
        S_skin=0.0,
        S_can=0.0,
        S_so=0.0,
        S_sw=0.0,
        groundwater=0.0,
        S_glac=0.0,
        S_al=0.0,
    )

    _, runoff_ts = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff_ts


def hydrologic_model(precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams, landuse_types: Sequence[str] | None = None) -> jnp.ndarray:
    _require_jax()
    _validate_inputs(precip, evap, temp)
    n_cells = precip.shape[1]
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) must match number of cells ({n_cells})"
            )
        landuse_codes = encode_landuse_types(landuse_types)

        def single_cell_model_with_landuse(precip_cell, evap_cell, temp_cell, params_cell, landuse_code):
            return _single_cell_model_fixed(precip_cell, evap_cell, temp_cell, params_cell, landuse_code)

        return jax.vmap(single_cell_model_with_landuse, in_axes=(1, 1, 1, 0, 0), out_axes=1)(precip, evap, temp, params, landuse_codes)
    else:
        return jax.vmap(_single_cell_model_fixed, in_axes=(1, 1, 1, 0), out_axes=1)(precip, evap, temp, params)


def validate_cell_params(params: HydroParams, landuse_types: Sequence[str] | None = None) -> None:
    def get_shape(x):
        return getattr(x, 'shape', (1,))[0] if hasattr(x, 'shape') else 1

    n_cells = get_shape(params.snow.day_frac)

    def check_shapes(obj, path=""):
        if hasattr(obj, '_fields'):
            for field_name in obj._fields:
                field_value = getattr(obj, field_name)
                check_shapes(field_value, f"{path}.{field_name}")
        elif hasattr(obj, 'shape'):
            if obj.shape[0] != n_cells:
                raise ValueError(
                    f"Parameter {path} has shape {obj.shape}, expected first dimension to be {n_cells}"
                )
        elif isinstance(obj, (int, float)):
            pass
        else:
            raise ValueError(f"Unexpected parameter type at {path}: {type(obj)}")

    check_shapes(params, "params")
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) doesn't match parameter arrays ({n_cells})"
            )
    if jnp.any(params.snow.day_frac < 0) or jnp.any(params.snow.day_frac > 1):
        raise ValueError("Snow day_frac must be between 0 and 1")
    if jnp.any(params.landuse.imperv_frac < 0) or jnp.any(params.landuse.imperv_frac > 1):
        raise ValueError("Impervious fraction must be between 0 and 1")
