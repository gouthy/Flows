"""State update helper functions wrapping the physics modules."""
from __future__ import annotations
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from .parameters import (
    HydroState,
    HydroParams,
    GlacierParams,
    PermafrostParams,
    WetlandParams,
    WaterBodyParams,
)
from .physics import (
    snow_module,
    skin_canopy_module_corrected,
    soil_module,
    groundwater_process,
    glacier_module,
    permafrost_module,
    wetland_module,
    water_body_module,
)


def update_snow_state_jax(state: HydroState, P: jnp.ndarray, T: jnp.ndarray, params: HydroParams) -> Tuple[HydroState, Dict[str, Any]]:
    snow = snow_module(P, T, params.snow, state.S_sn, state.S_snlq)
    state = state._replace(S_sn=snow["S_sn_new"], S_snlq=snow["S_snlq_new"])
    return state, snow


def update_canopy_state_jax_corrected(state: HydroState, P: jnp.ndarray, snow: dict, PET: jnp.ndarray, params: HydroParams) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    canopy = skin_canopy_module_corrected(P, snow["P_sn"], snow["R_sn"], PET, lu, state.S_skin, state.S_can)
    state = state._replace(
        S_skin=canopy["S_skin_new"],
        S_can=canopy["S_can_new"],
    )
    return state, canopy


def update_soil_state_jax(state: HydroState, P: jnp.ndarray, T: jnp.ndarray, PET: jnp.ndarray, snow: dict, canopy: dict, params: HydroParams) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    soil = soil_module(P, T, PET, snow, canopy, lu, state.S_so)
    state = state._replace(S_so=soil["S_so_new"])
    return state, soil


def update_groundwater_state_jax(state: HydroState, P: jnp.ndarray, PET: jnp.ndarray, snow: dict, soil: dict, params: HydroParams) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    gw = groundwater_process(P=P, snow=snow, soil=soil, PET=PET, params=lu.groundwater, S_sw=state.S_sw, S_gw=state.groundwater)
    state = state._replace(
        S_sw=gw["S_sw_new"],
        groundwater=gw["S_gw_new"],
    )
    return state, gw


def update_glacier_state_jax(state: HydroState, P: jnp.ndarray, T: jnp.ndarray, PET: jnp.ndarray, params: GlacierParams) -> Tuple[HydroState, Dict[str, Any]]:
    glac = glacier_module(P, T, PET, params, state.S_glac)
    new_state = state._replace(S_glac=glac["S_glac_new"])
    return new_state, glac


def update_permafrost_state_jax(state: HydroState, P: jnp.ndarray, T: jnp.ndarray, PET: jnp.ndarray, params: PermafrostParams) -> Tuple[HydroState, Dict[str, Any]]:
    perm = permafrost_module(P, T, PET, params, state.S_al)
    new_state = state._replace(S_al=perm["S_al_new"])
    return new_state, perm


def update_wetland_state_jax(state: HydroState, P: jnp.ndarray, T: jnp.ndarray, PET: jnp.ndarray, params: WetlandParams) -> Tuple[HydroState, Dict[str, Any]]:
    wet = wetland_module(P, PET, params, state.S_sw)
    new_state = state._replace(S_sw=wet["S_sw_new"])
    return new_state, wet


def update_water_body_state_jax(state: HydroState, P: jnp.ndarray, PET: jnp.ndarray, params: WaterBodyParams) -> Tuple[HydroState, Dict[str, Any]]:
    wb = water_body_module(P, PET, params, state.S_sw)
    new_state = state._replace(S_sw=wb["S_sw_new"])
    return new_state, wb
