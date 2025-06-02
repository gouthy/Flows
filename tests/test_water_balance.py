import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax.numpy as jnp
from hydro_model import (
    default_landuse_lookup,
    SnowParams,
    HydroParams,
    HydroState,
    update_snow_state_jax,
    update_canopy_state_jax_corrected,
    update_soil_state_jax,
    update_groundwater_state_jax,
    update_glacier_state_jax,
    update_permafrost_state_jax,
    update_wetland_state_jax,
    update_water_body_state_jax,
    get_landuse_encoding,
)


def evaluate_water_balance_errors_final(
    P: float,
    T_srf: float,
    PET: float,
    params: HydroParams,
    initial_state: HydroState,
    landuse_code: int = -1,
    tolerance: float = 1e-6,
):
    """Compute water balance errors for a single timestep."""
    results = {}
    lu = params.landuse
    current_state = initial_state

    is_watr = jnp.equal(landuse_code, 8)
    is_lake = jnp.equal(landuse_code, 9)
    is_water_body = jnp.logical_or(is_watr, is_lake)

    if lu.water_body.is_water_body or is_water_body:
        new_state, wb = update_water_body_state_jax(current_state, P, PET, lu.water_body)
        delta = new_state.S_sw - current_state.S_sw
        balance = P - wb["ET_water"] - wb["R_water"] - delta
        results["overall"] = {
            "water_balance_error": float(balance),
            "total_inputs": float(P),
            "total_outputs": float(wb["ET_water"] + wb["R_water"]),
            "total_storage_change": float(delta),
            "is_balanced": abs(float(balance)) < tolerance,
        }
        return results

    if lu.wetland.f_wet > 0.0:
        new_state, wet = update_wetland_state_jax(current_state, P, T_srf, PET, lu.wetland)
        delta = new_state.S_sw - current_state.S_sw
        balance = P * lu.wetland.f_wet - wet["E_wet"] - wet["R_sw"] - wet["R_gw"] - delta
        results["overall"] = {
            "water_balance_error": float(balance),
            "total_inputs": float(P * lu.wetland.f_wet),
            "total_outputs": float(wet["E_wet"] + wet["R_sw"] + wet["R_gw"]),
            "total_storage_change": float(delta),
            "is_balanced": abs(float(balance)) < tolerance,
        }
        return results

    if lu.glacier.DDF_glac > 0.0:
        new_state, glac = update_glacier_state_jax(current_state, P, T_srf, PET, lu.glacier)
        delta = new_state.S_glac - current_state.S_glac
        balance = P - glac["R_glac"] - glac["E_glac"] - delta
        results["overall"] = {
            "water_balance_error": float(balance),
            "total_inputs": float(P),
            "total_outputs": float(glac["R_glac"] + glac["E_glac"]),
            "total_storage_change": float(delta),
            "is_balanced": abs(float(balance)) < tolerance,
        }
        return results

    if lu.permafrost.max_active_depth > 0.0:
        new_state, perm = update_permafrost_state_jax(current_state, P, T_srf, PET, lu.permafrost)
        delta = new_state.S_al - current_state.S_al
        balance = P - perm["R_perm"] - perm["E_al"] - delta
        results["overall"] = {
            "water_balance_error": float(balance),
            "total_inputs": float(P),
            "total_outputs": float(perm["R_perm"] + perm["E_al"]),
            "total_storage_change": float(delta),
            "is_balanced": abs(float(balance)) < tolerance,
        }
        return results

    snow_state, snow = update_snow_state_jax(current_state, P, T_srf, params)
    canopy_state, canopy = update_canopy_state_jax_corrected(snow_state, P, snow, PET, params)
    soil_state, soil = update_soil_state_jax(canopy_state, P, T_srf, PET, snow, canopy, params)
    gw_state, gw = update_groundwater_state_jax(soil_state, P, PET, snow, soil, params)

    delta_total = (
        (snow_state.S_sn - current_state.S_sn)
        + (snow_state.S_snlq - current_state.S_snlq)
        + (canopy_state.S_skin - snow_state.S_skin)
        + (canopy_state.S_can - snow_state.S_can)
        + (soil_state.S_so - canopy_state.S_so)
        + (gw_state.S_sw - soil_state.S_sw)
        + (gw_state.groundwater - soil_state.groundwater)
    )

    system_et = (
        canopy["E_skin_tot"]
        + canopy["E_can_tot"]
        + soil["E_T"]
        + soil["E_bs"]
        + gw["E_sw"]
    )
    system_runoff = canopy["R_imp"] + gw["R_sw"] + gw["R_gw"]
    total_output = system_et + system_runoff

    balance = P - total_output - delta_total

    results["overall"] = {
        "water_balance_error": float(balance),
        "total_inputs": float(P),
        "total_outputs": float(total_output),
        "total_storage_change": float(delta_total),
        "is_balanced": abs(float(balance)) < tolerance,
    }
    return results


def test_water_balance_all_landuses():
    lookup = default_landuse_lookup()
    encoding = get_landuse_encoding()
    snow_params = SnowParams(day_frac=0.35)
    P = 5.0
    T_srf = 285.0
    PET = 3.0
    tolerance = 1e-5

    for name, code in encoding.items():
        params = HydroParams(snow=snow_params, landuse=lookup[name])
        state = HydroState(
            S_sn=0.0,
            S_snlq=0.0,
            S_skin=0.0,
            S_can=0.0,
            S_so=params.landuse.soil.S_so_max * 0.3,
            S_sw=1.0,
            groundwater=5.0,
            S_glac=0.0,
            S_al=0.0,
        )
        results = evaluate_water_balance_errors_final(
            P, T_srf, PET, params, state, landuse_code=code, tolerance=tolerance
        )
        assert results["overall"]["is_balanced"]
