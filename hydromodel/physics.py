"""Core physical process functions used in the hydrologic model."""
from __future__ import annotations
from typing import Any, Dict

import jax.numpy as jnp

from .parameters import (
    SnowParams,
    LandUseParams,
    WaterBodyParams,
    GlacierParams,
    PermafrostParams,
    WetlandParams,
    GroundwaterParams,
)


def snow_module(P: float, T_srf: float, params: SnowParams, S_sn: float, S_snlq: float) -> Dict[str, jnp.ndarray]:
    """Snow accumulation, melt, retention, and refreeze."""
    frac_snow = (params.T_sn_max - T_srf) / (params.T_sn_max - params.T_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    P_sn = P * frac_snow

    DDF = params.day_frac * 8.3 + 0.7
    melt_pot = jnp.maximum(0.0, DDF * (T_srf - params.T_melt))
    R_sn_raw = jnp.minimum(melt_pot, S_sn + P_sn)

    max_liquid = params.f_snlq_max * (S_sn + P_sn)
    possible_ret = max_liquid - S_snlq
    F_snlq = jnp.clip(R_sn_raw, 0.0, possible_ret)

    R_sn = R_sn_raw - F_snlq

    refreeze = jnp.where(T_srf < params.T_melt, S_snlq + F_snlq, 0.0)

    dS_snlq = F_snlq - refreeze
    S_snlq_new = jnp.clip(S_snlq + dS_snlq, 0.0, None)

    dS_sn = P_sn - R_sn_raw + refreeze
    S_sn_new = jnp.clip(S_sn + dS_sn, 0.0, None)

    return {
        "P_sn": jnp.maximum(0.0, P_sn),
        "R_sn": jnp.maximum(0.0, R_sn),
        "F_snlq": jnp.maximum(0.0, F_snlq),
        "refreeze": jnp.maximum(0.0, refreeze),
        "S_sn_new": jnp.maximum(0.0, S_sn_new),
        "S_snlq_new": jnp.maximum(0.0, S_snlq_new),
    }


def skin_canopy_module_corrected(
    P: jnp.ndarray,
    P_sn: jnp.ndarray,
    R_sn: jnp.ndarray,
    PET: jnp.ndarray,
    lu: LandUseParams,
    S_skin: jnp.ndarray,
    S_can: jnp.ndarray,
) -> Dict[str, Any]:
    """Canopy module using total area approach."""
    P_ra = P - P_sn
    R_imp = P_ra * lu.imperv_frac
    P_ra_p = P_ra * (1 - lu.imperv_frac)

    total_pervious_water = P_ra_p + R_sn

    f_skin = lu.canopy.f_bare * (1 - lu.imperv_frac)
    f_canv = lu.canopy.f_veg * (1 - lu.imperv_frac)
    f_pervious_total = f_skin + f_canv

    out: Dict[str, Any] = {}
    tiny = 1e-6

    for prc, f_area, S_current in [
        ("skin", f_skin, S_skin),
        ("can", f_canv, S_can),
    ]:
        if f_pervious_total > tiny:
            allocated_water = total_pervious_water * (f_area / f_pervious_total)
        else:
            allocated_water = 0.0

        available_water = S_current + allocated_water

        if prc == "skin":
            storage_capacity = lu.canopy.cap0 * f_area
        else:
            storage_capacity = lu.canopy.LAI * f_area

        storage_capacity = jnp.maximum(storage_capacity, tiny)

        f_wet = jnp.minimum(1.0, available_water / storage_capacity)

        potential_et = PET * lu.crop_coeff * f_area * f_wet
        actual_et = jnp.minimum(available_water, potential_et)

        water_after_et = available_water - actual_et
        excess_water = jnp.maximum(water_after_et - storage_capacity, 0.0)

        new_storage = jnp.clip(water_after_et - excess_water, 0.0, storage_capacity)

        out[f"S_{prc}_new"] = new_storage
        out[f"E_{prc}_tot"] = actual_et
        out[f"R_{prc}_tot"] = excess_water

    out["R_imp"] = R_imp
    out["R_tr"] = out["R_skin_tot"] + out["R_can_tot"]

    return out


def soil_module(
    P: float,
    T_srf: float,
    PET: float,
    snow: dict,
    canopy: dict,
    lu: LandUseParams,
    S_so: float,
) -> Dict[str, Any]:
    params = lu.soil
    canopy_params = lu.canopy

    R_tr = canopy["R_tr"]
    E_can = canopy["E_can_tot"]
    E_skin = canopy["E_skin_tot"]
    f_bare = canopy_params.f_bare * (1 - lu.imperv_frac)
    f_veg = canopy_params.f_veg * (1 - lu.imperv_frac)

    S_so_sg = jnp.where(
        S_so > params.S_so_sg_min,
        params.S_so_sg_max
        - (params.S_so_sg_max - params.S_so_sg_min)
          * (1 - (S_so - params.S_so_sg_min)
                 / (params.S_so_max - params.S_so_sg_min)) ** (1/(1+params.b)),
        S_so,
    )
    D_sg = params.S_so_sg_max - params.S_so_sg_min
    c1 = jnp.where(
        D_sg > 0,
        jnp.minimum(1.0, ((params.S_so_sg_max - S_so_sg)/D_sg)**(1+params.b)),
        1.0,
    )
    c2 = jnp.where(
        D_sg > 0,
        jnp.maximum(0.0, ((params.S_so_sg_max - S_so_sg - R_tr)/D_sg)**(1+params.b)),
        0.0,
    )

    is_frozen = (T_srf < 273.15)
    no_through = (R_tr <= 0) | (S_so_sg + R_tr <= params.S_so_sg_min)
    over_subgrid = (S_so_sg + R_tr > params.S_so_sg_max)

    ideal_infil = jnp.where(
        is_frozen, 0.0,
        jnp.where(
          no_through, R_tr,
          jnp.where(
            over_subgrid,
            jnp.maximum(0.0, params.S_so_max - S_so),
            jnp.minimum(R_tr,
               jnp.maximum(0.0, S_so - params.S_so_max)
             + (D_sg/(1+params.b))*(c1-c2),
            )
          )
        )
    )
    avail_cap = jnp.maximum(0.0, params.S_so_max - S_so)
    infiltration = jnp.clip(ideal_infil, 0.0, jnp.minimum(R_tr, avail_cap))
    R_srf = R_tr - infiltration

    PET_T = jnp.maximum(0.0, PET * lu.crop_coeff - E_can)
    PET_bs = jnp.maximum(0.0, PET * lu.crop_coeff - E_skin)

    theta_T = jnp.clip(
      (S_so - params.S_so_wilt) /
      (params.f_so_crit * params.S_so_max - params.S_so_wilt),
      0.0, 1.0,
    )
    theta_bs = jnp.clip(
      (S_so - params.f_so_bs_low*params.S_so_max) /
      ((1-params.f_so_bs_low)*params.S_so_max),
      0.0, 1.0,
    )

    E_T_rel = PET_T * theta_T
    E_bs_rel = PET_bs * theta_bs

    E_T_cell = E_T_rel * f_veg
    E_bs_cell = E_bs_rel * f_bare

    R_gr_low = jnp.where(
        params.S_so_max > 0,
        params.R_gr_min * params.dt * (S_so/params.S_so_max),
        0.0,
    )
    frac2 = jnp.where(
        (params.S_so_max-params.S_so_grmax) > 0,
        jnp.maximum(0.0, S_so-params.S_so_grmax)/(params.S_so_max-params.S_so_grmax),
        0.0,
    )
    R_gr_high = jnp.where(
        S_so > params.S_so_grmax,
        (params.R_gr_max-params.R_gr_min)*params.dt*(frac2**params.R_gr_exp),
        0.0,
    )
    cond_low = (S_so <= params.S_so_grmin) | (T_srf < 273.15)
    cond_mid = (S_so <= params.S_so_grmax)
    R_gr = jnp.where(
        cond_low,
        jnp.zeros_like(S_so),
        jnp.where(
            cond_mid,
            R_gr_low,
            R_gr_low + R_gr_high,
        )
    )

    dS_so = infiltration - R_gr - E_T_cell - E_bs_cell
    S_so_new = jnp.clip(S_so + dS_so, 0.0, params.S_so_max)

    return {
        "infiltration": infiltration,
        "R_srf": R_srf,
        "R_gr": R_gr,
        "E_T": E_T_cell,
        "E_bs": E_bs_cell,
        "ΔS_so": dS_so,
        "S_so_new": S_so_new,
    }


def water_body_module(
    P: float,
    PET: float,
    params: WaterBodyParams,
    S_sw: float,
    inflow: float = 0.0,
) -> Dict[str, float]:
    """Water body processing."""
    available_water = P + S_sw + inflow
    ET_water = jnp.minimum(PET * params.et_factor, available_water)
    remaining_after_ET = jnp.maximum(0.0, available_water - ET_water)
    outflow_rate = 0.1
    R_water = remaining_after_ET * outflow_rate
    S_sw_new = jnp.maximum(0.0, remaining_after_ET - R_water)
    return {
        "S_sw_new": S_sw_new,
        "ET_water": ET_water,
        "R_water": R_water,
        "outflow_rate_used": outflow_rate,
    }


def glacier_module(
    P: float,
    T: float,
    PET: float,
    params: GlacierParams,
    S_glac: float,
) -> Dict[str, float]:
    melt_potential = jnp.maximum(0.0, params.DDF_glac * (T - 273.15))
    melt_actual = jnp.minimum(melt_potential, S_glac)
    refreeze = params.refreeze_frac * melt_actual
    sublim = jnp.minimum(params.sublimation_frac * PET, S_glac + P)
    S_glac_new = jnp.clip(S_glac + P - melt_actual - sublim + refreeze, 0.0, None)
    R_glac = (melt_actual - refreeze) / (params.LAG_glac + 1.0)
    return {
        "S_glac_new": S_glac_new,
        "R_glac": R_glac,
        "E_glac": sublim,
        "melt_potential": melt_potential,
        "melt_actual": melt_actual,
    }


def permafrost_module(
    P: float,
    T: float,
    PET: float,
    params: PermafrostParams,
    S_al: float,
) -> Dict[str, float]:
    thaw_depth = jnp.minimum(
        params.max_active_depth,
        params.thaw_rate * jnp.maximum(T - 273.15, 0.0),
    )
    cap = params.theta_sat * thaw_depth
    fill = jnp.minimum(P, cap - S_al)
    E_al = jnp.minimum(PET, S_al + fill)
    R_perm = P - fill
    S_al_new = jnp.clip(S_al + fill - E_al, 0.0, cap)
    return {"S_al_new": S_al_new, "R_perm": R_perm, "E_al": E_al}


def wetland_module(
    P: float,
    PET: float,
    params: WetlandParams,
    S_sw: float,
) -> Dict[str, float]:
    f = params.f_wet
    SWin = P * f
    available_for_ET = S_sw + SWin
    E_wet = jnp.minimum(PET * f, available_for_ET)
    remaining_after_ET = S_sw + SWin - E_wet
    R_total = remaining_after_ET / (params.LAG_sw + 1.0)
    S_sw_new = jnp.maximum(0.0, remaining_after_ET - R_total)
    return {
        "S_sw_new": S_sw_new,
        "E_wet": E_wet,
        "R_sw": R_total,
        "R_gw": 0.0,
    }


def groundwater_process(
    P: float,
    snow: dict,
    soil: dict,
    PET: float,
    params: GroundwaterParams,
    S_sw: float,
    S_gw: float,
) -> Dict[str, jnp.ndarray]:
    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_srf = soil["R_srf"]
    R_gr = soil["R_gr"]

    f_sw = jnp.maximum(params.f_lake, params.f_wetland)

    P_ra = P - P_sn

    sw_inputs = (P_ra + R_sn) * f_sw + R_srf

    E_sw_potential = PET * f_sw
    E_sw = jnp.minimum(E_sw_potential, S_sw)

    R_sw = S_sw / (params.LAG_sw + 1.0)

    dS_sw = sw_inputs - E_sw - R_sw
    S_sw_new = jnp.clip(S_sw + dS_sw, 0.0, None)

    R_gw = S_gw / (params.LAG_gw + 1.0)

    dS_gw = R_gr - R_gw
    S_gw_new = jnp.clip(S_gw + dS_gw, 0.0, None)

    return {
        "f_sw": f_sw,
        "P_ra": P_ra,
        "R_srf": R_srf,
        "R_sw": R_sw,
        "E_sw": E_sw,
        "ΔS_sw": dS_sw,
        "S_sw_new": S_sw_new,
        "R_gw": R_gw,
        "ΔS_gw": dS_gw,
        "S_gw_new": S_gw_new,
    }

