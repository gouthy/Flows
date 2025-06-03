from __future__ import annotations
from typing import NamedTuple, Sequence, Tuple, Dict, Any, Optional, List
try:
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_map
except Exception:
    jax = None
    jnp = None
    tree_map = None


def _require_jax():
    if jnp is None:
        raise ImportError("JAX is required to use hydropy.model. Please install jax.")


class GlacierParams(NamedTuple):
    DDF_glac: float
    refreeze_frac: float
    sublimation_frac: float
    LAG_glac: float


class PermafrostParams(NamedTuple):
    max_active_depth: float
    thaw_rate: float
    theta_sat: float


class SnowParams(NamedTuple):
    day_frac: float
    T_sn_min: float = 272.05
    T_sn_max: float = 276.45
    T_melt: float = 273.15
    f_snlq_max: float = 0.06


class CanopyParams(NamedTuple):
    f_bare: float
    f_veg: float
    LAI: float
    cap0: float


class SoilParams(NamedTuple):
    S_so_max: float
    S_so_wilt: float
    S_so_grmin: float
    S_so_grmax: float
    S_so_sg_min: float
    S_so_sg_max: float
    b: float
    R_gr_min: float
    R_gr_max: float
    dt: float
    f_so_crit: float = 0.75
    f_so_bs_low: float = 0.05
    R_gr_exp: float = 1.5


class WaterBodyParams(NamedTuple):
    is_water_body: bool = False
    et_factor: float = 1.2
    surface_resistance: float = 0.0
    albedo: float = 0.06


class GroundwaterParams(NamedTuple):
    f_lake: float = 0.0
    f_wetland: float = 0.0
    LAG_sw: float = 0.0
    LAG_gw: float = 0.0


class WetlandParams(NamedTuple):
    f_wet: float
    LAG_sw: float
    LAG_gw: float


# Default parameters

def _default_inactive_glacier() -> GlacierParams:
    return GlacierParams(DDF_glac=0.0, refreeze_frac=0.0, sublimation_frac=0.0, LAG_glac=1.0)


def _default_inactive_permafrost() -> PermafrostParams:
    return PermafrostParams(max_active_depth=0.0, thaw_rate=0.0, theta_sat=0.0)


def _default_inactive_wetland() -> WetlandParams:
    return WetlandParams(f_wet=0.0, LAG_sw=1.0, LAG_gw=1.0)


def _default_inactive_water_body() -> WaterBodyParams:
    return WaterBodyParams(et_factor=1.0)


def is_water_landuse(landuse_type: str) -> bool:
    water_body_types = {"WATR", "LAKE"}
    return landuse_type in water_body_types


class LandUseParams(NamedTuple):
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams
    imperv_frac: float
    glacier: GlacierParams
    permafrost: PermafrostParams
    wetland: WetlandParams
    water_body: WaterBodyParams
    crop_coeff: float = 1.0


class HydroParams(NamedTuple):
    snow: SnowParams
    landuse: LandUseParams


class HydroState(NamedTuple):
    S_sn: float
    S_snlq: float
    S_skin: float
    S_can: float
    S_so: float
    S_sw: float
    groundwater: float
    S_glac: float
    S_al: float


def default_landuse_lookup() -> Dict[str, LandUseParams]:
    inactive_glacier = _default_inactive_glacier()
    inactive_permafrost = _default_inactive_permafrost()
    inactive_wetland = _default_inactive_wetland()
    inactive_water_body = _default_inactive_water_body()

    return {
        "ENF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.5, cap0=0.25),
            soil=SoilParams(
                S_so_max=250.0, S_so_wilt=60.0,
                S_so_grmin=30.0, S_so_grmax=180.0,
                S_so_sg_min=10.0, S_so_sg_max=200.0,
                b=0.35, R_gr_min=1e-4, R_gr_max=1e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=3.0, LAG_gw=12.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "DBF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=6.0, cap0=0.30),
            soil=SoilParams(
                S_so_max=220.0, S_so_wilt=50.0,
                S_so_grmin=25.0, S_so_grmax=160.0,
                S_so_sg_min=10.0, S_so_sg_max=180.0,
                b=0.30, R_gr_min=1e-4, R_gr_max=2e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=2.5, LAG_gw=10.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "MF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.8, cap0=0.28),
            soil=SoilParams(
                S_so_max=230.0, S_so_wilt=55.0,
                S_so_grmin=28.0, S_so_grmax=170.0,
                S_so_sg_min=10.0, S_so_sg_max=190.0,
                b=0.32, R_gr_min=1e-4, R_gr_max=1.5e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=3.0, LAG_gw=11.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "OS": LandUseParams(
            canopy=CanopyParams(f_bare=0.2, f_veg=0.8, LAI=1.5, cap0=0.10),
            soil=SoilParams(
                S_so_max=100.0, S_so_wilt=30.0,
                S_so_grmin=15.0, S_so_grmax=80.0,
                S_so_sg_min=5.0, S_so_sg_max=90.0,
                b=0.20, R_gr_min=5e-4, R_gr_max=2e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.5, LAG_gw=7.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "GRA": LandUseParams(
            canopy=CanopyParams(f_bare=0.3, f_veg=0.7, LAI=2.0, cap0=0.12),
            soil=SoilParams(
                S_so_max=120.0, S_so_wilt=35.0,
                S_so_grmin=18.0, S_so_grmax=90.0,
                S_so_sg_min=5.0, S_so_sg_max=100.0,
                b=0.22, R_gr_min=5e-4, R_gr_max=2.5e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.0, LAG_gw=8.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "SAV": LandUseParams(
            canopy=CanopyParams(f_bare=0.25, f_veg=0.75, LAI=3.5, cap0=0.15),
            soil=SoilParams(
                S_so_max=140.0, S_so_wilt=40.0,
                S_so_grmin=20.0, S_so_grmax=100.0,
                S_so_sg_min=8.0, S_so_sg_max=120.0,
                b=0.25, R_gr_min=5e-4, R_gr_max=3e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.5, LAG_gw=9.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "CROP": LandUseParams(
            canopy=CanopyParams(f_bare=0.2, f_veg=0.8, LAI=4.0, cap0=0.18),
            soil=SoilParams(
                S_so_max=160.0, S_so_wilt=45.0,
                S_so_grmin=22.0, S_so_grmax=110.0,
                S_so_sg_min=10.0, S_so_sg_max=130.0,
                b=0.25, R_gr_min=1e-3, R_gr_max=4e-3,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=2.0, LAG_gw=10.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.2
        ),
        "URB": LandUseParams(
            canopy=CanopyParams(f_bare=0.1, f_veg=0.2, LAI=1.0, cap0=0.05),
            soil=SoilParams(
                S_so_max=60.0, S_so_wilt=15.0,
                S_so_grmin=8.0, S_so_grmax=35.0,
                S_so_sg_min=2.0, S_so_sg_max=50.0,
                b=0.1, R_gr_min=1e-5, R_gr_max=1e-4,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.0, LAG_gw=5.0),
            imperv_frac=0.7,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=1.0
        ),
        "WATR": LandUseParams(
            canopy=CanopyParams(f_bare=1.0, f_veg=0.0, LAI=0.0, cap0=0.0),
            soil=SoilParams(
                S_so_max=1e-6,
                S_so_wilt=0.0, S_so_grmin=0.0, S_so_grmax=0.0,
                S_so_sg_min=0.0, S_so_sg_max=1e-6,
                b=0.0, R_gr_min=0.0, R_gr_max=0.0, dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=1.0, f_wetland=0.0, LAG_sw=1.0, LAG_gw=1.0),
            imperv_frac=1.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=WaterBodyParams(is_water_body=True, et_factor=1.2),
            crop_coeff=0.0
        ),
        "LAKE": LandUseParams(
            canopy=CanopyParams(f_bare=1.0, f_veg=0.0, LAI=0.0, cap0=0.0),
            soil=SoilParams(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86400.0),
            groundwater=GroundwaterParams(f_lake=1.0),
            imperv_frac=1.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=WaterBodyParams(et_factor=1.1),
            crop_coeff=0.0
        ),
        "WET": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.0, cap0=0.20),
            soil=SoilParams(
                S_so_max=80.0, S_so_wilt=0.0,
                S_so_grmin=0.0, S_so_grmax=20.0,
                S_so_sg_min=0.0, S_so_sg_max=80.0,
                b=0.1, R_gr_min=0.0, R_gr_max=0.0,
                dt=86400.0
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=4.0, LAG_gw=15.0),
            imperv_frac=0.0,
            glacier=inactive_glacier,
            permafrost=inactive_permafrost,
            water_body=inactive_water_body,
            wetland=WetlandParams(f_wet=1.0, LAG_sw=4.0, LAG_gw=15.0),
            crop_coeff=1.0
        ),
        "GLAC": LandUseParams(
            canopy=CanopyParams(f_bare=1.0, f_veg=0.0, LAI=0.0, cap0=0.0),
            soil=SoilParams(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86400.0),
            groundwater=GroundwaterParams(),
            imperv_frac=1.0,
            glacier=GlacierParams(DDF_glac=5.0, refreeze_frac=0.2, sublimation_frac=0.1, LAG_glac=30.0),
            permafrost=inactive_permafrost,
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=0.0
        ),
        "PERM": LandUseParams(
            canopy=CanopyParams(f_bare=1.0, f_veg=0.0, LAI=0.0, cap0=0.0),
            soil=SoilParams(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86400.0),
            groundwater=GroundwaterParams(),
            imperv_frac=1.0,
            glacier=inactive_glacier,
            permafrost=PermafrostParams(max_active_depth=2.0, thaw_rate=0.01, theta_sat=300.0),
            wetland=inactive_wetland,
            water_body=inactive_water_body,
            crop_coeff=0.0
        ),
    }


def snow_module(P: float,
                T_srf: float,
                params: SnowParams,
                S_sn: float,
                S_snlq: float):
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



def skin_canopy_module_corrected(P: jnp.ndarray,
                                P_sn: jnp.ndarray,
                                R_sn: jnp.ndarray,
                                PET: jnp.ndarray,
                                lu: LandUseParams,
                                S_skin: jnp.ndarray,
                                S_can: jnp.ndarray) -> Dict[str, Any]:
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
        S_so
    )
    D_sg = params.S_so_sg_max - params.S_so_sg_min
    c1 = jnp.where(
        D_sg > 0,
        jnp.minimum(1.0, ((params.S_so_sg_max - S_so_sg)/D_sg)**(1+params.b)),
        1.0
    )
    c2 = jnp.where(
        D_sg > 0,
        jnp.maximum(0.0, ((params.S_so_sg_max - S_so_sg - R_tr)/D_sg)**(1+params.b)),
        0.0
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
             + (D_sg/(1+params.b))*(c1-c2)
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
      0.0, 1.0
    )
    theta_bs = jnp.clip(
      (S_so - params.f_so_bs_low*params.S_so_max) /
      ((1-params.f_so_bs_low)*params.S_so_max),
      0.0, 1.0
    )

    E_T_rel = PET_T * theta_T
    E_bs_rel = PET_bs * theta_bs

    E_T_cell = E_T_rel * f_veg
    E_bs_cell = E_bs_rel * f_bare

    R_gr_low = jnp.where(
        params.S_so_max > 0,
        params.R_gr_min * params.dt * (S_so/params.S_so_max),
        0.0
    )
    frac2 = jnp.where(
        (params.S_so_max-params.S_so_grmax) > 0,
        jnp.maximum(0.0, S_so-params.S_so_grmax)/(params.S_so_max-params.S_so_grmax),
        0.0
    )
    R_gr_high = jnp.where(
        S_so > params.S_so_grmax,
        (params.R_gr_max-params.R_gr_min)*params.dt*(frac2**params.R_gr_exp),
        0.0
    )
    cond_low = (S_so <= params.S_so_grmin) | (T_srf < 273.15)
    cond_mid = (S_so <= params.S_so_grmax)
    R_gr = jnp.where(
        cond_low,
        jnp.zeros_like(S_so),
        jnp.where(
            cond_mid,
            R_gr_low,
            R_gr_low + R_gr_high
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
    inflow: float = 0.0
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
        "outflow_rate_used": outflow_rate
    }


def glacier_module(
    P: float,
    T: float,
    PET: float,
    params: GlacierParams,
    S_glac: float
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
        "melt_actual": melt_actual
    }


def permafrost_module(
    P: float,
    T: float,
    PET: float,
    params: PermafrostParams,
    S_al: float
) -> Dict[str, float]:
    thaw_depth = jnp.minimum(
        params.max_active_depth,
        params.thaw_rate * jnp.maximum(T - 273.15, 0.0)
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
    S_sw: float
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
        "R_gw": 0.0
    }

def groundwater_process(
    P: float,
    snow: dict,
    soil: dict,
    PET: float,
    params: GroundwaterParams,
    S_sw: float,
    S_gw: float,
) -> dict:
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


def update_snow_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    params: HydroParams
) -> Tuple[HydroState, Dict[str, Any]]:
    snow = snow_module(P, T, params.snow, state.S_sn, state.S_snlq)
    state = state._replace(S_sn=snow["S_sn_new"], S_snlq=snow["S_snlq_new"])
    return state, snow


def update_canopy_state_jax_corrected(
    state: HydroState,
    P: jnp.ndarray,
    snow: dict,
    PET: jnp.ndarray,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    canopy = skin_canopy_module_corrected(P, snow["P_sn"], snow["R_sn"], PET,
                                         lu, state.S_skin, state.S_can)
    state = state._replace(
        S_skin=canopy["S_skin_new"],
        S_can=canopy["S_can_new"],
    )
    return state, canopy


def update_soil_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    PET: jnp.ndarray,
    snow: dict,
    canopy: dict,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    soil = soil_module(P, T, PET, snow, canopy, lu, state.S_so)
    state = state._replace(S_so=soil["S_so_new"])
    return state, soil


def update_groundwater_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    PET: jnp.ndarray,
    snow: dict,
    soil: dict,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    gw = groundwater_process(
        P=P, snow=snow, soil=soil, PET=PET,
        params=lu.groundwater,
        S_sw=state.S_sw, S_gw=state.groundwater,
    )
    state = state._replace(
        S_sw=gw["S_sw_new"],
        groundwater=gw["S_gw_new"],
    )
    return state, gw


def update_glacier_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    PET: jnp.ndarray,
    params: GlacierParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    glac = glacier_module(P, T, PET, params, state.S_glac)
    new_state = state._replace(S_glac=glac["S_glac_new"])
    return new_state, glac


def update_permafrost_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    PET: jnp.ndarray,
    params: PermafrostParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    perm = permafrost_module(P, T, PET, params, state.S_al)
    new_state = state._replace(S_al=perm["S_al_new"])
    return new_state, perm


def update_wetland_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    PET: jnp.ndarray,
    params: WetlandParams
) -> Tuple[HydroState, Dict[str, Any]]:
    wet = wetland_module(P, PET, params, state.S_sw)
    new_state = state._replace(S_sw=wet["S_sw_new"])
    return new_state, wet


def update_water_body_state_jax(
    state: HydroState,
    P: jnp.ndarray,
    PET: jnp.ndarray,
    params: WaterBodyParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    wb = water_body_module(P, PET, params, state.S_sw)
    new_state = state._replace(S_sw=wb["S_sw_new"])
    return new_state, wb

def create_landuse_encoding():
    landuse_types = list(default_landuse_lookup().keys())
    return {lu_type: i for i, lu_type in enumerate(landuse_types)}


def get_landuse_encoding():
    return create_landuse_encoding()


LANDUSE_ENCODING = get_landuse_encoding()
LANDUSE_DECODING = {v: k for k, v in LANDUSE_ENCODING.items()}


def encode_landuse_types(landuse_types: Sequence[str]) -> jnp.ndarray:
    encoded = [LANDUSE_ENCODING[lu_type] for lu_type in landuse_types]
    return jnp.array(encoded, dtype=jnp.int32)


def decode_landuse_types(encoded_types: jnp.ndarray) -> List[str]:
    return [LANDUSE_DECODING[int(code)] for code in encoded_types]


def _single_cell_model_fixed(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
    landuse_code: int = -1,
) -> jnp.ndarray:
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
            lambda: water_body_branch(),
            lambda: jax.lax.cond(
                wetland_active,
                lambda: wetland_branch(),
                lambda: jax.lax.cond(
                    glacier_active,
                    lambda: glacier_branch(),
                    lambda: jax.lax.cond(
                        permafrost_active,
                        lambda: permafrost_branch(),
                        lambda: standard_branch()
                    )
                )
            )
        )

    init = HydroState(
        S_sn=0.0, S_snlq=0.0,
        S_skin=0.0, S_can=0.0,
        S_so=0.0, S_sw=0.0,
        groundwater=0.0,
        S_glac=0.0, S_al=0.0
    )

    _, runoff_ts = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff_ts

def _validate_inputs(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray
) -> None:
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


def build_cell_params(
    per_cell: Sequence[HydroParams],
    landuse_types: Sequence[str] = None
) -> Tuple[HydroParams, Optional[Sequence[str]]]:
    _require_jax()
    if not per_cell:
        raise ValueError("per_cell sequence cannot be empty")
    if not isinstance(per_cell, (list, tuple)):
        raise ValueError("per_cell must be a list or tuple of HydroParams")
    for i, params in enumerate(per_cell):
        if not isinstance(params, HydroParams):
            raise ValueError(f"Element {i} is not a HydroParams instance: {type(params)}")

    landuse_sequence = None
    if landuse_types is not None:
        if len(landuse_types) != len(per_cell):
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) must match "
                f"number of cells ({len(per_cell)})"
            )
        valid_landuse_types = set(default_landuse_lookup().keys())
        for i, lu_type in enumerate(landuse_types):
            if not isinstance(lu_type, str):
                raise ValueError(f"Landuse type {i} must be a string, got {type(lu_type)}")
            if lu_type not in valid_landuse_types:
                raise ValueError(
                    f"Unknown landuse type '{lu_type}' at index {i}. "
                    f"Valid types: {sorted(valid_landuse_types)}"
                )
        landuse_sequence = list(landuse_types)

    try:
        stacked_params = tree_map(lambda *xs: jnp.stack(xs), *per_cell)
        n_cells = len(per_cell)
        if hasattr(stacked_params.snow, 'day_frac'):
            if stacked_params.snow.day_frac.shape != (n_cells,):
                raise ValueError(
                    f"Stacking failed: expected shape ({n_cells},), "
                    f"got {stacked_params.snow.day_frac.shape}"
                )
        return stacked_params, landuse_sequence
    except Exception as e:
        raise ValueError(f"Failed to stack cell parameters: {str(e)}") from e



def build_cell_params_from_landuse(
    landuse_types: Sequence[str],
    snow_params: SnowParams = None,
    custom_overrides: Dict[str, Dict[str, Any]] = None
) -> Tuple[HydroParams, Sequence[str]]:
    _require_jax()
    if not landuse_types:
        raise ValueError("landuse_types sequence cannot be empty")
    lookup = default_landuse_lookup()
    if snow_params is None:
        snow_params = SnowParams(day_frac=0.35)
    per_cell_params = []
    for i, lu_type in enumerate(landuse_types):
        if not isinstance(lu_type, str):
            raise ValueError(f"Landuse type at index {i} must be a string, got {type(lu_type)}")
        if lu_type not in lookup:
            valid_types = sorted(lookup.keys())
            raise ValueError(
                f"Unknown landuse type '{lu_type}' at index {i}. "
                f"Valid types: {valid_types}"
            )
        landuse_params = lookup[lu_type]
        if custom_overrides and lu_type in custom_overrides:
            overrides = custom_overrides[lu_type]
            landuse_params = _apply_parameter_overrides(landuse_params, overrides, lu_type)
        hydro_params = HydroParams(
            snow=snow_params,
            landuse=landuse_params
        )
        per_cell_params.append(hydro_params)
    return build_cell_params(per_cell_params, landuse_types)

def hydrologic_model(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
    landuse_types: Sequence[str] = None,
) -> jnp.ndarray:
    _require_jax()
    _validate_inputs(precip, evap, temp)
    n_cells = precip.shape[1]
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) must match "
                f"number of cells ({n_cells})"
            )
        landuse_codes = encode_landuse_types(landuse_types)
        def single_cell_model_with_landuse(precip_cell, evap_cell, temp_cell, params_cell, landuse_code):
            return _single_cell_model_fixed(precip_cell, evap_cell, temp_cell, params_cell, landuse_code)
        return jax.vmap(
            single_cell_model_with_landuse,
            in_axes=(1, 1, 1, 0, 0),
            out_axes=1
        )(precip, evap, temp, params, landuse_codes)
    else:
        return jax.vmap(
            _single_cell_model_fixed,
            in_axes=(1, 1, 1, 0),
            out_axes=1
        )(precip, evap, temp, params)


def validate_cell_params(params: HydroParams, landuse_types: Sequence[str] = None) -> None:
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
                    f"Parameter {path} has shape {obj.shape}, "
                    f"expected first dimension to be {n_cells}"
                )
        elif isinstance(obj, (int, float)):
            pass
        else:
            raise ValueError(f"Unexpected parameter type at {path}: {type(obj)}")
    check_shapes(params, "params")
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(
                f"Number of landuse types ({len(landuse_types)}) "
                f"doesn't match parameter arrays ({n_cells})"
            )
    if jnp.any(params.snow.day_frac < 0) or jnp.any(params.snow.day_frac > 1):
        raise ValueError("Snow day_frac must be between 0 and 1")
    if jnp.any(params.landuse.imperv_frac < 0) or jnp.any(params.landuse.imperv_frac > 1):
        raise ValueError("Impervious fraction must be between 0 and 1")


__all__ = [
    "SnowParams", "CanopyParams", "SoilParams", "GroundwaterParams",
    "GlacierParams", "PermafrostParams", "WetlandParams", "WaterBodyParams",
    "LandUseParams", "HydroParams", "HydroState",
    "build_cell_params", "hydrologic_model", "default_landuse_lookup",
    "is_water_landuse", "build_cell_params_from_landuse", "validate_cell_params",
]
