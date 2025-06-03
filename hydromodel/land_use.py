"""Land use related utilities and default parameter lookups."""
from __future__ import annotations
from typing import Dict, Sequence, List, Any

import jax.numpy as jnp

from .parameters import (
    GlacierParams,
    PermafrostParams,
    WetlandParams,
    WaterBodyParams,
    LandUseParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
)


def _default_inactive_glacier() -> GlacierParams:
    return GlacierParams(DDF_glac=0.0, refreeze_frac=0.0, sublimation_frac=0.0, LAG_glac=1.0)


def _default_inactive_permafrost() -> PermafrostParams:
    return PermafrostParams(max_active_depth=0.0, thaw_rate=0.0, theta_sat=0.0)


def _default_inactive_wetland() -> WetlandParams:
    return WetlandParams(f_wet=0.0, LAG_sw=1.0, LAG_gw=1.0)


def _default_inactive_water_body() -> WaterBodyParams:
    return WaterBodyParams(et_factor=1.0)


LOOKUP_CACHE: Dict[str, LandUseParams] | None = None


def default_landuse_lookup() -> Dict[str, LandUseParams]:
    global LOOKUP_CACHE
    if LOOKUP_CACHE is not None:
        return LOOKUP_CACHE
    inactive_glacier = _default_inactive_glacier()
    inactive_permafrost = _default_inactive_permafrost()
    inactive_wetland = _default_inactive_wetland()
    inactive_water_body = _default_inactive_water_body()

    lookup = {
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
    LOOKUP_CACHE = lookup
    return lookup


def is_water_landuse(landuse_type: str) -> bool:
    return landuse_type in {"WATR", "LAKE"}


def create_landuse_encoding() -> Dict[str, int]:
    landuse_types = list(default_landuse_lookup().keys())
    return {lu_type: i for i, lu_type in enumerate(landuse_types)}


def get_landuse_encoding() -> Dict[str, int]:
    return create_landuse_encoding()


LANDUSE_ENCODING: Dict[str, int] = get_landuse_encoding()
LANDUSE_DECODING = {v: k for k, v in LANDUSE_ENCODING.items()}


def encode_landuse_types(landuse_types: Sequence[str]) -> jnp.ndarray:
    encoded = [LANDUSE_ENCODING[lu] for lu in landuse_types]
    return jnp.array(encoded, dtype=jnp.int32)


def decode_landuse_types(encoded_types: jnp.ndarray) -> List[str]:
    return [LANDUSE_DECODING[int(code)] for code in encoded_types]


def _apply_parameter_overrides(params: LandUseParams, overrides: Dict[str, Any], name: str) -> LandUseParams:
    values = params._asdict()
    for key, val in overrides.items():
        if key not in values:
            raise KeyError(f"Unknown parameter '{key}' for land use '{name}'")
        orig = values[key]
        if isinstance(orig, tuple) and not isinstance(orig, (float, int)):
            values[key] = orig.__class__(*(overrides.get(key, orig)))
        else:
            values[key] = val
    return LandUseParams(**values)
