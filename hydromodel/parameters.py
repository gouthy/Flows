# Parameter data classes for hydrological model
from __future__ import annotations
from typing import NamedTuple

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
