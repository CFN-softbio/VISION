from typing import List, Dict, Tuple

_TRACKED_PVS_11BM: List[str] = [
    'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr',
    'XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr',
    'XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr',
    'XF:11BMB-ES{BS-Ax:X}Mtr',
    'XF:11BMB-ES{BS-Ax:Y}Mtr',
    'XF:11BMB-ES{BS-Ax:Phi}Mtr',
    'XF:11BMB-ES{SM:1-Ax:Srot}Mtr',
    "XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime",
    "XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod",
    "XF:11BMB-ES{Det:PIL2M}:cam1:NumImages",
    "XF:11BMB-ES{Det:PIL2M}:cam1:Acquire",
    "XF:11BM-ES:{LINKAM}:STARTHEAT",
    "XF:11BM-ES:{LINKAM}:SETPOINT:SET",
    "XF:11BM-ES:{LINKAM}:RAMPRATE:SET",
    "XF:11BM-ES:{LINKAM}:TEMP",
]

_DEFAULTS_11BM: Dict[str, float] = {
    'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr': 0.0,
    'XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr': 0.0,
    'XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr': 0.0,
    'XF:11BMB-ES{BS-Ax:X}Mtr': 0.0,
    'XF:11BMB-ES{BS-Ax:Y}Mtr': 0.0,
    'XF:11BMB-ES{BS-Ax:Phi}Mtr': 0.0,
    'XF:11BMB-ES{SM:1-Ax:Srot}Mtr': 0.0,
    "XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime": 0.0,
    "XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod": 0.0,
    "XF:11BMB-ES{Det:PIL2M}:cam1:NumImages": 0,
    "XF:11BMB-ES{Det:PIL2M}:cam1:Acquire": 0,
    "XF:11BM-ES:{LINKAM}:STARTHEAT": 0,
    "XF:11BM-ES:{LINKAM}:SETPOINT:SET": 25,
    "XF:11BM-ES:{LINKAM}:RAMPRATE:SET": 10,
    "XF:11BM-ES:{LINKAM}:TEMP": 25,
}

def get_pv_config(beamline: str = "11BM") -> tuple[List[str], Dict[str, float]]:
    """
    Return (tracked_pvs, default_values) for the requested beamline.
    Extend this function when new CMS / beamlines are added.
    """
    beamline_upper = beamline.upper()

    if beamline_upper == "11BM" or beamline_upper == "BLUESKY":
        return _TRACKED_PVS_11BM, _DEFAULTS_11BM

    raise ValueError(f"Unknown beamline '{beamline}'")
