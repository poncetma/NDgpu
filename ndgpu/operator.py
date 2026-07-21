"""Backward-compatible re-export surface for the diffusion/transport operators.

The operators now live in three focused modules:

- ``stencil`` -- the matrix-free structured finite-volume stencil
  (:class:`~ndgpu.stencil.GroupOperator`) and the boundary-condition vocabulary
  (``normalize_bc``, ``face_alpha``, ``harmonic_mean``, ``robin_face_term``)
  shared by every spatial operator.
- ``sp3`` -- the dedicated symmetric two-moment SP3 / SDP1 block
  (:class:`~ndgpu.sp3.SP3GroupOperator`).
- ``spn`` -- the general order-N SDPN / SPN block
  (:class:`~ndgpu.spn.SDPNGroupOperator`), the symmetrizing-congruence CG path
  (:class:`~ndgpu.spn.CongruentSDPNOperator`), the coefficient tables, and the
  Marshak/congruence machinery.

Importing from ``ndgpu.operator`` continues to work; new code may import from
the focused modules directly.
"""

from __future__ import annotations

from .sp3 import SP3GroupOperator
from .spn import (CongruentSDPNOperator, SDPNGroupOperator, _congruence_available,
                  _congruence_transform, _diag_similarity, _marshak_faces,
                  _strip_vacuum, _SDPN_C, _SDPN_G, _SPN_C, _SPN_G)
from .stencil import (BC_REFLECTIVE, BC_VACUUM, BC_ZERO_FLUX, GroupOperator,
                      _face_valid, face_alpha, harmonic_mean, normalize_bc,
                      robin_face_term)

__all__ = [
    "BC_REFLECTIVE", "BC_VACUUM", "BC_ZERO_FLUX",
    "GroupOperator", "SP3GroupOperator", "SDPNGroupOperator",
    "CongruentSDPNOperator",
    "face_alpha", "harmonic_mean", "normalize_bc", "robin_face_term",
    "_SDPN_C", "_SDPN_G", "_SPN_C", "_SPN_G",
    "_congruence_available", "_congruence_transform", "_diag_similarity",
    "_marshak_faces", "_strip_vacuum", "_face_valid",
]
