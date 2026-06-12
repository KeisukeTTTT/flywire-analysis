"""Rigorous lateral-inhibition analysis foundation for the FlyWire optic lobe.

Combines orthogonal axes that the original notebooks treated implicitly or not at
all: processing *stage* (neuropil), *lateral* offset (Δcolumn hex distance), edge
*sign* (NT), and *path length* (mono- vs poly-synaptic). This lets "lateral
inhibition" be defined operationally -- within-stage, spatially offset, and
separated into direct vs interneuron-mediated -- and distinguished from
feedforward / feedback inhibition and disinhibition.
"""

from __future__ import annotations

from .hexgeom import (
    CENTER_MIN_DISTANCE,
    DEFAULT_INWARD_METHOD,
    HEXN,
    INWARD_METHODS,
    REFERENCE_COLUMN_TYPE,
    RIM_MAX_DISTANCE,
    ColumnGeometry,
    axial_to_cart,
    hex_distance,
    interior_cells,
    load_column_assignment,
    pq_hemi_maps,
    region_from_boundary_distance,
)
from .classifier import (
    STAGE_RANK,
    SUBLAYER_RANK,
    LateralInhibitionCriteria,
    classify_inhibition,
    lateral_inhibition_index,
    output_sign_fraction,
    stage_rank,
)
from .edge_region import tag_edges
from .nt_sign import EXCITATORY_NT, INHIBITORY_NT, SIGN_VALUE, add_sign, classify_nt
from .pathtrace import (
    RadialKernels,
    SignedConnGraph,
    disinhibition_onto,
    ffi_motifs,
    net_sign_to_target,
    path_length_distribution,
    rms_radius,
)
from .stage import (
    FAMILY_TO_STAGE,
    LOCAL_INTERNEURON_FAMILIES,
    MEDULLA_SUBLAYERS,
    NEUROPIL_TO_STAGE,
    MedullaDepthRuler,
    assign_medulla_layer,
    assign_medulla_sublayer,
    assign_stage,
    assign_stage_from_manager,
    attach_medulla_sublayer,
    dominant_neuropils,
    is_intrinsic_inhibitory_mediator,
    load_or_build_medulla_sublayer,
    medulla_sublayer_cache_path,
    medulla_sublayer_from_rel_depth,
    neuropil_base,
    neuropil_to_stage,
)

__all__ = [
    # hexgeom
    "CENTER_MIN_DISTANCE",
    "DEFAULT_INWARD_METHOD",
    "HEXN",
    "INWARD_METHODS",
    "REFERENCE_COLUMN_TYPE",
    "RIM_MAX_DISTANCE",
    "ColumnGeometry",
    "axial_to_cart",
    "hex_distance",
    "interior_cells",
    "load_column_assignment",
    "pq_hemi_maps",
    "region_from_boundary_distance",
    # nt_sign
    "EXCITATORY_NT",
    "INHIBITORY_NT",
    "SIGN_VALUE",
    "add_sign",
    "classify_nt",
    # stage
    "FAMILY_TO_STAGE",
    "LOCAL_INTERNEURON_FAMILIES",
    "MEDULLA_SUBLAYERS",
    "NEUROPIL_TO_STAGE",
    "MedullaDepthRuler",
    "assign_medulla_layer",
    "assign_medulla_sublayer",
    "assign_stage",
    "assign_stage_from_manager",
    "attach_medulla_sublayer",
    "dominant_neuropils",
    "is_intrinsic_inhibitory_mediator",
    "load_or_build_medulla_sublayer",
    "medulla_sublayer_cache_path",
    "medulla_sublayer_from_rel_depth",
    "neuropil_base",
    "neuropil_to_stage",
    # edge_region
    "tag_edges",
    # pathtrace
    "RadialKernels",
    "SignedConnGraph",
    "disinhibition_onto",
    "ffi_motifs",
    "net_sign_to_target",
    "path_length_distribution",
    "rms_radius",
    # classifier
    "STAGE_RANK",
    "LateralInhibitionCriteria",
    "classify_inhibition",
    "lateral_inhibition_index",
    "output_sign_fraction",
    "stage_rank",
]
