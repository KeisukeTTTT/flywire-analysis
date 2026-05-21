"""FlyWire dataset descriptor (column names, scale factors, etc.)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    display_name: str

    input_neuron_types: tuple
    input_neuron_label: str

    id_column: str = "root_id"
    type_column: str = "primary_type"
    type_pre_column: str = "pre_primary_type"
    type_post_column: str = "post_primary_type"

    scale_factor: float = 1000.0
    x_offset: float = 0.0

    raw_data_subdir: str = ""
    geometry_cache_subdir: str = ""

    weight_column: str = "weight"
    nt_type_column: str = "predictedNt"


FLYWIRE_DESCRIPTOR = DatasetDescriptor(
    name="flywire",
    display_name="FlyWire",
    input_neuron_types=("R1-6",),
    input_neuron_label="photoreceptor",
    scale_factor=6000.0,
    x_offset=17.0,
    raw_data_subdir="flywire",
    geometry_cache_subdir="flywire",
    weight_column="syn_count",
    nt_type_column="nt_type",
)
