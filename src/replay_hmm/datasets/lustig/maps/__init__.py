"""The Lustig map builder, a port of the `map_creation_code` MATLAB pipeline.

Stages, as in the MATLAB originals: `session.load_session` (na_loadSession),
`linearize.linearize` (na_linearize), and in `build` `build_map_2d` (na_buildMap2D),
`build_map_linear` (na_buildMapLinear, normalized/absolute), `build_map_wheel`
(na_buildMapWheel), `spatial_info` (na_spatialInfo) and `build_all_maps` (na_buildAllMaps).
`archive` writes the npz; `batch` is `replay-hmm maps build --dataset lustig`.
"""
