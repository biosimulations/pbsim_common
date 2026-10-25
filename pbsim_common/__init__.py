standard_types = {
    # A result replaces the one before it: a step that fires again (its inputs changed) must not append its rows
    # to the previous result's, which is what bigraph-schema's list apply does.
    "numeric_result": {
        "time": "overwrite[list[float]]",
        "columns": "overwrite[list[string]]",
        "values": "overwrite[list[list[float]]]",
        # 'n_spacial_dimensions': 'tuple[int, int]'
    },
    "numeric_results": "map[numeric_result]",
    "columns_of_interest": "list[string]",
    # ReaDDy related types
    "particle": {"type_name": "string", "position": "tuple[float,float,float]", "neighbor_ids": "list[integer]"},
    "topology": {"type_name": "string", "particle_ids": "list[integer]"},
}
__all__ = ["standard_types"]
