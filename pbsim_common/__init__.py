standard_types = {
    "numeric_result": {
        "time": "list[float]",
        "columns": "list[string]",
        "values": "list[list[float]]",
        # 'n_spacial_dimensions': 'tuple[int, int]'
    },
    "numeric_results": "map[numeric_result]",
    "columns_of_interest": "list[string]",
    # ReaDDy related types
    "particle": {"type_name": "string", "position": "tuple[float,float,float]", "neighbor_ids": "list[integer]"},
    "topology": {"type_name": "string", "particle_ids": "list[integer]"},
}
__all__ = ["standard_types"]
