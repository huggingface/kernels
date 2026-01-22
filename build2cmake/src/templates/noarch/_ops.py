import torch
ops = torch.ops.{{ ops_name }}

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"{{ ops_name }}::{op_name}"
