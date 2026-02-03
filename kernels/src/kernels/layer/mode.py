from enum import Flag, auto


class Mode(Flag):
    """
    Kernelize mode

    The `Mode` flag is used by [`kernelize`] to select kernels for the given mode. Mappings can be registered for
    specific modes.

    Attributes:
        INFERENCE: The kernel is used for inference.
        TRAINING: The kernel is used for training.
        TORCH_COMPILE: The kernel is used with `torch.compile`.
        FALLBACK: In a kernel mapping, this kernel is used when no other mode matches.

    Note:
        Different modes can be combined. For instance, `INFERENCE | TORCH_COMPILE` should be used for layers that
        are used for inference *with* `torch.compile`.

    """

    _NONE = 0
    FALLBACK = auto()
    TRAINING = auto()
    INFERENCE = auto()
    TORCH_COMPILE = auto()

    def __or__(self, other: "Mode") -> "Mode":
        union = super().__or__(other)

        if Mode.INFERENCE in union and Mode.TRAINING in union:
            raise ValueError("Mode.INFERENCE and Mode.TRAINING are mutually exclusive.")

        if Mode.FALLBACK in union and union != Mode.FALLBACK:
            raise ValueError("Mode.FALLBACK cannot be combined with other modes.")

        return union
