import os
import platform


def _add_additional_dll_paths():
    """
    Add special Windows DLL paths.

    Recent versions of Python do use PATH as a load path anymore. This can
    cause issues with libraries that are used by kernels such as dnnl. Here
    we add several known paths.
    """

    if platform.system() == "Windows":
        _dll_paths = [
            # Add Intel oneAPI directories for XPU support
            r"C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\bin",
        ]

        for _path in _dll_paths:
            if os.path.exists(_path):
                try:
                    os.add_dll_directory(_path)
                except Exception:
                    pass  # Ignore if already added or permission issues
