"""
AUTO-GENERATED DISPATCH WRAPPER.

Select backend with environment variable:
  QALLSE_BACKEND=fast|reference|fastest
(default: fast)
"""

from ._backend import backend as _backend

if _backend() == "reference":
    from .reference.utils import *  # noqa: F401,F403
elif _backend() == "fast":
    from .fast.utils import *  # noqa: F401,F403
elif _backend() == "fastest":
    from .fastest.utils import *  # noqa: F401,F403
else:
    raise RuntimeError("Unreachable backend state")
