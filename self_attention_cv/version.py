import sys

__version__ = "1.2.4"

msg = "Self_attention_cv is only compatible with Python 3.0 and newer."

if sys.version_info < (3, 0):
    raise ImportError(msg)
