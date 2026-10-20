"""Spoorcast: predict the probability that a Dutch passenger train departs late."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spoorcast")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"
