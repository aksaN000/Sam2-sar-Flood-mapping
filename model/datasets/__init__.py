"""Dataset classes for the active evaluation surface.

The Bangladesh test set dataset class lives under ../future_work/ along
with its construction pipeline; it is not imported from here because the
test set is not part of the principal evaluation in this thesis.
"""

from .sen1floods11 import Sen1Floods11Dataset

__all__ = ["Sen1Floods11Dataset"]
