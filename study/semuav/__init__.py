"""SemUAV-VLM full-study research package.

The package deliberately separates generated integration fixtures from scientific
outputs.  A result is publishable only when the full-profile gates pass.
"""

from .registry import RunContext, StudyGateError

__all__ = ["RunContext", "StudyGateError"]
__version__ = "1.0.0"
