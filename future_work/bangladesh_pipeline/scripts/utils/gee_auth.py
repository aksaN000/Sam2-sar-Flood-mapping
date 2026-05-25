"""Earth Engine initialization helper with a clearer failure message.

The default `ee.Initialize()` raises a generic exception when credentials
are missing or expired, which leaves new users guessing what to do. This
helper wraps the call so the failure points the user at the right next
command.
"""

from __future__ import annotations


def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize the Earth Engine session, with a friendly error.

    Parameters
    ----------
    project
        Optional Google Cloud project id. Required when your Earth Engine
        account is registered against a Cloud project rather than the
        legacy free-tier credentials. Pass None to use whatever default
        is stored in your local Earth Engine credentials file.

    Raises
    ------
    RuntimeError
        If Earth Engine credentials are missing, expired, or otherwise
        unusable. The message tells the user how to fix it.
    """
    import ee

    try:
        if project is None:
            ee.Initialize()
        else:
            ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run "
            "`earthengine authenticate` in a normal user shell, sign in "
            "with a Google account that has Earth Engine access, then "
            "re-run this script. If your account is registered against "
            "a Google Cloud project, also pass the project id via the "
            "`project` argument or the EARTHENGINE_PROJECT environment "
            f"variable. Original error: {exc!r}"
        ) from exc
