import importlib.util


def is_extra_installed(extra):
    """
    Check if the specified extra package is installed.

    :param extra: The extra category to check (e.g., 'agents', 'synthesizers').
    :return: Boolean indicating if the specified extra package is installed.
    """
    # Define the mapping of extras to their respective packages
    extras_mapping = {
        "agents": "openai",
        "synthesizers": "elevenlabs",
        "telephony": "twilio",
        "misc": "sentry-sdk",
        "transcribers": "google-cloud-speech",
    }

    # Get the package name for the specified extra
    package = extras_mapping.get(extra)

    if not package:
        raise ValueError(f"Unknown extra category: {extra}")

    # Check if the package is available
    package_spec = importlib.util.find_spec(package)
    return package_spec is not None
