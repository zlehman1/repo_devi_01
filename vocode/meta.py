import pkg_resources


def is_extra_installed(extra):
    """
    Check if the specified extra package is installed.

    :param extra: The extra category to check (e.g., 'agents', 'synthesizers').
    :return: Boolean indicating if the specified extra package is installed.
    """
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

    try:
        # Attempt to get the distribution for the package
        pkg_resources.get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False
