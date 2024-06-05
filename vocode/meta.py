import importlib.util

from loguru import logger


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


def ensure_punkt_installed():
    try:
        from nltk.data import find

        find("tokenizers/punkt")
        logger.debug("'punkt' tokenizer is already installed.")
    except LookupError:
        from nltk import download

        # If not installed, download 'punkt'
        logger.info("Downloading 'punkt' tokenizer...")
        download("punkt")
        logger.info("'punkt' tokenizer downloaded successfully.")
    except ImportError as err:
        if is_extra_installed("synthesizers"):
            raise Exception(
                "The 'punkt' tokenizer is required for the Eleven Labs synthesizer."
            ) from err
