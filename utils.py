import re # Not strictly needed for the final version of parse_gcs_uri below, but good to have for utils.

class InvalidGCSPathError(ValueError):
    """Custom error for invalid GCS paths."""
    pass

def parse_gcs_uri(gcs_uri_string: str) -> tuple[str, str]:
    """
    Parses a GCS URI string into bucket name and blob name/prefix.

    Args:
        gcs_uri_string: The GCS URI (e.g., "gs://bucket-name/path/to/blob").

    Returns:
        A tuple (bucket_name, blob_name_or_prefix).
        blob_name_or_prefix can be an empty string if the URI is just "gs://bucket-name/".

    Raises:
        InvalidGCSPathError: If the URI is not a valid GCS path.
    """
    if not isinstance(gcs_uri_string, str) or not gcs_uri_string.startswith("gs://"):
        raise InvalidGCSPathError(
            f"Invalid GCS URI: Must be a string starting with 'gs://'. Got: {gcs_uri_string}"
        )

    path_without_scheme = gcs_uri_string[5:]
    if not path_without_scheme: # Handles "gs://"
        raise InvalidGCSPathError(f"Invalid GCS URI: Path after 'gs://' is empty. Got: {gcs_uri_string}")

    parts = path_without_scheme.split('/', 1)
    bucket_name = parts[0]
    if not bucket_name: # Handles "gs:///"
        raise InvalidGCSPathError(f"Invalid GCS URI: Bucket name is empty. Got: {gcs_uri_string}")

    blob_name_or_prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name_or_prefix
