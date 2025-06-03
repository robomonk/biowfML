import unittest
from utils import parse_gcs_uri, InvalidGCSPathError

class TestParseGCSURI(unittest.TestCase):

    def test_valid_uris(self):
        self.assertEqual(parse_gcs_uri("gs://bucket-name/path/to/blob"), ("bucket-name", "path/to/blob"))
        self.assertEqual(parse_gcs_uri("gs://bucket-name/blob"), ("bucket-name", "blob"))
        self.assertEqual(parse_gcs_uri("gs://bucket-name/"), ("bucket-name", ""))
        self.assertEqual(parse_gcs_uri("gs://bucket-name"), ("bucket-name", "")) # Technically same as above

    def test_invalid_scheme(self):
        with self.assertRaisesRegex(InvalidGCSPathError, "Must be a string starting with 'gs://'"):
            parse_gcs_uri("http://bucket-name/path/to/blob")
        with self.assertRaisesRegex(InvalidGCSPathError, "Must be a string starting with 'gs://'"):
            parse_gcs_uri("gcs://bucket-name/path/to/blob") # Common typo
        with self.assertRaisesRegex(InvalidGCSPathError, "Must be a string starting with 'gs://'"):
            parse_gcs_uri("/bucket-name/path/to/blob")
        with self.assertRaisesRegex(InvalidGCSPathError, "Must be a string starting with 'gs://'"):
            parse_gcs_uri(123) # Non-string input

    def test_empty_or_malformed_path(self):
        with self.assertRaisesRegex(InvalidGCSPathError, "Path after 'gs://' is empty"):
            parse_gcs_uri("gs://")
        with self.assertRaisesRegex(InvalidGCSPathError, "Bucket name is empty"):
            parse_gcs_uri("gs:///")
        with self.assertRaisesRegex(InvalidGCSPathError, "Bucket name is empty"):
            parse_gcs_uri("gs:///path/to/blob")

    def test_bucket_name_only(self):
        # This case is handled by parse_gcs_uri returning an empty string for the blob name/prefix
        self.assertEqual(parse_gcs_uri("gs://my-bucket"), ("my-bucket", ""))

if __name__ == '__main__':
    unittest.main()
