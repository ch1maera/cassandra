"""Unity Catalog volume file operations for Cassandra.

This module provides functionality to load and save files from Unity Catalog volumes
as HuggingFace Datasets.
"""

from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from fnmatch import fnmatch
import pandas as pd
from datasets import Dataset, load_dataset
import io

from cassandra.config import CassandraConfig
from cassandra.data.exceptions import VolumeNotFoundError, DataLoadError, DataSaveError


class UnityCatalogVolumes:
    """Unity Catalog volume file operations.

    Provides methods to load, list, and save files in Unity Catalog volumes,
    converting between various file formats and HuggingFace Datasets.
    """

    def __init__(self, config: CassandraConfig):
        """Initialize with Cassandra configuration.

        Args:
            config: CassandraConfig instance

        Raises:
            ValueError: If configuration is incomplete
        """
        if not config.is_configured():
            raise ValueError(
                "Configuration incomplete. Required: databricks_profile and warehouse_id"
            )
        self.config = config
        self.client = config.get_databricks_client()

    @staticmethod
    def _parse_volume_path(volume_path: str) -> Dict[str, str]:
        """Parse volume path into components.

        Args:
            volume_path: Path like "/Volumes/catalog/schema/volume/path/to/file"

        Returns:
            Dictionary with 'catalog', 'schema', 'volume', 'path' keys

        Raises:
            ValueError: If path format is invalid
        """
        # Normalize path
        path = volume_path.strip()
        if path.startswith("/Volumes/"):
            path = path[9:]  # Remove /Volumes/ prefix
        elif path.startswith("Volumes/"):
            path = path[8:]  # Remove Volumes/ prefix

        parts = path.split("/", 3)  # Split into catalog/schema/volume/path
        if len(parts) < 3:
            raise ValueError(
                f"Invalid volume path '{volume_path}'. "
                "Expected format: '/Volumes/catalog/schema/volume/path'"
            )

        return {
            "catalog": parts[0],
            "schema": parts[1],
            "volume": parts[2],
            "path": parts[3] if len(parts) > 3 else "",
        }

    @staticmethod
    def _detect_format(file_path: str) -> str:
        """Auto-detect file format from extension.

        Args:
            file_path: Path to file

        Returns:
            Format string: 'csv', 'json', 'parquet', or 'text'
        """
        suffix = Path(file_path).suffix.lower()
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".txt": "text",
        }
        return format_map.get(suffix, "text")

    def _read_file_content(self, file_path: str) -> bytes:
        """Read file content from workspace.

        Args:
            file_path: Full path to file in /Volumes/...

        Returns:
            File contents as bytes

        Raises:
            VolumeNotFoundError: If file doesn't exist
            DataLoadError: If read fails
        """
        try:
            # Use workspace files API to read
            response = self.client.files.download(file_path)
            return response.contents.read()
        except Exception as e:
            if "does not exist" in str(e).lower():
                raise VolumeNotFoundError(f"File not found: {file_path}") from e
            raise DataLoadError(f"Failed to read file: {file_path}") from e

    def _load_file(
        self,
        file_path: str,
        file_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load single file as DataFrame.

        Args:
            file_path: Path to file
            file_format: Format override (csv, json, parquet, text)

        Returns:
            pandas DataFrame

        Raises:
            DataLoadError: If loading fails
        """
        # Detect format if not specified
        if file_format is None:
            file_format = self._detect_format(file_path)

        # Read file content
        content = self._read_file_content(file_path)

        try:
            if file_format == "csv":
                return pd.read_csv(io.BytesIO(content))
            elif file_format == "json":
                return pd.read_json(io.BytesIO(content), lines=True)
            elif file_format == "parquet":
                return pd.read_parquet(io.BytesIO(content))
            elif file_format == "text":
                # Read as text file, one line per row
                text = content.decode("utf-8")
                lines = text.strip().split("\n")
                return pd.DataFrame({"text": lines})
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        except Exception as e:
            raise DataLoadError(
                f"Failed to parse file {file_path} as {file_format}"
            ) from e

    def list_files(
        self,
        volume_path: str,
        pattern: str = "*",
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """List files in Unity Catalog volume.

        Args:
            volume_path: Volume path (e.g., "/Volumes/main/default/data")
            pattern: Glob pattern for filtering files (e.g., "*.json")
            recursive: Whether to search subdirectories

        Returns:
            List of file metadata dictionaries with 'path', 'size_bytes', 'file_type'

        Raises:
            VolumeNotFoundError: If volume path doesn't exist
            DataLoadError: If listing fails
        """
        # Ensure path starts with /Volumes/
        if not volume_path.startswith("/Volumes/"):
            volume_path = f"/Volumes/{volume_path}"

        try:
            all_files = []

            def _list_recursive(path: str) -> None:
                """Recursively list files."""
                try:
                    items = list(self.client.files.list_directory_contents(path))
                except Exception as e:
                    if "does not exist" in str(e).lower():
                        raise VolumeNotFoundError(
                            f"Volume path not found: {path}"
                        ) from e
                    raise

                for item in items:
                    if item.is_directory:
                        if recursive:
                            _list_recursive(item.path)
                    else:
                        # Check if file matches pattern
                        if fnmatch(Path(item.path).name, pattern):
                            all_files.append(
                                {
                                    "path": item.path,
                                    "size_bytes": item.file_size or 0,
                                    "file_type": "file",
                                    "modified_at": getattr(item, "modification_time", None),
                                }
                            )

            _list_recursive(volume_path)
            return all_files

        except VolumeNotFoundError:
            raise
        except Exception as e:
            # Provide more detailed error message
            error_msg = str(e)
            raise DataLoadError(
                f"Failed to list files in {volume_path}: {error_msg}"
            ) from e

    def load_files(
        self,
        volume_path: str,
        pattern: str = "*",
        file_format: Optional[str] = None,
        recursive: bool = True,
    ) -> Dataset:
        """Load files from Unity Catalog volume as HuggingFace Dataset.

        Args:
            volume_path: Volume path
            pattern: File pattern (e.g., "*.csv", "*.json")
            file_format: Format override (auto-detected if None)
            recursive: Search subdirectories

        Returns:
            HuggingFace Dataset with combined file data

        Raises:
            VolumeNotFoundError: If volume doesn't exist
            DataLoadError: If loading fails

        Examples:
            >>> volumes = UnityCatalogVolumes(config)
            >>> dataset = volumes.load_files(
            ...     "/Volumes/main/default/data",
            ...     pattern="*.json"
            ... )
        """
        # List all matching files
        files = self.list_files(volume_path, pattern=pattern, recursive=recursive)

        if not files:
            raise DataLoadError(
                f"No files found matching pattern '{pattern}' in {volume_path}"
            )

        # Load each file and concatenate
        dfs = []
        for file_info in files:
            df = self._load_file(file_info["path"], file_format=file_format)
            dfs.append(df)

        # Concatenate all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Convert to Dataset
        dataset = Dataset.from_pandas(combined_df, preserve_index=False)
        return dataset

    def preview_file(
        self,
        volume_path: str,
        filename: str,
        n_rows: int = 10,
        file_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """Quick preview of volume file data.

        Args:
            volume_path: Volume path (e.g., "/Volumes/main/default/data")
            filename: File name or pattern (e.g., "data.csv", "*.json")
            n_rows: Number of rows to preview
            file_format: Format override (auto-detected if None)

        Returns:
            pandas DataFrame with preview data (limited to n_rows)

        Raises:
            VolumeNotFoundError: If file doesn't exist
            DataLoadError: If loading fails

        Examples:
            >>> volumes = UnityCatalogVolumes(config)
            >>> df = volumes.preview_file(
            ...     "/Volumes/main/default/data",
            ...     "data.csv",
            ...     n_rows=10
            ... )
        """
        # Ensure volume path starts with /Volumes/
        if not volume_path.startswith("/Volumes/"):
            volume_path = f"/Volumes/{volume_path}"

        # Check if filename contains wildcards
        if "*" in filename or "?" in filename:
            # Find matching files
            files = self.list_files(volume_path, pattern=filename, recursive=False)

            if not files:
                raise DataLoadError(
                    f"No files found matching pattern '{filename}' in {volume_path}"
                )

            # Use first matching file
            file_path = files[0]["path"]

            # Note: If multiple files match, we only preview the first one
            if len(files) > 1:
                import logging
                logger = logging.getLogger("cassandra")
                logger.info(
                    f"Multiple files matched pattern '{filename}', previewing first: "
                    f"{file_path.split('/')[-1]}"
                )
        else:
            # Construct full file path directly
            file_path = f"{volume_path.rstrip('/')}/{filename}"

        # Load file as DataFrame
        df = self._load_file(file_path, file_format=file_format)

        # Apply row limit
        return df.head(n_rows)

    def save_files(
        self,
        dataset: Dataset,
        volume_path: str,
        filename: str,
        file_format: Literal["csv", "json", "parquet"] = "parquet",
    ) -> None:
        """Save HuggingFace Dataset to volume file.

        Args:
            dataset: Dataset to save
            volume_path: Volume directory path
            filename: Output filename
            file_format: File format (csv, json, parquet)

        Raises:
            DataSaveError: If save fails

        Note:
            This creates the file in the volume using workspace files API.
        """
        # Ensure volume path starts with /Volumes/
        if not volume_path.startswith("/Volumes/"):
            volume_path = f"/Volumes/{volume_path}"

        # Full output path
        output_path = f"{volume_path.rstrip('/')}/{filename}"

        try:
            # Convert to pandas
            df = dataset.to_pandas()

            # Convert to bytes based on format
            if file_format == "csv":
                content = df.to_csv(index=False).encode("utf-8")
            elif file_format == "json":
                content = df.to_json(orient="records", lines=True).encode("utf-8")
            elif file_format == "parquet":
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                content = buffer.getvalue()
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            # Upload to volume
            self.client.files.upload(
                file_path=output_path,
                contents=io.BytesIO(content),
                overwrite=True,
            )

        except Exception as e:
            raise DataSaveError(f"Failed to save to {output_path}") from e
