"""Custom exceptions for Unity Catalog data operations."""


class CassandraDataError(Exception):
    """Base exception for Cassandra data operations."""

    pass


class TableNotFoundError(CassandraDataError):
    """Raised when Unity Catalog table doesn't exist."""

    pass


class VolumeNotFoundError(CassandraDataError):
    """Raised when Unity Catalog volume doesn't exist."""

    pass


class DataLoadError(CassandraDataError):
    """Raised when data loading fails."""

    pass


class DataSaveError(CassandraDataError):
    """Raised when data saving fails."""

    pass
