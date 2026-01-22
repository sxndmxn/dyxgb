"""Column renaming transform."""

import polars as pl

from dyxgb.transforms.base import StatelessTransform


class RenameTransform(StatelessTransform):
    """Rename columns to canonical names.

    Maps multiple possible source column names to a single canonical name.
    Handles missing columns gracefully - only renames columns that exist.

    Example config:
        rename:
            customer_id: id    # truth dataset column
            cust_id: id        # unknown dataset column
            CustomerId: id     # another possible name
            txn_amt: amount
            Amount: amount

    This allows both truth and unknown datasets to use the same canonical
    column names after transformation, regardless of their original naming.
    """

    name = "rename"

    def __init__(self, mapping: dict[str, str]) -> None:
        """Initialize rename transform.

        Args:
            mapping: Dictionary mapping source column names to canonical names.
                    Multiple source names can map to the same canonical name.
        """
        self.mapping = mapping

        # Build reverse mapping for validation (canonical -> list of sources)
        self._canonical_sources: dict[str, list[str]] = {}
        for source, canonical in mapping.items():
            if canonical not in self._canonical_sources:
                self._canonical_sources[canonical] = []
            self._canonical_sources[canonical].append(source)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename columns that exist in the DataFrame.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with renamed columns
        """
        # Only rename columns that actually exist in this dataframe
        # This handles the case where truth has "customer_id" but unknown has "cust_id"
        existing_columns = set(df.columns)

        # Build the rename map for this specific dataframe
        rename_map: dict[str, str] = {}
        renamed_to: set[str] = set()  # Track which canonical names we've already mapped

        for source, canonical in self.mapping.items():
            if source in existing_columns:
                # Check if we already renamed something to this canonical name
                if canonical in renamed_to:
                    # Skip - another source column was already renamed to this canonical
                    continue

                # Check if source and canonical are the same (no rename needed)
                if source == canonical:
                    renamed_to.add(canonical)
                    continue

                rename_map[source] = canonical
                renamed_to.add(canonical)

        if rename_map:
            return df.rename(rename_map)
        return df

    def get_params(self) -> dict[str, dict[str, str]]:
        """Get mapping for serialization."""
        return {"mapping": self.mapping}

    def set_params(self, params: dict[str, dict[str, str]]) -> None:
        """Set mapping from deserialization."""
        self.mapping = params.get("mapping", {})

    def __repr__(self) -> str:
        n_mappings = len(self.mapping)
        n_canonical = len(self._canonical_sources)
        return f"RenameTransform({n_mappings} mappings -> {n_canonical} canonical names)"
