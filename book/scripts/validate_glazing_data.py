"""
Validates glazing data CSV files against the OpenGLASSLab data schema.

The schema is defined as a Pydantic model (``GlazingRecord``).
Each CSV row is validated by attempting to construct a ``GlazingRecord`` from it;
Pydantic's ``ValidationError`` is then translated into ``GlazingDataError`` instances
for reporting.

Checks performed
----------------
1.  Required columns exist (derived from the model's field set)
2.  No nulls in mandatory fields
3.  Allowed values for categorical fields (``pane_composition_*``, ``pane_glass_type_*``)
4.  ``building_id`` follows the ``LOCATION_BUILDING`` format (e.g. ``AMSTERDAM_RIJKSMUSEUM``)
5.  ``batch_id`` follows the ``BATCH-##`` format (e.g. ``BATCH-01``)
6.  ``unit_id`` follows the ``UNIT-###`` format (e.g. ``UNIT-001``)
7.  ``unit_width`` and ``unit_height`` are positive numbers
8.  ``number_of_panes`` is a positive integer
9.  Inner pane fields are provided when ``number_of_panes >= 2``
10. Middle pane fields are provided when ``number_of_panes == 3``
11. ``pane_nominal_thickness_*`` values are positive when provided
12. ``spacer_thickness`` is non-negative when provided
13. Duplicate ``building_id`` + ``unit_id`` combination detection

Warnings (non-blocking)
-----------------------
- Inner or middle pane fields are filled for a single-pane unit
- ``spacer_thickness`` is missing for a multi-pane unit

Notes
-----
Both comma-separated and semicolon-separated CSV files are supported.
The delimiter is detected automatically by pandas.
Decimal values may use either ``.`` or ``,`` as the decimal separator.

Examples
--------
Validate all CSVs in the default data folder::

    python validate_glazing_data.py

Validate a single file::

    python validate_glazing_data.py path/to/G_AMSTERDAM_RIJKSMUSEUM.csv

Validate all CSVs in a given folder::

    python validate_glazing_data.py path/to/folder/
"""

import re
import sys
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strings that are treated as missing/null values during validation.
# "null" is included because upstream READMEs instruct researchers to write
# "NULL" (case-insensitive) in optional numeric fields when a value is
# genuinely not yet measured/known.
_NULL_STRINGS: set[str] = {"na", "n/a", "-", "", "null"}

# Strings that mean "I don't know" and should map to None in nullable fields.
_UNKNOWN_STRINGS: set[str] = _NULL_STRINGS | {"unknown"}

# <<< BEGIN GENERATED CONSTANTS — run scripts/generate_validators.py to update >>>
_BUILDING_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9\-]*_[A-Z][A-Z0-9\-]*$")
_BATCH_ID_PATTERN = re.compile(r"^BATCH-\d{2,}$")
_UNIT_ID_PATTERN = re.compile(r"^UNIT-\d{3,}$")
# <<< END GENERATED CONSTANTS >>>

# ---------------------------------------------------------------------------
# Shared pre-validators (BeforeValidator functions)
# ---------------------------------------------------------------------------

def _none_if_null(v: object) -> object:
    """
    Convert null-like strings to ``None`` before type validation.

    Uses ``_UNKNOWN_STRINGS`` so that both generic null markers (``"na"``,
    ``"n/a"``, ``"-"``, ``""``) and ``"unknown"`` are normalized to ``None``
    in nullable fields.

    Parameters
    ----------
    v : object
        Raw value from the CSV row dict.

    Returns
    -------
    object
        ``None`` if the value is null-like or ``"unknown"``, otherwise the
        original value.
    """
    if isinstance(v, str) and v.strip().lower() in _UNKNOWN_STRINGS:
        return None
    return v


def _coerce_int(v: object) -> object:
    """
    Coerce a string digit to ``int`` before type validation.

    Parameters
    ----------
    v : object
        Raw value, typically a string read from a CSV cell.

    Returns
    -------
    object
        An ``int`` if the value is a valid integer string, otherwise the
        original value (Pydantic will raise a type error downstream).
    """
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return v
    return v


def _coerce_float(v: object) -> object:
    """
    Coerce a string to ``float`` before type validation.

    Handles both ``.`` and ``,`` as decimal separators (common in
    European CSV exports).

    Parameters
    ----------
    v : object
        Raw value, typically a string read from a CSV cell.

    Returns
    -------
    object
        A ``float`` if the value is a valid decimal string, otherwise the
        original value (Pydantic will raise a type error downstream).
    """
    if isinstance(v, str):
        try:
            return float(v.strip().replace(",", "."))
        except ValueError:
            return v
    return v


# Annotated type aliases used in the model
_NullableStr = Annotated[str | None, BeforeValidator(_none_if_null)]
_NullableFloat = Annotated[float | None, BeforeValidator(_none_if_null), BeforeValidator(_coerce_float)]
_MandatoryFloat = Annotated[float, BeforeValidator(_coerce_float)]

# ---------------------------------------------------------------------------
# Pydantic schema — single source of truth for the glazing data format
# ---------------------------------------------------------------------------

_PaneComposition = Literal["monolithic", "laminated"]
_GlassType = Literal["annealed", "heat-strengthened", "fully-tempered", "unknown"]


class GlazingRecord(BaseModel):
    """
    Pydantic model representing one row of glazing data.

    All CSV values arrive as strings; ``BeforeValidator`` functions handle
    type coercion and null normalization before Pydantic applies its own
    type and constraint checks.

    Attributes
    ----------
    building_id : str
        Primary key (inherited). Unique building identifier in
        ``LOCATION_BUILDING`` format (e.g. ``AMSTERDAM_RIJKSMUSEUM``). Mandatory.
    batch_id : str
        Primary key (inherited). Batch identifier in ``BATCH-##`` format
        (e.g. ``BATCH-01``). Mandatory.
    unit_id : str
        Secondary key. Glazing unit identifier in ``UNIT-###`` format
        (e.g. ``UNIT-001``). Unique within each ``building_id``. Mandatory.
    unit_width : float
        Nominal width of the glazing unit in millimetres. Positive, mandatory.
    unit_height : float
        Nominal height of the glazing unit in millimetres. Positive, mandatory.
    number_of_panes : int
        Number of panes in the original glazing unit (1, 2, or 3). Mandatory.
    pane_composition_outer : {"monolithic", "laminated"}
        Composition of the outer (or only) pane. Mandatory.
    pane_composition_inner : {"monolithic", "laminated"} or None
        Composition of the inner pane. Required when ``number_of_panes >= 2``.
    pane_composition_middle : {"monolithic", "laminated"} or None
        Composition of the middle pane. Required when ``number_of_panes == 3``.
    pane_glass_type_outer : {"annealed", "heat-strengthened", "fully-tempered", "unknown"}
        Glass type of the outer (or only) pane. Mandatory.
    pane_glass_type_inner : {"annealed", "heat-strengthened", "fully-tempered", "unknown"} or None
        Glass type of the inner pane. Required when ``number_of_panes >= 2``.
    pane_glass_type_middle : {"annealed", "heat-strengthened", "fully-tempered", "unknown"} or None
        Glass type of the middle pane. Required when ``number_of_panes == 3``.
    pane_nominal_thickness_outer : float
        Nominal thickness of the outer (or only) pane in millimetres. Positive, mandatory.
    pane_nominal_thickness_inner : float or None
        Nominal thickness of the inner pane in millimetres. Required when
        ``number_of_panes >= 2``; ``None`` (or ``"n/a"``) for single panes.
    pane_nominal_thickness_middle : float or None
        Nominal thickness of the middle pane in millimetres. Required when
        ``number_of_panes == 3``; ``None`` (or ``"n/a"``) otherwise.
    spacer_thickness : float or None
        Width of the spacer in millimetres. Non-negative when provided.
        Expected for IGUs; ``None`` (or ``"n/a"``) for single panes.
    spacer_material : str
        Material of the spacer (e.g. ``"aluminium"``). Use ``"unknown"``
        if not available, ``"n/a"`` for single panes. Mandatory.
    spacer_code : str
        Code printed on the spacer (e.g. ``"ZFG SANCO 4/82"``). Use
        ``"unknown"`` if not available, ``"n/a"`` for single panes. Mandatory.
    surface_coating : str or None
        Surface coating description (e.g. ``"low-e coating"``,
        ``"no-coating"``, ``"unknown"``). Optional.
    notes : str or None
        Free-text observations, anomalies, or special handling notes. Optional.
    """

    model_config = {"str_strip_whitespace": True}

    # <<< BEGIN GENERATED FIELDS — run scripts/generate_validators.py to update >>>
    building_id: str
    batch_id: str
    unit_id: str
    unit_width: _MandatoryFloat
    unit_height: _MandatoryFloat
    number_of_panes: Annotated[int, BeforeValidator(_coerce_int)]
    pane_composition_outer: Literal["monolithic", "laminated"]
    pane_composition_inner: Annotated[
        Literal["monolithic", "laminated"] | None, BeforeValidator(_none_if_null)
    ] = None
    pane_composition_middle: Annotated[
        Literal["monolithic", "laminated"] | None, BeforeValidator(_none_if_null)
    ] = None
    pane_glass_type_outer: Literal["annealed", "heat-strengthened", "fully-tempered", "unknown"]
    pane_glass_type_inner: Annotated[
        Literal["annealed", "heat-strengthened", "fully-tempered", "unknown"] | None, BeforeValidator(_none_if_null)
    ] = None
    pane_glass_type_middle: Annotated[
        Literal["annealed", "heat-strengthened", "fully-tempered", "unknown"] | None, BeforeValidator(_none_if_null)
    ] = None
    pane_nominal_thickness_outer: _MandatoryFloat
    pane_nominal_thickness_inner: _NullableFloat = None
    pane_nominal_thickness_middle: _NullableFloat = None
    spacer_thickness: _NullableFloat = None
    spacer_material: str
    spacer_code: str
    surface_coating: _NullableStr = None
    notes: _NullableStr = None

    @field_validator("building_id")
    @classmethod
    def check_building_id_format(cls, v: str) -> str:
        if not _BUILDING_ID_PATTERN.match(v):
            raise ValueError(
                f"building_id '{v}' does not match required format LOCATION_BUILDING (e.g. AMSTERDAM_RIJKSMUSEUM)."
            )
        return v

    @field_validator("batch_id")
    @classmethod
    def check_batch_id_format(cls, v: str) -> str:
        if not _BATCH_ID_PATTERN.match(v):
            raise ValueError(
                f"batch_id '{v}' does not match required format BATCH-## (e.g. BATCH-01)."
            )
        return v

    @field_validator("unit_id")
    @classmethod
    def check_unit_id_format(cls, v: str) -> str:
        if not _UNIT_ID_PATTERN.match(v):
            raise ValueError(
                f"unit_id '{v}' does not match required format UNIT-### (e.g. UNIT-001)."
            )
        return v

    @field_validator("unit_width", "unit_height", "pane_nominal_thickness_outer", mode="after")
    @classmethod
    def check_positive_required(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be positive (> 0), got {v}.")
        return v
    @field_validator("pane_nominal_thickness_inner", "pane_nominal_thickness_middle", mode="after")
    @classmethod
    def check_positive_optional(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"Value must be positive (> 0) when provided, got {v}.")
        return v
    @field_validator("number_of_panes", mode="after")
    @classmethod
    def check_number_of_panes_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"number_of_panes must be a positive integer, got {v}.")
        return v

    @field_validator("spacer_thickness", mode="after")
    @classmethod
    def check_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"Value must be non-negative, got {v}.")
        return v
    # <<< END GENERATED FIELDS >>>

    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_pane_fields_consistency(self) -> "GlazingRecord":
        """
        Validate that inner and middle pane fields are provided according
        to ``number_of_panes``.

        - For IGUs (``number_of_panes >= 2``): inner pane composition, glass
          type, and nominal thickness must all be filled.
        - For triple IGUs (``number_of_panes >= 3``): middle pane fields must
          additionally be filled.
        """
        n = self.number_of_panes

        if n >= 2:
            missing = []
            if self.pane_composition_inner is None:
                missing.append("pane_composition_inner")
            if self.pane_glass_type_inner is None:
                missing.append("pane_glass_type_inner")
            if self.pane_nominal_thickness_inner is None:
                missing.append("pane_nominal_thickness_inner")
            if missing:
                raise ValueError(
                    f"number_of_panes is {n} but inner pane fields are missing: "
                    f"{', '.join(missing)}. Fill these fields or use 'n/a' only "
                    f"for single-pane units."
                )

        if n >= 3:
            missing = []
            if self.pane_composition_middle is None:
                missing.append("pane_composition_middle")
            if self.pane_glass_type_middle is None:
                missing.append("pane_glass_type_middle")
            if self.pane_nominal_thickness_middle is None:
                missing.append("pane_nominal_thickness_middle")
            if missing:
                raise ValueError(
                    f"number_of_panes is {n} but middle pane fields are missing: "
                    f"{', '.join(missing)}. Fill these fields or use 'n/a' only "
                    f"for single and double-pane units."
                )

        return self


# ---------------------------------------------------------------------------
# Reporting: domain-specific error container
# ---------------------------------------------------------------------------

class GlazingDataError(Exception):
    """
    Validation issue found in a glazing data file.

    Inherits from ``Exception`` so instances can be raised and caught
    with standard Python error handling if needed.

    Attributes
    ----------
    row : int or None
        1-based index of the data row where the issue was found.
        ``None`` indicates a file-level issue (e.g. missing column).
    field : str
        Name of the column associated with the issue.
    message : str
        Human-readable description of the issue.
    severity : {"ERROR", "WARNING"}
        ``"ERROR"`` blocks a file from passing validation;
        ``"WARNING"`` is informational only.
    """

    def __init__(
        self, row: int | None, field: str, message: str, severity: str = "ERROR"
    ):
        """
        Parameters
        ----------
        row : int or None
            1-based data row index, or ``None`` for file-level issues.
        field : str
            Name of the column associated with the issue.
        message : str
            Human-readable description of the issue.
        severity : {"ERROR", "WARNING"}, optional
            Issue severity level. Default is ``"ERROR"``.
        """
        self.row = row
        self.field = field
        self.message = message
        self.severity = severity
        super().__init__(str(self))

    def __str__(self) -> str:
        row_part = f"row {self.row}" if self.row is not None else "file level"
        return f"[{self.severity}] In column '{self.field}' at {row_part}: {self.message}"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _required_columns_from_model() -> set[str]:
    """
    Derive the set of required column names from the ``GlazingRecord`` model.

    Returns
    -------
    set of str
        All field names declared on ``GlazingRecord``.
    """
    return set(GlazingRecord.model_fields.keys())


def validate_record(row_index: int, row: dict[str, str]) -> list[GlazingDataError]:
    """
    Validate a single CSV row against the ``GlazingRecord`` Pydantic model.

    Attempts to construct a ``GlazingRecord`` from the row dict. Any
    ``ValidationError`` raised by Pydantic is unpacked into a list of
    ``GlazingDataError`` instances, one per failing field or constraint.
    Additional warning-level checks are appended afterwards.

    Parameters
    ----------
    row_index : int
        1-based index of the data row within the file (used for reporting).
    row : dict of {str: str}
        Mapping of column names to raw string cell values for one CSV row.

    Returns
    -------
    list of GlazingDataError
        All errors and warnings found in this row. Empty if the row is valid.
    """
    errors: list[GlazingDataError] = []

    try:
        record = GlazingRecord(**row)
    except PydanticValidationError as exc:
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "unknown"
            errors.append(GlazingDataError(row_index, field, err["msg"]))
        return errors  # skip warning checks if the row already has errors

    # ------------------------------------------------------------------
    # Warning-level checks (cannot be expressed as Pydantic errors)
    # ------------------------------------------------------------------

    # Inner/middle fields filled for a single-pane unit
    if record.number_of_panes == 1:
        unexpected = []
        if record.pane_composition_inner is not None:
            unexpected.append("pane_composition_inner")
        if record.pane_glass_type_inner is not None:
            unexpected.append("pane_glass_type_inner")
        if record.pane_nominal_thickness_inner is not None:
            unexpected.append("pane_nominal_thickness_inner")
        if record.pane_composition_middle is not None:
            unexpected.append("pane_composition_middle")
        if record.pane_glass_type_middle is not None:
            unexpected.append("pane_glass_type_middle")
        if record.pane_nominal_thickness_middle is not None:
            unexpected.append("pane_nominal_thickness_middle")
        if unexpected:
            errors.append(GlazingDataError(
                row_index,
                "pane fields",
                f"number_of_panes is 1 but inner/middle pane fields are filled: "
                f"{', '.join(unexpected)}. These should be 'n/a' for single-pane units.",
                severity="WARNING",
            ))

    # Spacer info missing for a multi-pane unit
    if record.number_of_panes >= 2 and record.spacer_thickness is None:
        errors.append(GlazingDataError(
            row_index,
            "spacer_thickness",
            f"number_of_panes is {record.number_of_panes} but spacer_thickness is "
            f"missing. Provide the spacer width in millimetres.",
            severity="WARNING",
        ))

    return errors


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a glazing data CSV file into a pandas DataFrame.

    Supports both comma-separated and semicolon-separated files. The
    delimiter is detected automatically with pandas.

    Encoding is detected automatically: UTF-8 (with optional BOM) is
    tried first, falling back to cp1252 for files saved by Windows tools.

    All cell values are stored as stripped strings so that downstream
    validation functions receive a consistent type regardless of the
    original column dtype.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per data record. All values are strings;
        empty cells and genuine ``NaN`` values become ``""``. Column
        names are stripped of surrounding whitespace.
    """
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(
                filepath,
                sep=None,               # auto-detect delimiter
                engine="python",        # required for sep=None
                encoding=encoding,
                dtype=str,              # read all columns as strings
                keep_default_na=False,  # preserve "NA" strings as-is
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not decode {filepath} with any supported encoding (utf-8, cp1252)."
        )

    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else str(v).strip()))
    return df


# ---------------------------------------------------------------------------
# File-level validator
# ---------------------------------------------------------------------------

def validate_file(filepath: Path) -> bool:
    """
    Validate a single glazing data CSV file and print a report to stdout.

    Checks that all required columns are present, then validates each row
    against the ``GlazingRecord`` Pydantic model via ``validate_record``.
    Duplicate ``building_id`` + ``unit_id`` combinations are checked across
    all rows after per-row validation. Results are printed followed by a
    summary line.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the CSV file to validate.

    Returns
    -------
    bool
        ``True`` if the file contains no ``ERROR``-severity issues.
        ``WARNING``-only files still return ``True``.
    """
    print(f"\n{'=' * 60}")
    print(f"Validating: {filepath}")
    print(f"{'=' * 60}")

    try:
        df = load_csv(filepath)
    except Exception as exc:
        print(f"[ERROR] File | Could not read file: {exc}")
        return False

    all_errors: list[GlazingDataError] = []

    # Check 1 – required columns present
    required = _required_columns_from_model()
    missing = required - set(df.columns)
    for col in sorted(missing):
        all_errors.append(GlazingDataError(
            None, col, f"Required column '{col}' is missing from the file."
        ))

    rows = df.to_dict(orient="records")

    # Per-row validation via Pydantic
    for idx, row in enumerate(rows, start=1):
        all_errors.extend(validate_record(idx, row))

    # Check 13 – duplicate building_id + unit_id combinations
    # unit_id must be unique within each building_id (README: "Unique within each building_id")
    seen_ids: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        bid = str(row.get("building_id", "")).strip()
        uid = str(row.get("unit_id", "")).strip()
        combo = f"{bid}|{uid}"
        if bid and uid and bid.lower() not in _UNKNOWN_STRINGS:
            if combo in seen_ids:
                all_errors.append(GlazingDataError(
                    idx, "building_id + unit_id",
                    f"Duplicate combination building_id='{bid}', unit_id='{uid}' "
                    f"(first seen at row {seen_ids[combo]})."
                ))
            else:
                seen_ids[combo] = idx

    # Print results
    errors = [e for e in all_errors if e.severity == "ERROR"]
    warnings = [e for e in all_errors if e.severity == "WARNING"]

    if not all_errors:
        print(f"All checks passed. ({len(rows)} row(s) validated)")
        return True

    for e in sorted(all_errors, key=lambda x: (x.row or 0, x.severity)):
        print(e)

    print(
        f"\nSummary: {len(rows)} row(s) | "
        f"{len(errors)} error(s) | {len(warnings)} warning(s)"
    )
    return len(errors) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_files(target: str) -> list[Path]:
    """
    Resolve a CLI argument to a list of CSV file paths.

    Parameters
    ----------
    target : str
        A path to a single CSV file, a directory, or a glob pattern.

    Returns
    -------
    list of pathlib.Path
        Sorted list of CSV file paths matching the target. Returns an
        empty list if no matches are found.
    """
    p = Path(target)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    return sorted(p.parent.glob(p.name))


def main() -> None:
    """
    Entry point for the glazing data validation CLI.

    Resolves CSV file paths from command-line arguments (or the default
    data folder), validates each file, and exits with code ``0`` if all
    files pass or ``1`` if any file contains at least one ``ERROR``.
    """
    default_data_dir = (
        Path(__file__).resolve().parent.parent / "data" / "02_glazing_data"
    )

    targets = sys.argv[1:] if len(sys.argv) > 1 else [str(default_data_dir)]

    files: list[Path] = []
    for t in targets:
        found = collect_files(t)
        if not found:
            print(f"[WARNING] No CSV files found for: {t}")
        files.extend(found)

    if not files:
        print("No CSV files to validate. Exiting.")
        sys.exit(1)

    overall_pass = True
    for f in files:
        passed = validate_file(f)
        if not passed:
            overall_pass = False

    print(f"\n{'=' * 60}")
    if overall_pass:
        print(f"All {len(files)} file(s) passed validation.")
    else:
        print("Validation failed for one or more files. See details above.")
    print(f"{'=' * 60}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
