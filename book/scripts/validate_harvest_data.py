"""
Validates harvest data CSV files against the OpenGLASSLab data schema.

The schema is defined as a Pydantic model (``HarvestRecord``).
Each CSV row is validated by attempting to construct a ``HarvestRecord`` from it;
Pydantic's ``ValidationError`` is then translated into ``HarvestDataError`` instances
for reporting.

Checks performed
----------------
1. Required columns exist (derived from the model's field set)
2. No nulls in mandatory fields
3. Allowed values for categorical fields (``building_use``, ``glazing_type``, etc.)
4. ``building_id`` follows the ``LOCATION_BUILDING`` format (e.g. ``AMSTERDAM_RIJKSMUSEUM``)
5. ``batch_id`` follows the ``BATCH-##`` format (e.g. ``BATCH-01``)
6. ``years_in_service`` equals ``year_of_harvesting - year_of_installation``
7. ``year_of_installation`` and ``year_of_harvesting`` are plausible 4-digit years
8. ``harvest_date`` is a valid ISO 8601 date (``YYYY-MM-DD``), if provided
9. Duplicate ``building_id`` + ``batch_id`` combination detection
10. At least one of ``year_of_installation`` or ``years_in_service`` is provided
    (the other is derivable from ``year_of_harvesting``, which is always mandatory)

Warnings (non-blocking)
-----------------------
- ``harvest_date`` is in the future
- ``years_in_service`` is blank but can be derived from the two year fields
- ``years_in_service`` contains a unit string (e.g. ``"55 years"``) — value is
extracted automatically but the cell should be corrected to a plain integer

Notes
-----
Both comma-separated and semicolon-separated CSV files are supported.
The delimiter is detected automatically by pandas.
TODO: consider switching to Excel input instead

Examples
--------
Validate all CSVs in the default data folder::

    python validate_harvest_data.py

Validate a single file::

    python validate_harvest_data.py path/to/H0001.csv

Validate all CSVs in a given folder::

    python validate_harvest_data.py path/to/folder/
"""

import re
import sys
from datetime import date
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
# NOTE: "unknown" is intentionally excluded — the README instructs researchers
# to write "unknown" in mandatory fields when the value cannot be determined.
# Nullable fields handle "unknown" explicitly via _none_if_null_or_unknown().
# "null" is included because the READMEs instruct researchers to write "NULL"
# (case-insensitive) in fields such as year_of_installation, years_in_service,
# floor_level, and harvest_date when the value is unknown.
_NULL_STRINGS: set[str] = {"na", "n/a", "-", "", "null"}

# Strings that mean "I don't know" and should map to None in nullable fields.
# "unknown" is kept separate so it can be accepted as a literal value in
# fields like facade_orientation that explicitly list it as an allowed value.
_UNKNOWN_STRINGS: set[str] = _NULL_STRINGS | {"unknown"}

# Reasonable year range for installations / harvesting
_MIN_YEAR: int = 1800
_MAX_YEAR: int = date.today().year  # cannot be in the future

# <<< BEGIN GENERATED CONSTANTS — run scripts/generate_validators.py to update >>>
_BUILDING_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9\-]*_[A-Z][A-Z0-9\-]*$")
_BATCH_ID_PATTERN = re.compile(r"^BATCH-\d{2,}$")
# <<< END GENERATED CONSTANTS >>>

# ---------------------------------------------------------------------------
# Shared pre-validators (BeforeValidator functions)
# ---------------------------------------------------------------------------

def _none_if_null(v: object) -> object:
    """
    Convert null-like strings to ``None`` before type validation.

    Uses ``_UNKNOWN_STRINGS`` so that both generic null markers (``"na"``,
    ``"n/a"``, ``"-"``, ``""``) and ``"unknown"`` are normalized to ``None``
    in nullable fields. Fields that accept ``"unknown"`` as a valid literal
    value (e.g. ``facade_orientation``) must not use this validator.

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


# Annotated type aliases used in the model
_NullableInt = Annotated[int | None, BeforeValidator(_none_if_null), BeforeValidator(_coerce_int)]
_NullableStr = Annotated[str | None, BeforeValidator(_none_if_null)]

# ---------------------------------------------------------------------------
# Pydantic schema — single source of truth for the harvest data format
# ---------------------------------------------------------------------------

class HarvestRecord(BaseModel):
    """
    Pydantic model representing one row of harvest data.

    All CSV values arrive as strings; ``BeforeValidator`` functions handle
    type coercion and null normalization before Pydantic applies its own
    type and constraint checks.

    Attributes
    ----------
    building_id : str
        Primary key. Unique building identifier in ``LOCATION_BUILDING`` format
        (e.g. ``AMSTERDAM_RIJKSMUSEUM``). Mandatory.
    batch_id : str
        Primary key. Batch identifier in ``BATCH-##`` format
        (e.g. ``BATCH-01``). Mandatory.
    building_country : str
        Country of the harvested building (e.g. ``"The Netherlands"``). Mandatory.
    building_city : str
        City of the harvested building (e.g. ``"Amsterdam"``). Mandatory.
    building_street : str or None
        Street of the harvested building. Optional.
    building_coordinates : str or None
        ISO 6709 coordinates of the building (e.g. ``"+52.3702+004.8952/"``).
        Optional.
    building_use : {"residential", "office", "industrial", "commercial", "institutional", "mixed-use"}
        Primary functional use of the building. Mandatory.
    year_of_installation : int or None
        Year the glazing was installed. ``None`` if unknown, but at least
        one of ``year_of_installation``/``years_in_service`` must be provided.
    year_of_harvesting : int
        Year the glazing was harvested. Mandatory.
    years_in_service : int or None
        Number of years the glazing was in service. If provided, must equal
        ``year_of_harvesting - year_of_installation``. Trailing text such as
        ``"55 years"`` is stripped automatically. At least one of
        ``year_of_installation``/``years_in_service`` must be provided, since
        the other can always be derived from ``year_of_harvesting``.
    glazing_type : {"single", "double", "triple", "other"}
        Type of glazing unit. Mandatory.
    building_element_type : {"window", "door", "roof/skylight", "partition wall", "facade/curtain wall", "floor", "balustrade", "canopy", "other"}
        Building component in which the glazing was installed. Mandatory.
    exposure_type : {"exterior", "interior"}
        Whether the glazing is part of the exterior envelope or an interior
        environment. Mandatory.
    facade_orientation : {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "horizontal", "unknown"}
        Cardinal direction the facade faces. Mandatory.
    floor_level : int or None
        Floor number (0 = ground floor, negative = below ground). Must be
        provided in every row; use ``NULL``/``unknown`` in the CSV if the
        floor level is genuinely not known.
    harvester : str or None
        Company or organization that performed the harvesting. Optional.
    harvest_date : date or None
        Date the glazing was harvested (ISO 8601 ``YYYY-MM-DD``). Optional.
    notes : str or None
        Free-text observations, anomalies, or special handling notes. Optional.
    """

    model_config = {"str_strip_whitespace": True}

    # <<< BEGIN GENERATED FIELDS — run scripts/generate_validators.py to update >>>
    building_id: str
    batch_id: str
    building_country: str
    building_city: str
    building_street: _NullableStr = None
    building_coordinates: _NullableStr = None
    building_use: Literal["residential", "office", "industrial", "commercial", "institutional", "mixed-use"]
    year_of_installation: _NullableInt = None
    year_of_harvesting: Annotated[int, BeforeValidator(_coerce_int)]
    years_in_service: _NullableInt = None
    glazing_type: Literal["single", "double", "triple", "other"]
    building_element_type: Literal["window", "door", "roof/skylight", "partition wall", "facade/curtain wall", "floor", "balustrade", "canopy", "other"]
    exposure_type: Literal["exterior", "interior"]
    facade_orientation: Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "horizontal", "unknown"]
    floor_level: _NullableInt = None
    harvester: _NullableStr = None
    harvest_date: Annotated[date | None, BeforeValidator(_none_if_null)] = None
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

    @field_validator("year_of_installation", "year_of_harvesting", mode="after")
    @classmethod
    def check_year_range(cls, v: int | None) -> int | None:
        if v is not None and not (_MIN_YEAR <= v <= _MAX_YEAR):
            raise ValueError(f"Year {v} is outside plausible range [{_MIN_YEAR}–{_MAX_YEAR}].")
        return v
    # <<< END GENERATED FIELDS >>>

    @model_validator(mode="after")
    def check_years_in_service(self) -> "HarvestRecord":
        """
        Validate that years_in_service = year_of_harvesting - year_of_installation,
        when both year fields are known, and that at least one of
        year_of_installation / years_in_service is provided.
        """
        yi = self.year_of_installation
        yh = self.year_of_harvesting
        yis = self.years_in_service

        if yi is None and yis is None:
            raise ValueError(
                "At least one of year_of_installation or years_in_service must be "
                "provided; the other can be derived from year_of_harvesting."
            )

        if yi is not None and yh is not None:
            if yh < yi:
                raise ValueError(
                    f"year_of_harvesting ({yh}) is before year_of_installation ({yi})."
                )
            expected = yh - yi
            if yis is not None and yis != expected:
                raise ValueError(
                    f"years_in_service ({yis}) does not match "
                    f"year_of_harvesting - year_of_installation "
                    f"({yh} - {yi} = {expected})."
                )
        return self


# ---------------------------------------------------------------------------
# Reporting: domain-specific error container
# ---------------------------------------------------------------------------

class HarvestDataError(Exception):
    """
    Validation issue found in a harvest data file.

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
    Derive the set of required column names from the ``HarvestRecord`` model.

    Returns
    -------
    set of str
        All field names declared on ``HarvestRecord``.
    """
    return set(HarvestRecord.model_fields.keys())


def validate_record(row_index: int, row: dict[str, str]) -> list[HarvestDataError]:
    """
    Validate a single CSV row against the ``HarvestRecord`` Pydantic model.

    Attempts to construct a ``HarvestRecord`` from the row dict. Any
    ``ValidationError`` raised by Pydantic is unpacked into a list of
    ``HarvestDataError`` instances, one per failing field or constraint.
    Additional warning-level checks (e.g. future ``harvest_date``,
    missing but derivable ``years_in_service``) are appended afterwards.

    Parameters
    ----------
    row_index : int
        1-based index of the data row within the file (used for reporting).
    row : dict of {str: str}
        Mapping of column names to raw string cell values for one CSV row.

    Returns
    -------
    list of HarvestDataError
        All errors and warnings found in this row. Empty if the row is valid.
    """
    errors: list[HarvestDataError] = []

    try:
        record = HarvestRecord(**row)
    except PydanticValidationError as exc:
        for err in exc.errors():
            # loc is a tuple of field names / indices; take the first element
            field = str(err["loc"][0]) if err["loc"] else "unknown"
            errors.append(HarvestDataError(row_index, field, err["msg"]))
        return errors  # skip warning checks if the row already has errors

    # ------------------------------------------------------------------
    # Warning-level checks (cannot be expressed as Pydantic errors)
    # ------------------------------------------------------------------

    # harvest_date in the future
    if record.harvest_date is not None and record.harvest_date > date.today():
        errors.append(HarvestDataError(
            row_index, "harvest_date",
            f"harvest_date '{record.harvest_date}' is in the future.",
            severity="WARNING",
        ))

    # years_in_service is blank but can be derived
    if (
        record.years_in_service is None
        and record.year_of_installation is not None
    ):
        derived = record.year_of_harvesting - record.year_of_installation
        errors.append(HarvestDataError(
            row_index, "years_in_service",
            f"Field is empty but can be derived: "
            f"{record.year_of_harvesting} - {record.year_of_installation} = {derived}.",
            severity="WARNING",
        ))

    return errors


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a harvest data CSV file into a pandas DataFrame.

    Supports both comma-separated and semicolon-separated files. The
    delimiter is detected automatically with pandas.
    TODO: switch to Excel spreadsheets in the future

    Encoding is detected automatically: UTF-8 (with optional BOM) is
    tried first, falling back to cp1252 for files saved by Windows tools
    that encode characters such as ``°`` (degree sign) as Latin-1 bytes.

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
    Validate a single harvest data CSV file and print a report to stdout.

    Checks that all required columns are present, then validates each row
    against the ``HarvestRecord`` Pydantic model via ``validate_record``.
    Duplicate ``harvesting_id`` values are checked across all rows after
    per-row validation. Results are printed followed by a summary line.

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

    all_errors: list[HarvestDataError] = []

    # Check 1 – required columns present
    required = _required_columns_from_model()
    missing = required - set(df.columns)
    for col in sorted(missing):
        all_errors.append(HarvestDataError(
            None, col, f"Required column '{col}' is missing from the file."
        ))

    rows = df.to_dict(orient="records")

    # Per-row validation via Pydantic
    for idx, row in enumerate(rows, start=1):
        all_errors.extend(validate_record(idx, row))

    # Check 9 – duplicate building_id + batch_id combinations
    seen_ids: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        bid = str(row.get("building_id", "")).strip()
        batchid = str(row.get("batch_id", "")).strip()
        combo = f"{bid}|{batchid}"
        if bid and batchid and bid.lower() not in _UNKNOWN_STRINGS:
            if combo in seen_ids:
                all_errors.append(HarvestDataError(
                    idx, "building_id + batch_id",
                    f"Duplicate combination building_id='{bid}', batch_id='{batchid}' "
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
    Entry point for the harvest data validation CLI.

    Resolves CSV file paths from command-line arguments (or the default
    data folder), validates each file, and exits with code ``0`` if all
    files pass or ``1`` if any file contains at least one ``ERROR``.
    """
    default_data_dir = (
        Path(__file__).resolve().parent.parent / "data" / "01_harvest_data"
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
