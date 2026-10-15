"""
Validates test data CSV files against the OpenGLASSLab data schema.

The schema is defined as a Pydantic model (``TestRecord``).
Each CSV row is validated by attempting to construct a ``TestRecord`` from it;
Pydantic's ``ValidationError`` is then translated into ``TestDataError`` instances
for reporting.

Checks performed
----------------
1.  Required columns exist (derived from the model's field set)
2.  No nulls in mandatory fields
3.  Allowed values for categorical fields (``surface_id``, ``prestress_state``, etc.)
4.  ``building_id`` follows the ``LOCATION_BUILDING`` format (e.g. ``AMSTERDAM_RIJKSMUSEUM``)
5.  ``batch_id`` follows the ``BATCH-##`` format (e.g. ``BATCH-01``)
6.  ``unit_id`` follows the ``UNIT-###`` format (e.g. ``UNIT-001``)
7.  ``specimen_id`` follows the ``SPEC-###`` format (e.g. ``SPEC-001``)
8.  ``date_time`` follows the ``YYYY-MM-DD hh:mm:ss`` format
9.  Non-negative/positive checks for numeric fields (thicknesses, ring diameters,
    rates, loads, fracture stresses)
10. Duplicate ``building_id`` + ``batch_id`` + ``unit_id`` + ``specimen_id``
    combination detection

Notes
-----
Both comma-separated and semicolon-separated CSV files are supported.
The delimiter is detected automatically by pandas.
Decimal values may use either ``.`` or ``,`` as the decimal separator.

Examples
--------
Validate all CSVs in the default data folder::

    python validate_test_data.py

Validate a single file::

    python validate_test_data.py path/to/T_AMSTERDAM_CLEYNDERTWEG.csv

Validate all CSVs in a given folder::

    python validate_test_data.py path/to/folder/
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
_SPECIMEN_ID_PATTERN = re.compile(r"^SPEC-\d{3,}$")
_DATE_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
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
# Pydantic schema — single source of truth for the test data format
# ---------------------------------------------------------------------------

class TestRecord(BaseModel):
    """
    Pydantic model representing one row of test data.

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
        Primary key (inherited). Glazing unit identifier in ``UNIT-###``
        format. Unique within each ``building_id`` + ``batch_id`` combination.
        Mandatory.
    specimen_id : str
        Secondary key. Specimen identifier in ``SPEC-###`` format
        (e.g. ``SPEC-001``). Unique within each
        ``building_id`` + ``batch_id`` + ``unit_id`` combination. Mandatory.
    specimen_pretreatment : str or None
        Free-text description of any treatment applied before testing. Optional.
    specimen_predetection : str or None
        Whether scratches or surface defects were detected prior to testing. Optional.
    surface_id : {"S1", "S2", "S3", "S4", "S5", "S6"}
        Surface tested, numbered from outside inward. Mandatory.
    air_or_tin : {"air", "tin", "unknown", "n/a"}
        Air-side or tin-side of float glass; ``"n/a"`` for non-float glass. Mandatory.
    prestress_state : {"annealed", "heat-strengthened", "fully-tempered", "unknown"}
        Classification of the specimen's prestress condition. Mandatory.
    prestress_measured : float or None
        Measured surface prestress in MPa (negative = compression). Optional.
    prestress_device : str or None
        Device/instrument/method used to measure surface prestress. Optional.
    thickness_nominal : float
        Nominal thickness of the tested glass specimen, in mm. Non-negative, mandatory.
    thickness_measured : float
        Measured thickness of the glass specimen, in mm. Non-negative, mandatory.
    specimen_code : str or None
        Internal specimen naming/coding convention used by the lab. Optional.
    diameter_supportring : float
        Diameter of the support ring, in mm. Non-negative, mandatory.
    diameter_loadring : float
        Diameter of the load ring, in mm. Non-negative, mandatory.
    rings_standard : {"ASTM", "CEN", "none"}
        Whether the ring diameters comply with ASTM, CEN, or neither. Mandatory.
    target_stressrate : float
        Intended stress rate during the test. Non-negative, mandatory.
    test_control : {"force-controlled-kns", "displacement-controlled-mms"}
        Whether the machine was steered by force or displacement. Mandatory.
    target_testrate : float
        Intended loading/displacement rate during the test. Non-negative, mandatory.
    fracture_location : str
        Location where fracture initiated. Mandatory.
    fracture_notes : str or None
        Free-text observations related to the fracture event. Optional.
    fracture_load : float
        Maximum applied load recorded at specimen fracture, in kN. Non-negative, mandatory.
    fracture_time : float
        Time elapsed from test start until specimen fracture, in seconds. Non-negative, mandatory.
    true_testrate : float
        Actual achieved loading-rate slope. Non-negative, mandatory.
    temperature : float or None
        Ambient or specimen temperature during testing, in degrees Celsius. Optional.
    humidity : float or None
        Relative humidity during testing, in percent. Non-negative when provided. Optional.
    date_time : str
        Timestamp of when the test was performed (``YYYY-MM-DD hh:mm:ss``). Mandatory.
    fracture_stress_effective : float or None
        Fracture stress accounting for load, geometry, and test setup dimensions.
        Positive when provided. Optional.
    fracture_stress_equivalent_5s : float or None
        Equivalent fracture stress normalized to 5 seconds. Positive when provided. Optional.
    fracture_stress_equivalent_60s : float or None
        Equivalent fracture stress normalized to 60 seconds. Positive when provided. Optional.
    """

    model_config = {"str_strip_whitespace": True}

    # <<< BEGIN GENERATED FIELDS — run scripts/generate_validators.py to update >>>
    building_id: str
    batch_id: str
    unit_id: str
    specimen_id: str
    specimen_pretreatment: _NullableStr = None
    specimen_predetection: Annotated[
        Literal["not inspected", "none detected", "detected - naked eye", "detected - microscope", "detected - scanner", "unknown"] | None, BeforeValidator(_none_if_null)
    ] = None
    surface_id: Literal["S1", "S2", "S3", "S4", "S5", "S6"]
    air_or_tin: Literal["air", "tin", "unknown", "n/a"]
    prestress_state: Literal["annealed", "heat-strengthened", "fully-tempered", "unknown"]
    prestress_measured: _NullableFloat = None
    prestress_device: _NullableStr = None
    specimen_shape: Literal["square", "circular"]
    specimen_edgelength: _NullableFloat = None
    specimen_diameter: _MandatoryFloat
    thickness_nominal: _MandatoryFloat
    thickness_measured: _MandatoryFloat
    specimen_code: _NullableStr = None
    diameter_supportring: _MandatoryFloat
    diameter_loadring: _MandatoryFloat
    rings_standard: Literal["ASTM", "CEN", "none"]
    target_stressrate: _MandatoryFloat
    machine_control: Literal["force-controlled-kns", "displacement-controlled-mms"]
    target_machinerate: _MandatoryFloat
    fracture_location: Literal["inside the load ring", "at the load ring", "outside the load ring", "at the support ring", "outside the support ring", "at the edge", "unknown"]
    fracture_notes: _NullableStr = None
    fracture_load: _MandatoryFloat
    fracture_time: _MandatoryFloat
    true_loadrate: _MandatoryFloat
    true_stressrate: _MandatoryFloat
    temperature: _NullableFloat = None
    humidity: _NullableFloat = None
    date_time: str
    fracture_stress: _MandatoryFloat
    fracture_stress_equivalent_5s: _NullableFloat = None
    fracture_stress_equivalent_60s: _NullableFloat = None

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

    @field_validator("specimen_id")
    @classmethod
    def check_specimen_id_format(cls, v: str) -> str:
        if not _SPECIMEN_ID_PATTERN.match(v):
            raise ValueError(
                f"specimen_id '{v}' does not match required format SPEC-### (e.g. SPEC-001)."
            )
        return v

    @field_validator("date_time")
    @classmethod
    def check_date_time_format(cls, v: str) -> str:
        if not _DATE_TIME_PATTERN.match(v):
            raise ValueError(
                f"date_time '{v}' does not match required format YYYY-MM-DD hh:mm:ss (e.g. 2026-10-23 10:03:35)."
            )
        return v

    @field_validator("fracture_stress", mode="after")
    @classmethod
    def check_positive_required(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Value must be positive (> 0), got {v}.")
        return v
    @field_validator("fracture_stress_equivalent_5s", "fracture_stress_equivalent_60s", mode="after")
    @classmethod
    def check_positive_optional(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"Value must be positive (> 0) when provided, got {v}.")
        return v

    @field_validator("specimen_edgelength", "specimen_diameter", "thickness_nominal", "thickness_measured", "diameter_supportring", "diameter_loadring", "target_stressrate", "target_machinerate", "fracture_load", "fracture_time", "true_loadrate", "true_stressrate", "humidity", mode="after")
    @classmethod
    def check_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"Value must be non-negative, got {v}.")
        return v
    # <<< END GENERATED FIELDS >>>


# ---------------------------------------------------------------------------
# Reporting: domain-specific error container
# ---------------------------------------------------------------------------

class TestDataError(Exception):
    """
    Validation issue found in a test data file.

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
    Derive the set of required column names from the ``TestRecord`` model.

    Returns
    -------
    set of str
        All field names declared on ``TestRecord``.
    """
    return set(TestRecord.model_fields.keys())


def validate_record(row_index: int, row: dict[str, str]) -> list[TestDataError]:
    """
    Validate a single CSV row against the ``TestRecord`` Pydantic model.

    Attempts to construct a ``TestRecord`` from the row dict. Any
    ``ValidationError`` raised by Pydantic is unpacked into a list of
    ``TestDataError`` instances, one per failing field or constraint.

    Parameters
    ----------
    row_index : int
        1-based index of the data row within the file (used for reporting).
    row : dict of {str: str}
        Mapping of column names to raw string cell values for one CSV row.

    Returns
    -------
    list of TestDataError
        All errors and warnings found in this row. Empty if the row is valid.
    """
    errors: list[TestDataError] = []

    try:
        TestRecord(**row)
    except PydanticValidationError as exc:
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "unknown"
            errors.append(TestDataError(row_index, field, err["msg"]))
        return errors

    return errors


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a test data CSV file into a pandas DataFrame.

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
    Validate a single test data CSV file and print a report to stdout.

    Checks that all required columns are present, then validates each row
    against the ``TestRecord`` Pydantic model via ``validate_record``.
    Duplicate ``building_id`` + ``batch_id`` + ``unit_id`` + ``specimen_id``
    combinations are checked across all rows after per-row validation.
    Results are printed followed by a summary line.

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

    all_errors: list[TestDataError] = []

    # Check 1 – required columns present
    required = _required_columns_from_model()
    missing = required - set(df.columns)
    for col in sorted(missing):
        all_errors.append(TestDataError(
            None, col, f"Required column '{col}' is missing from the file."
        ))

    rows = df.to_dict(orient="records")

    # Per-row validation via Pydantic
    for idx, row in enumerate(rows, start=1):
        all_errors.extend(validate_record(idx, row))

    # Check 10 – duplicate building_id + batch_id + unit_id + specimen_id combinations
    seen_ids: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        bid = str(row.get("building_id", "")).strip()
        batchid = str(row.get("batch_id", "")).strip()
        uid = str(row.get("unit_id", "")).strip()
        sid = str(row.get("specimen_id", "")).strip()
        combo = f"{bid}|{batchid}|{uid}|{sid}"
        if bid and batchid and uid and sid and bid.lower() not in _UNKNOWN_STRINGS:
            if combo in seen_ids:
                all_errors.append(TestDataError(
                    idx, "building_id + batch_id + unit_id + specimen_id",
                    f"Duplicate combination building_id='{bid}', batch_id='{batchid}', "
                    f"unit_id='{uid}', specimen_id='{sid}' (first seen at row {seen_ids[combo]})."
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
    Entry point for the test data validation CLI.

    Resolves CSV file paths from command-line arguments (or the default
    data folder), validates each file, and exits with code ``0`` if all
    files pass or ``1`` if any file contains at least one ``ERROR``.
    """
    default_data_dir = (
        Path(__file__).resolve().parent.parent / "data" / "03_test_data"
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
