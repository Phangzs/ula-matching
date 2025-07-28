import csv, os, pathlib, tempfile
from hash_ids import hash_student_id


def hash_column_in_csv(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    id_column: str = "student_id",
    encoding: str = "utf-8",
) -> pathlib.Path:
    """
    Hash every value in id_column and write the result.
    """

    # Create temp file if required
    in_path = pathlib.Path(in_path)
    if out_path is None:
        # create temp file in same dir so os.replace() can be atomic
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=in_path.parent, suffix=".tmp", text=True
        )
        os.close(tmp_fd)                     # we'll reopen with pathlib
        out_path = pathlib.Path(tmp_name)
        _replace_after = True
    else:
        out_path = pathlib.Path(out_path)
        _replace_after = False


    # For every file, replace ID column with hash
    with in_path.open(newline="", encoding=encoding) as fin, out_path.open("w", newline="", encoding=encoding) as fout:

        reader = csv.DictReader(fin)

        # force header read for Py ≤3.10
        first_row = None
        if reader.fieldnames is None:
            try:
                first_row = next(reader)
            except StopIteration:
                raise ValueError(f"{in_path} appears to be empty")

        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Column “{id_column}” not found in {in_path} "
                f"(headers: {reader.fieldnames})"
            )

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        if first_row is not None:
            first_row[id_column] = hash_student_id(first_row[id_column])
            writer.writerow(first_row)

        for row in reader:
            row[id_column] = hash_student_id(row[id_column])
            writer.writerow(row)

    # atomic replace if wrote to a temp file
    if _replace_after:
        os.replace(out_path, in_path)
        return in_path
    return out_path


def hash_all_csvs(
    folder: str | pathlib.Path,
    *,
    id_column: str = "ID",
    suffix: str = "_hashed",
    encoding: str = "utf-8",
) -> list[pathlib.Path]:
    """
    Apply hash_column_in_csv() to every "*.csv" in folder, outputting to a copy.

    Returns a list of the files that were written.
    """
    folder = pathlib.Path(folder)
    written: list[pathlib.Path] = []

    for src in sorted(folder.glob("*.csv")):
        dst = src.with_name(src.stem + suffix + src.suffix)
        hash_column_in_csv(src, dst, id_column=id_column, encoding=encoding)
        written.append(dst)

    return written
