"""
从 data/EEG_features 与 data/save_info 打包网页上传用 zip（EEF_features/ + save_info/）。

用法（在项目根目录）:
  python scripts/build_sample_upload_zips.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def save_info_files_for_subject(save_info_dir: Path, subject: int) -> list[Path]:
    files = sorted(save_info_dir.glob(f"{subject}_*_save_info.csv"))
    if len(files) != 4:
        raise ValueError(f"subject {subject}: expected 4 save_info csv, got {len(files)}: {files}")
    return files


def add_subject_to_zip(
    zf: zipfile.ZipFile,
    subject: int,
    eeg_features_dir: Path,
    save_info_dir: Path,
) -> None:
    mat = eeg_features_dir / f"{subject}.mat"
    if not mat.is_file():
        raise FileNotFoundError(mat)
    arcname = f"EEF_features/{subject}.mat"
    zf.write(mat, arcname)
    for csv_path in save_info_files_for_subject(save_info_dir, subject):
        zf.write(csv_path, f"save_info/{csv_path.name}")


def build_zip(
    out_path: Path,
    subjects: list[int],
    eeg_features_dir: Path,
    save_info_dir: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for s in subjects:
            add_subject_to_zip(zf, s, eeg_features_dir, save_info_dir)
    print(f"Wrote {out_path} ({len(subjects)} subject(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sample/*.zip for docs upload demo.")
    parser.add_argument("--eeg-features-dir", type=Path, default=Path("data/EEG_features"))
    parser.add_argument("--save-info-dir", type=Path, default=Path("data/save_info"))
    parser.add_argument("--out-dir", type=Path, default=Path("Sample"))
    args = parser.parse_args()

    presets: list[tuple[str, list[int]]] = [
        ("sample_upload_subject1.zip", [1]),
        ("sample_upload_subject2.zip", [2]),
        ("sample_upload_subject10.zip", [10]),
        ("sample_upload_subjects_2_10.zip", [2, 10]),
    ]

    for name, subjects in presets:
        build_zip(
            args.out_dir / name,
            subjects,
            args.eeg_features_dir,
            args.save_info_dir,
        )


if __name__ == "__main__":
    main()
