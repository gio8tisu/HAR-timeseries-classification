import argparse
import csv
import random
import os
import pathlib
import shutil


random.seed(1)


def main(args):
    # Create list with subjects IDs.
    if args.metadata:
        # Read IDs from metadata file.
        subjects = subjects_from_metadata(args.metadata)
    else:
        # Assume subjects from 1 to 24 (inclusive).
        subjects = list(range(1, 25))

    # Shuffle subjects list to divide into train and test subjects.
    random.shuffle(subjects)

    # Define subjects for train and test.
    train_n_subjects = int(args.ratio * len(subjects))
    train_subjects = subjects[:train_n_subjects]
    test_subjects = subjects[train_n_subjects:]
    if args.verbose:
        print("Number of training subjects:", len(train_subjects))
        print("Number of testing subjects:", len(test_subjects))
        print("Training subjects:", train_subjects)
        print("Testing subjects:", test_subjects)

    data_root = pathlib.Path(args.data_root)
    # Create train and test subdirectories.
    train_dir = data_root / pathlib.Path('train')
    test_dir = data_root / pathlib.Path('test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    if args.verbose:
        print(f"Created {train_dir} and {test_dir} directories")

    # Iterate over data_root directories.
    for root, dirs, files in os.walk(args.data_root):
        root = pathlib.Path(root)
        # Skip unwanted directories.
        if (root == data_root or root == train_dir or root == test_dir
                or train_dir in root.parents or test_dir in root.parents):
            continue

        # Create corresponding subdirectory on test and train directories.
        train_target_dir = train_dir / root.name
        test_target_dir = test_dir / root.name
        os.mkdir(train_target_dir)
        os.mkdir(test_target_dir)
        if args.verbose:
            print(f"Created {train_target_dir} and {test_target_dir} directories")

        # Copy files inside subdirectory.
        for file_ in files:
            from_file = root / pathlib.Path(file_)
            try:
                # Infer subject ID from filename.
                subject = from_file.stem.split("_")[1]
                # Define target file depending on subject ID.
                if subject in train_subjects:
                    to_file = train_target_dir / from_file.name
                elif subject in test_subjects:
                    to_file = test_target_dir / from_file.name
                else:
                    raise ValueError
                if args.verbose:
                    print(f"Copying {from_file} to {to_file}")
                shutil.copy(from_file, to_file)
            except (IndexError, ValueError):
                if args.verbose:
                    print(f"Skipping {from_file}")

    if args.verbose:
        print("Done.")


def subjects_from_metadata(metadata_file):
    subjects = []
    with open(metadata_file, newline="") as metadata:
        csv_reader = csv.reader(metadata)
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            subjects.append(row[0])
    return subjects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--ratio", default=0.8, help="train/(train + test) ratio")
    parser.add_argument("--metadata", help="metadata file path")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
