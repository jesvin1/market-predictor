from nse_prediction.db import bootstrap_storage


def main():
    db_path, layout = bootstrap_storage()

    print("Market storage is ready.")
    print(f"DuckDB file: {db_path}")
    print("Folders:")
    for label, path in layout.items():
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main()
