import argparse
import os
import shutil


def main(src: list, dst: str, revision: str = "v2"):
    # Loop through each source directory
    for rev_dir in os.listdir(src):
        # Determine the destination storage
        destination_storage = None
        for storage in dst:
            if rev_dir in os.listdir(storage):
                destination_storage = storage
                break
        
        if not destination_storage:
            print(f"No matching storage found for {rev_dir}. Skipping.")
            continue

        destination_path = os.path.join(destination_storage, rev_dir, f"info_label_{revision}")
        os.makedirs(destination_path, exist_ok=True)
        
        # Copy the entire directory
        shutil.copytree(os.path.join(src, rev_dir), destination_path, dirs_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DPRT data preprocessing")

    default_src = [
        "/data/Samsung 8TB 1",
        "/data/Samsung 500GB",
        "/data/Samsung 8TB 2"
    ]
    
    parser.add_argument(
        "--src", type=str, default="/data/KRadar_refined_label_by_UWIPL", help="Paths to the refined labels."
    )
    parser.add_argument(
        "--dst", type=str, nargs="+", default=default_src, help="Path to the dataset"
    )
    args = parser.parse_args()

    main(src=args.src, dst=args.dst)
