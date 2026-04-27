from pathlib import Path
import numpy as np
import pandas as pd
import base64
import zlib

def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded


def create_results_csv() -> None:
    # paths
    cur_dir = Path(__file__).parent
    available_dir_in_parent_dir = cur_dir.parent
    pred_dir = Path(f"{available_dir_in_parent_dir}/results/")   # folder with pred_000000.npy, pred_000001.npy,
    out_csv = Path("../submission.csv")
    print(pred_dir)
    cur_dir = Path(__file__).parent
    available_dir_in_parent_dir = cur_dir.parent

    print(cur_dir)
    print(available_dir_in_parent_dir)

    if pred_dir.is_dir():
        print(f"Found predictions in {pred_dir}")



    rows = []
    pred_files = sorted(pred_dir.glob("test_*.npy"))

    for pred_path in pred_files:
        depth = np.load(pred_path)

        idx = pred_path.stem.split("_")[-1]
        img_id = f"test_{idx}_depth"

        encoded_depth = encode_depth(depth)

        rows.append({
            "id": img_id,
            "Depths": encoded_depth,
        })


    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(out_csv, index=False)

    print(f"Saved submission to {out_csv}")
    print(f"Number of predictions: {len(df)}")

if __name__ == "__main__":
    create_results_csv()
