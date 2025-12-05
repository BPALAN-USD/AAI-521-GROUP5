import json
import os
import random
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import splitfolders
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
KAGGLE_JSON_LOCAL = BASE_DIR / "kaggle.json"
DATA_DIR = BASE_DIR / "statefarm_data"
DOWNLOAD_DIR = BASE_DIR / "downloads"
DEFAULT_EPOCHS = 10
COMPETITION_NAME = "state-farm-distracted-driver-detection"


CLASS_LABELS = {
    0: "normal driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "hair and makeup",
    9: "talking to passenger",
}


def check_numpy_available() -> bool:
    """
    Ensure NumPy is importable in the current environment and show its version.
    This helps debug environment mismatches where YOLO reports 'Numpy is not available'.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - environment-specific
        st.error(
            "NumPy is not available in the Python environment that is running this app.\n\n"
            "Please install it in the same environment, for example:\n"
            "`pip install numpy` (or `pip install -r requirements.txt`)."
        )
        st.exception(e)
        return False

    st.info(f"NumPy detected. Version: {np.__version__}")
    return True


def ensure_kaggle_credentials() -> bool:
    """
    Ensure Kaggle credentials are configured from the bundled kaggle.json file.
    No upload UI is required – this uses the local file directly.
    """
    if not KAGGLE_JSON_LOCAL.exists():
        st.error(
            f"`kaggle.json` was not found next to this app at: {KAGGLE_JSON_LOCAL}\n\n"
            "Please place your Kaggle API token file there and restart the app."
        )
        return False

    try:
        with open(KAGGLE_JSON_LOCAL, "r") as f:
            creds = json.load(f)
    except Exception as e:
        st.error(f"Failed to read `kaggle.json`: {e}")
        return False

    username = creds.get("username")
    key = creds.get("key")
    if not username or not key:
        st.error("`kaggle.json` does not contain valid `username` and `key` fields.")
        return False

    # Export as environment variables for the Kaggle CLI.
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    # Also install to ~/.kaggle for tools that expect it.
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_target = kaggle_dir / "kaggle.json"
    try:
        shutil.copy(KAGGLE_JSON_LOCAL, kaggle_json_target)
        kaggle_json_target.chmod(0o600)
    except Exception as e:
        st.warning(f"Could not copy kaggle.json to ~/.kaggle: {e}")

    return True


def download_competition_dataset():
    """
    Download the State Farm distracted driver dataset via Kaggle CLI.
    """
    if not ensure_kaggle_credentials():
        return False

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / f"{COMPETITION_NAME}.zip"

    if zip_path.exists():
        st.info("Dataset zip already downloaded. Skipping re-download.")
        return True

    status = st.status("Starting Kaggle download…", expanded=True)
    status.write(f"Using downloads directory: `{DOWNLOAD_DIR}`")
    status.write(
        "Calling Kaggle CLI: "
        f"`kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_DIR}`"
    )

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION_NAME,
        "-p",
        str(DOWNLOAD_DIR),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        status.update(state="error", label="Kaggle CLI not found")
        st.error(
            "The `kaggle` CLI was not found.\n\n"
            "Install it with `pip install kaggle` and make sure it is on your PATH."
        )
        return False

    if result.returncode != 0:
        status.update(state="error", label="Kaggle download failed")
        st.error(f"Kaggle download failed:\n\n{result.stderr}")
        if result.stdout:
            st.code(result.stdout, language="bash")
        return False

    status.update(state="complete", label="Kaggle download finished")
    if result.stdout:
        st.code(result.stdout, language="bash")

    # Show resulting file size if present
    if zip_path.exists():
        size_gb = zip_path.stat().st_size / (1024**3)
        st.info(f"Downloaded zip size: {size_gb:.2f} GB at `{zip_path}`")

    st.success("Dataset zip downloaded successfully.")
    return True


def extract_dataset():
    """
    Unzip the main competition archive and any nested zip files into DATA_DIR.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / f"{COMPETITION_NAME}.zip"

    if not zip_path.exists():
        st.error("Dataset zip file not found. Please download it first.")
        return False

    if any(DATA_DIR.iterdir()):
        st.info(f"Data directory `{DATA_DIR}` already contains files. Skipping extraction.")
        return True

    status = st.status("Extracting dataset archives…", expanded=True)
    status.write(f"Main zip: `{zip_path}`")
    status.write(f"Target directory: `{DATA_DIR}`")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_count = len(zf.infolist())
            status.write(f"Main archive contains ~{file_count} entries.")
            progress = st.progress(0.0, text="Extracting main archive…")
            # We cannot get fine-grained progress from ZipFile, so just fake incremental updates.
            # This is mainly to show that work is being done.
            for i, member in enumerate(zf.infolist(), start=1):
                zf.extract(member, DATA_DIR)
                if i % max(1, file_count // 20) == 0:
                    progress.progress(min(i / file_count, 1.0))
            progress.progress(1.0)
    except Exception as e:
        status.update(state="error", label="Extraction failed")
        st.error(f"Failed to extract main zip: {e}")
        return False

    # Extract any nested zips (e.g. imgs.zip, driver_imgs_list.csv.zip)
    nested = list(DATA_DIR.glob("*.zip"))
    if nested:
        status.write(f"Found {len(nested)} nested archives to extract.")
    for idx, inner_zip in enumerate(nested, start=1):
        status.write(f"[{idx}/{len(nested)}] Extracting nested archive: {inner_zip.name}")
        try:
            with zipfile.ZipFile(inner_zip, "r") as zf:
                zf.extractall(DATA_DIR)
        except Exception as e:
            st.warning(f"Failed to extract nested zip {inner_zip.name}: {e}")

    status.update(state="complete", label="Extraction complete")
    st.success("Extraction complete.")
    return True


def prepare_train_val_test_splits():
    """
    Create train/val/test splits from imgs/train using splitfolders.ratio.
    """
    source_dir = DATA_DIR / "imgs" / "train"
    output_dir = DATA_DIR / "output"

    if not source_dir.exists():
        st.error(f"Expected training images at `{source_dir}`, but they were not found.")
        return False

    if output_dir.exists():
        st.info(f"Split directory `{output_dir}` already exists. Skipping splitting.")
        return True

    status = st.status("Creating train/val/test splits…", expanded=True)
    status.write(f"Source directory: `{source_dir}`")
    status.write(f"Output directory: `{output_dir}`")
    status.write("Using ratio: train=0.7, val=0.15, test=0.15")

    try:
        splitfolders.ratio(
            str(source_dir),
            output=str(output_dir),
            seed=1337,
            ratio=(0.7, 0.15, 0.15),
        )
    except Exception as e:
        status.update(state="error", label="Failed to create splits")
        st.error(f"Failed to create splits: {e}")
        return False

    status.update(state="complete", label="Dataset splitting complete")
    st.success("Dataset splitting complete.")
    return True


def train_model(epochs: int) -> Path | None:
    """
    Train a YOLO classification model on the prepared dataset.
    Returns the path to the best model weights or None on failure.
    """
    # First, explicitly check for NumPy so we can give a clear error if it's missing.
    if not check_numpy_available():
        return None

    data_path = DATA_DIR / "output"
    if not data_path.exists():
        st.error(f"Split data directory `{data_path}` not found. Run the previous steps first.")
        return None

    status = st.status("Starting YOLO training…", expanded=True)
    status.write(f"Training data directory: `{data_path}`")
    status.write(f"Project directory for runs: `{BASE_DIR / 'runs'}`")
    status.write(f"Requested epochs: {epochs}")

    # Use the YOLO CLI in a subprocess so we can stream epoch-by-epoch logs into the UI.
    cmd = [
        "yolo",
        "classify",
        "train",
        f"model={BASE_DIR / 'yolo11n-cls.pt'}",
        f"data={data_path}",
        f"epochs={epochs}",
        f"project={BASE_DIR / 'runs'}",
        "name=statefarm_cls",
        "exist_ok=True",
    ]
    status.write("Running command:")
    status.code(" ".join(str(part) for part in cmd), language="bash")

    progress = st.progress(0.0, text="Starting training…")
    log_placeholder = st.empty()
    log_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        status.update(state="error", label="YOLO CLI not found")
        st.error(
            "The `yolo` command was not found.\n\n"
            "Make sure Ultralytics is installed in this environment and the `yolo` CLI "
            "is on your PATH (e.g. `pip install ultralytics`)."
        )
        return None
    except Exception as e:  # pragma: no cover - runtime environment dependent
        status.update(state="error", label="Failed to start training process")
        st.error(f"Failed to start YOLO training process: {e}")
        return None

    detected_total_epochs: int | None = None
    current_epoch = 0

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            log_lines.append(line)
            # Show only the last ~40 lines to avoid an overly long UI block.
            log_placeholder.code("\n".join(log_lines[-40:]), language="bash")

            # Try to parse epoch progress from typical ultralytics log patterns, e.g.:
            # "Epoch 1/50" or "[1/50]"
            match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
            if not match:
                match = re.search(r"\[(\d+)\s*/\s*(\d+)\]", line)

            if match:
                try:
                    current_epoch = int(match.group(1))
                    detected_total_epochs = int(match.group(2))
                except ValueError:
                    pass

            if detected_total_epochs:
                frac = min(max(current_epoch / detected_total_epochs, 0.0), 1.0)
                progress.progress(frac, text=f"Training epoch {current_epoch}/{detected_total_epochs}…")

    return_code = proc.wait()

    if return_code != 0:
        status.update(state="error", label="Training failed")
        progress.progress(0.0, text="Training failed.")
        st.error("YOLO training process failed. See the log output above for details.")
        return None

    progress.progress(1.0, text="Training complete.")

    # Find the most recent run directory that matches our name and contains best.pt
    runs_dir = BASE_DIR / "runs"
    if not runs_dir.exists():
        status.update(state="error", label="Training finished but runs directory not found")
        st.error(f"Training finished but `{runs_dir}` does not exist.")
        return None

    candidate_runs = sorted(
        runs_dir.glob("statefarm_cls*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    best_path: Path | None = None
    for run_dir in candidate_runs:
        candidate = run_dir / "weights" / "best.pt"
        if candidate.exists():
            best_path = candidate
            break

    if best_path is None:
        status.update(state="error", label="Training finished but best.pt not found")
        st.error(
            "Training finished but `best.pt` was not found under "
            f"`{runs_dir}/statefarm_cls*`."
        )
        return None

    status.write(f"Detected Ultralytics run directory: `{best_path.parent.parent}`")
    status.update(state="complete", label="Training complete")
    st.success(f"Training complete. Best model saved at: {best_path}")
    return best_path


def evaluate_model(weights_path: Path):
    """
    Run validation on the trained model and show key metrics.
    """
    if not weights_path.exists():
        st.error(f"Provided weights path does not exist: {weights_path}")
        return

    status = st.status("Running evaluation…", expanded=True)
    status.write(f"Using weights: `{weights_path}`")

    # Lazily import YOLO here so evaluation only pulls in the heavy dependency when needed.
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        status.write("Model loaded, starting `model.val()`…")
        metrics = model.val()
    except Exception as e:
        status.update(state="error", label="Validation failed")
        st.error(f"Validation failed: {e}")
        return

    try:
        metrics_dict = metrics.results_dict
        status.write("Raw metrics dictionary:")
        st.json(metrics_dict)
    except Exception:
        status.write("Validation completed, but could not parse results dictionary.")

    # Show confusion matrix and results plots if available
    conf_mat = weights_path.parent.parent / "confusion_matrix_normalized.png"
    results_plot = weights_path.parent.parent / "results.png"

    if conf_mat.exists():
        status.write("Displaying normalized confusion matrix plot.")
        st.image(str(conf_mat), caption="Normalized Confusion Matrix")
    if results_plot.exists():
        status.write("Displaying training metrics plot (results.png).")
        st.image(str(results_plot), caption="Training Metrics (results.png)")

    status.update(state="complete", label="Evaluation complete")


def run_full_training_pipeline(epochs: int) -> Path | None:
    """
    Convenience helper to run: download -> extract -> split -> train.
    Returns best model path or None.
    """
    overall = st.status("Running full training pipeline…", expanded=True)
    overall.write("Step 1/4: Download dataset from Kaggle")
    if not download_competition_dataset():
        overall.update(state="error", label="Pipeline stopped during download")
        return None

    overall.write("Step 2/4: Extract dataset archives")
    if not extract_dataset():
        overall.update(state="error", label="Pipeline stopped during extraction")
        return None

    overall.write("Step 3/4: Create train/val/test splits")
    if not prepare_train_val_test_splits():
        overall.update(state="error", label="Pipeline stopped during splitting")
        return None

    overall.write(f"Step 4/4: Train YOLO classifier for {epochs} epochs")
    best_path = train_model(epochs)
    if best_path is None:
        overall.update(state="error", label="Pipeline stopped during training")
        return None

    overall.update(state="complete", label="Training pipeline complete")
    return best_path


def show_validation_samples(weights_path: Path, max_images: int = 30):
    """
    Show a small set of validation images with predicted vs true labels.
    """
    if not weights_path.exists():
        st.error(f"Model weights not found at: {weights_path}")
        return

    val_dir = DATA_DIR / "output" / "val"
    if not val_dir.exists():
        st.error(
            f"Validation directory `{val_dir}` not found. "
            "Run the training pipeline to create splits first."
        )
        return

    # Collect a balanced set of image paths per class so that each class is represented.
    class_dirs = [d for d in sorted(val_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        st.error(f"No validation images found under `{val_dir}`.")
        return
    num_classes = len(class_dirs)
    per_class = max(1, max_images // num_classes)
    leftover = max_images - per_class * num_classes

    sample_paths: list[Path] = []
    for class_dir in class_dirs:
        imgs = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not imgs:
            continue

        # Aim for an even number of samples per class, distributing leftovers across classes.
        n = min(len(imgs), per_class + (1 if leftover > 0 else 0))
        if leftover > 0:
            leftover -= 1

        if len(imgs) <= n:
            chosen = imgs
        else:
            chosen = random.sample(imgs, k=n)

        sample_paths.extend(chosen)
        if len(sample_paths) >= max_images:
            break

    if not sample_paths:
        st.error(f"No validation images found under `{val_dir}`.")
        return

    # Truncate in case we slightly exceeded the target due to class balancing.
    sample_paths = sample_paths[:max_images]

    status = st.status(
        f"Running predictions on {len(sample_paths)} validation images…",
        expanded=True,
    )
    status.write(f"Using weights: `{weights_path}`")

    model = load_or_get_model(weights_path)

    cols_per_row = 4
    cols = st.columns(cols_per_row)

    for idx, img_path in enumerate(sample_paths):
        try:
            result = model(str(img_path))[0]
            top1_idx = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
        except Exception as e:
            status.write(f"Failed prediction for {img_path.name}: {e}")
            continue

        true_folder = img_path.parent.name  # e.g. 'c0'
        try:
            true_idx = int(true_folder.lstrip("c"))
        except ValueError:
            true_idx = None

        true_label = CLASS_LABELS.get(true_idx, true_folder)
        pred_label = CLASS_LABELS.get(top1_idx, f"class {top1_idx}")

        col = cols[idx % cols_per_row]
        with col:
            st.image(
                str(img_path),
                caption=(
                    f"True: {true_label}\n"
                    f"Pred: {pred_label} ({top1_conf:.2f})"
                ),
                use_column_width=True,
            )

        if (idx + 1) % cols_per_row == 0 and (idx + 1) < len(sample_paths):
            cols = st.columns(cols_per_row)

    status.update(state="complete", label="Validation image sampling complete")


def load_or_get_model(weights_path: Path):
    """
    Load a YOLO model and cache it in Streamlit session_state to avoid reloading.
    """
    key = "loaded_model"
    key_path = "loaded_model_path"

    if (
        key in st.session_state
        and key_path in st.session_state
        and st.session_state[key_path] == str(weights_path)
    ):
        return st.session_state[key]

    # Show a clear spinner while the (potentially slow) YOLO model is being loaded.
    with st.spinner("Loading YOLO model for inference… this can take 10–30 seconds on first run."):
        from ultralytics import YOLO

        model = YOLO(str(weights_path))

    st.session_state[key] = model
    st.session_state[key_path] = str(weights_path)
    return model


def predict_image(weights_path: Path, uploaded_image):
    """
    Run a single image through the trained classifier and display prediction.
    """
    if not weights_path.exists():
        st.error(f"Model weights not found at: {weights_path}")
        return

    model = load_or_get_model(weights_path)

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        results = model(image)
        result = results[0]
        top1_idx = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)

        label = CLASS_LABELS.get(top1_idx, f"Class {top1_idx}")
        st.markdown(
            f"**Predicted class:** `{label}`  \n"
            f"**Confidence:** {top1_conf:.3f}"
        )

        # Show top-5 probabilities if available
        if hasattr(result.probs, "top5") and hasattr(result.probs, "top5conf"):
            st.write("Top-5 predictions:")
            for cls_idx, conf in zip(result.probs.top5, result.probs.top5conf):
                cls_idx = int(cls_idx)
                conf = float(conf)
                st.write(f"- {CLASS_LABELS.get(cls_idx, f'class {cls_idx}')} — {conf:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


def main():
    st.set_page_config(
        page_title="Driver Distraction Detection - YOLO Classifier",
        layout="wide",
    )

    st.title("Driver Distraction Detection")
    st.write(
        "End-to-end workflow based on your Jupyter notebook, organized into three steps:\n"
        "1) **Train model**, 2) **Evaluate & visualize**, 3) **Predict from uploaded image**."
    )

    # Keep track of preferred weights path in session state
    if "best_weights_path" not in st.session_state:
        # Default: what training function will create
        st.session_state["best_weights_path"] = str(
            BASE_DIR / "runs" / "statefarm_cls" / "weights" / "best.pt"
        )

    tab_train, tab_eval, tab_predict = st.tabs(
        [
            "1. Train model",
            "2. Evaluate & visualize",
            "3. Predict from image",
        ]
    )

    with tab_train:
        st.subheader("Train model (download → extract → split → train)")
        st.write(
            "Use the four steps below to run the full training workflow:\n"
            "- **Step 1**: Download dataset from Kaggle\n"
            "- **Step 2**: Extract archives\n"
            "- **Step 3**: Create train/val/test splits\n"
            "- **Step 4**: Train YOLO classifier and save `best.pt`"
        )

        # Initialize simple step-tracking state for the pipeline.
        if "pipeline_steps" not in st.session_state:
            st.session_state["pipeline_steps"] = {
                "download": False,
                "extract": False,
                "split": False,
                "train": False,
            }
        steps = st.session_state["pipeline_steps"]

        st.markdown("**Pipeline status overview**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label="1. Download",
                value="Done" if steps.get("download") else "Pending",
            )
        with c2:
            st.metric(
                label="2. Extract",
                value="Done" if steps.get("extract") else "Pending",
            )
        with c3:
            st.metric(
                label="3. Split data",
                value="Done" if steps.get("split") else "Pending",
            )
        with c4:
            st.metric(
                label="4. Train model",
                value="Done" if steps.get("train") else "Pending",
            )

        st.markdown("**Prerequisites**")
        st.markdown(
            "- `kaggle.json` present next to this script (no upload required)\n"
            "- `pip install kaggle` and `kaggle` CLI available on PATH"
        )

        epochs = st.number_input(
            "Number of training epochs",
            min_value=1,
            max_value=100,
            value=DEFAULT_EPOCHS,
            step=1,
        )

        st.markdown("**Step-by-step controls**")
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Step 1: Download dataset", key="btn_step_download"):
                if download_competition_dataset():
                    steps["download"] = True
            if st.button("Step 2: Extract dataset", key="btn_step_extract"):
                if extract_dataset():
                    steps["extract"] = True
        with col_right:
            if st.button("Step 3: Create train/val/test splits", key="btn_step_split"):
                if prepare_train_val_test_splits():
                    steps["split"] = True
            if st.button("Step 4: Train model only", key="btn_step_train"):
                best_path = train_model(epochs)
                if best_path is not None:
                    steps["train"] = True
                    st.session_state["best_weights_path"] = str(best_path)

        if DATA_DIR.exists():
            st.info(f"Current data directory: `{DATA_DIR}`")

    with tab_eval:
        st.subheader("Evaluate model & view validation samples")
        st.write(
            "Run validation on the trained model and view a few validation images "
            "with predicted vs true labels."
        )

        weights_text = st.text_input(
            "Model weights to use (`best.pt`)",
            value=st.session_state.get("best_weights_path", ""),
            key="eval_weights_input",
        )
        st.session_state["best_weights_path"] = weights_text
        weights_path = Path(weights_text) if weights_text else None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run evaluation (metrics & confusion matrix)", key="eval_button"):
                if weights_path is None or not weights_path.exists():
                    st.error("Please provide a valid path to existing weights.")
                else:
                    evaluate_model(weights_path)

        with col2:
            if st.button("Show sample validation predictions", key="val_samples_button"):
                if weights_path is None or not weights_path.exists():
                    st.error("Please provide a valid path to existing weights.")
                else:
                    show_validation_samples(weights_path)

    with tab_predict:
        st.subheader("Predict distraction class from uploaded image")
        st.write(
            "Upload a single image of a driver, and the trained YOLO classifier will "
            "predict which distraction class it belongs to."
        )

        weights_text = st.text_input(
            "Model weights to use for inference (`best.pt`)",
            value=st.session_state.get("best_weights_path", ""),
            key="predict_weights_input",
        )
        st.session_state["best_weights_path"] = weights_text
        weights_path = Path(weights_text) if weights_text else None

        uploaded_image = st.file_uploader(
            "Upload an image (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="predict_image_uploader",
        )

        if uploaded_image is not None and weights_path is not None:
            if st.button("Run prediction", key="predict_button"):
                predict_image(weights_path, uploaded_image)
        elif uploaded_image is not None and weights_path is None:
            st.info("Please provide a valid path to model weights first.")


if __name__ == "__main__":
    main()


