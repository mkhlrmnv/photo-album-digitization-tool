import reflex as rx
import zipfile
import io
import logging
from PIL import Image
import cv2
import numpy as np
import time
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to console
)

YOLO_MODEL = '../approach_with_custom_ml/training-runs/combination-with-synthetic-v2/photo_segmentation/weights/best.pt'


class State(rx.State):
    """The app state."""

    is_uploading: bool = False
    upload_progress: int = 0
    upload_message: str = ""
    file_name: str = ""
    file_size: int = 0
    file_images_count: int = 0
    extracted_images: list[tuple[str, bytes]] = []
    processing_status: str = ""
    is_processing: bool = False
    processing_progress: int = 0
    processed_images_count: int = 0
    total_objects_detected: int = 0
    processed_images: list[tuple[str, str]] = []
    MAX_FILE_SIZE = 50 * 1024 * 1024
    download_filename: str = ""

    @rx.var
    def app_status(self) -> str:
        """The current status of the application."""
        logging.info("Checking application status.")
        if self.is_processing:
            return "processing"
        if self.download_filename:
            return "results"
        if self.is_uploading:
            return "uploading"
        if self.upload_progress == 100:
            return "ready_to_process"
        return "upload"

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of image files."""
        logging.info("Starting file upload.")
        self.is_uploading = True
        self.upload_message = "Starting upload..."
        self.processing_status = ""
        self.download_filename = ""
        self.extracted_images = []
        self.file_name = ""
        self.file_size = 0
        self.file_images_count = 0
        yield

        # Validate and process each file
        total_files = len(files)
        self.file_images_count = total_files
        if total_files == 0:
            logging.warning("No valid image files uploaded.")
            self.upload_message = "Error: No valid image files uploaded."
            self.is_uploading = False
            return

        extracted_count = 0
        for i, file in enumerate(files):
            if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                logging.warning(f"Skipping unsupported file: {file.filename}")
                continue

            # Read file data
            self.file_name = file.filename
            self.file_size = file.size
            self.upload_message = f'Uploading "{self.file_name}"...'
            logging.info(f"Uploading file: {self.file_name} ({self.file_size} bytes).")
            yield

            try:
                file_data = await file.read()
                self.extracted_images.append((file.filename, file_data))
                extracted_count += 1
                self.upload_progress = int((i + 1) / total_files * 100)
                self.upload_message = f"Uploaded {extracted_count}/{total_files} images..."
                yield
            except Exception as e:
                logging.exception(f"Error reading file {file.filename}: {e}")
                self.upload_message = f"Error reading file {file.filename}: {e}"
                self.is_uploading = False
                return

        if extracted_count == 0:
            logging.warning("No valid images were uploaded.")
            self.upload_message = "Error: No valid images were uploaded."
            self.is_uploading = False
            return

        logging.info(f"Successfully uploaded {extracted_count} images.")
        self.upload_message = f"Successfully uploaded {extracted_count} images."
        self.processing_status = "Ready to process."
        self.is_uploading = False
        self.upload_progress = 100
        yield

    @rx.event
    def cancel_upload(self):
        """Reset the state and cancel the upload."""
        logging.info("Cancelling the upload and resetting the state.")
        self.is_uploading = False
        self.upload_progress = 0
        self.upload_message = ""
        self.file_name = ""
        self.file_size = 0
        self.file_images_count = 0
        self.extracted_images = []
        self.processing_status = ""
        self.download_filename = ""
        self.is_processing = False
        self.processing_progress = 0
        self.processed_images_count = 0
        self.total_objects_detected = 0
        self.processed_images = []

    @rx.event
    def process_images(self):
        """Process extracted images with YOLO model."""
        logging.info("Starting image processing.")
        if not self.extracted_images:
            logging.warning("No images to process.")
            return
        self.is_processing = True
        self.processing_status = "Loading YOLO model..."
        self.processed_images = []
        self.processed_images_count = 0
        self.total_objects_detected = 0
        yield
        try:
            model = YOLO(YOLO_MODEL)
            logging.info("YOLO model loaded successfully.")
        except NameError:
            logging.error("YOLO model could not be loaded.")
            self.processing_status = "Error: YOLO model could not be loaded. Please check installation."
            self.is_processing = False
            yield
            return
        try:
            total_images = len(self.extracted_images)
            upload_dir = rx.get_upload_dir()
            upload_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Clearing upload directory: {upload_dir}")
            for p in upload_dir.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception as e:
                    logging.exception(f"Failed to remove {p}: {e}")

            processed_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                processed_zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zf:
                for i, (image_name, image_data) in enumerate(self.extracted_images):
                    self.processing_status = f"Processing image {i + 1}/{total_images}..."
                    logging.info(f"Processing image {i + 1}/{total_images}: {image_name}")
                    self.processing_progress = int((i + 1) / total_images * 100)
                    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    img_width, img_height = pil_image.size

                    # Run YOLO model on the image
                    results = model(pil_image, verbose=False)

                    # Create a BGR image for processing / cropping
                    base_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    # collect crops for this image
                    crops = []
                    crop_idx = 0

                    for result in results:
                        # iterate masks (if present) and create smallest rotated rect crops
                        for mask in getattr(result, "masks", []).xy if getattr(result, "masks", None) else []:
                            points = np.array(mask, dtype=np.float32)
                            if points.size == 0 or len(points) < 3:
                                continue
                            try:
                                rect = cv2.minAreaRect(points)  # ((cx,cy),(w,h),angle)
                                (cx, cy), (w, h), angle = rect
                                # skip degenerate
                                if w <= 0 or h <= 0:
                                    continue
                                # ensure width >= height for consistent rotation handling
                                if w < h:
                                    angle += 90
                                    w, h = h, w
                                # rotation matrix around center
                                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                                # rotate whole image
                                rotated = cv2.warpAffine(base_image, M, (img_width, img_height), flags=cv2.INTER_CUBIC)
                                # compute crop coords
                                x = int(round(cx - w / 2.0))
                                y = int(round(cy - h / 2.0))
                                w_i = int(round(w))
                                h_i = int(round(h))
                                # clamp to image bounds
                                x = max(0, x)
                                y = max(0, y)
                                if x + w_i > img_width:
                                    w_i = img_width - x
                                if y + h_i > img_height:
                                    h_i = img_height - y
                                if w_i <= 0 or h_i <= 0:
                                    continue
                                crop = rotated[y:y + h_i, x:x + w_i]
                                if crop.size == 0:
                                    continue
                                # convert crop to JPEG bytes
                                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                buf = io.BytesIO()
                                crop_pil.save(buf, format="JPEG")
                                crop_bytes = buf.getvalue()
                                crop_name = f"{Path(image_name).stem}_crop_{crop_idx}.jpg"
                                crops.append((crop_name, crop_bytes))
                                # also save to upload_dir so UI can reference it
                                processed_filepath = upload_dir / crop_name
                                with open(processed_filepath, "wb") as pf:
                                    pf.write(crop_bytes)
                                self.processed_images.append((image_name, crop_name))
                                self.processed_images_count += 1
                                crop_idx += 1
                            except Exception:
                                # skip this mask on any issue
                                continue

                    # write all crops for this image into the zip (if none, skip)
                    for cname, cbytes in crops:
                        zf.writestr(cname, cbytes)

                    # update objects detected: count masks across all results
                    masks_count = 0
                    for result in results:
                        masks_count += len(getattr(result, "masks", []).xy) if getattr(result, "masks", None) else 0
                    self.total_objects_detected += masks_count

                    yield
            self.processing_progress = 100
            self.processing_status = f"Processing complete. Found {self.total_objects_detected} objects."
            logging.info(f"Processing complete. Total objects detected: {self.total_objects_detected}.")
            zip_filename = f"processed_{int(time.time())}.zip"
            zip_filepath = upload_dir / zip_filename
            with open(zip_filepath, "wb") as f:
                f.write(processed_zip_buffer.getvalue())
            self.download_filename = zip_filename
        except Exception as e:
            logging.exception(f"Error processing images: {e}")
            self.processing_status = f"Error: {e}"
        finally:
            self.is_processing = False
            yield