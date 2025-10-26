import reflex as rx
import zipfile
import io
import logging
from PIL import Image
import cv2
import numpy as np
import time

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
            model = YOLO("yolov8n.pt")
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
            processed_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                processed_zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zf:
                for i, (image_name, image_data) in enumerate(self.extracted_images):
                    self.processing_status = f"Processing image {i + 1}/{total_images}..."
                    logging.info(f"Processing image {i + 1}/{total_images}: {image_name}")
                    self.processing_progress = int((i + 1) / total_images * 100)
                    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    results = model(pil_image, verbose=False)
                    annotated_image = results[0].plot()
                    annotated_image_pil = Image.fromarray(
                        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    )
                    img_byte_arr = io.BytesIO()
                    annotated_image_pil.save(img_byte_arr, format="JPEG")
                    processed_image_bytes = img_byte_arr.getvalue()
                    processed_filename = f"processed_{image_name}"
                    processed_filepath = upload_dir / processed_filename
                    with open(processed_filepath, "wb") as f:
                        f.write(processed_image_bytes)
                    self.processed_images.append((image_name, processed_filename))
                    zf.writestr(image_name, processed_image_bytes)
                    self.processed_images_count += 1
                    self.total_objects_detected += len(results[0].boxes)
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