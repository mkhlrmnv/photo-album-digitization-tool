import reflex as rx
from app.states.state import State
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to console
)

def upload_form() -> rx.Component:
    """The form to upload a file."""
    logging.info("Rendering upload form.")
    return rx.el.div(
        rx.upload.root(
            rx.el.div(
                rx.icon("cloud_upload", class_name="w-10 h-10 text-gray-400"),
                rx.el.h3(
                    "Click to upload or drag and drop",
                    class_name="font-semibold text-gray-700",
                ),
                rx.el.p(
                    "ZIP files only (up to 50MB)", class_name="text-sm text-gray-500"
                ),
                class_name="flex flex-col items-center justify-center gap-2 text-center p-8",
            ),
            id="upload-zone",
            class_name="w-full max-w-lg border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-blue-500 transition-colors",
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload-zone")),
        ),
        class_name="flex items-center justify-center w-full",
    )


def upload_status() -> rx.Component:
    """The status of the upload."""
    logging.info("Rendering upload status.")
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.icon("file_archive", class_name="w-8 h-8 text-blue-600"),
                class_name="p-3 bg-blue-100 rounded-lg",
            ),
            rx.el.div(
                rx.el.p(
                    State.file_name,
                    class_name="font-semibold text-gray-800 text-sm truncate",
                ),
                rx.el.p(
                    f"{(State.file_size / 1024 / 1024).to_string()} MB",
                    class_name="text-xs text-gray-500",
                ),
                class_name="flex-1 min-w-0",
            ),
            rx.el.button(
                rx.icon("x", class_name="w-4 h-4"),
                on_click=State.cancel_upload,
                class_name="p-1 text-gray-500 hover:text-gray-800 hover:bg-gray-100 rounded-full",
            ),
            class_name="flex items-center gap-4 w-full",
        ),
        rx.el.div(
            rx.el.div(
                class_name="bg-blue-600 h-2 rounded-full transition-all duration-300",
                style={"width": State.upload_progress.to_string() + "%"},
            ),
            class_name="w-full bg-gray-200 rounded-full h-2 mt-2",
        ),
        rx.el.p(State.upload_message, class_name="text-sm text-gray-600 mt-2"),
        class_name="w-full max-w-lg p-4 bg-white border border-gray-200 rounded-xl shadow-sm",
    )


def results_summary() -> rx.Component:
    """The summary of the results."""
    logging.info("Rendering results summary.")
    return rx.el.div(
        rx.el.h3(
            "Processing Complete", class_name="text-lg font-semibold text-gray-800"
        ),
        rx.el.div(
            rx.el.div(
                rx.el.p("File", class_name="text-sm text-gray-500"),
                rx.el.p(State.file_name, class_name="font-medium text-gray-800"),
                class_name="flex-1",
            ),
            rx.el.div(
                rx.el.p("Images Found", class_name="text-sm text-gray-500"),
                rx.el.p(
                    State.file_images_count, class_name="font-medium text-gray-800"
                ),
                class_name="text-right",
            ),
            class_name="flex justify-between items-center mt-4 p-4 bg-gray-50 rounded-lg",
        ),
        rx.el.button(
            rx.icon("wand-sparkles", class_name="mr-2"),
            rx.cond(State.is_processing, "Processing...", "Process Images"),
            on_click=State.process_images,
            disabled=State.is_processing,
            class_name="mt-6 w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center disabled:bg-gray-400 disabled:cursor-not-allowed",
        ),
        class_name="w-full max-w-lg p-6 bg-white border border-gray-200 rounded-xl shadow-sm",
    )


def processed_gallery() -> rx.Component:
    """The gallery of processed images."""
    logging.info("Rendering processed gallery.")
    return rx.el.div(
        rx.el.h3(
            "Processed Images", class_name="text-xl font-semibold text-gray-800 mb-4"
        ),
        rx.el.div(
            rx.foreach(
                State.processed_images,
                lambda img: rx.el.div(
                    rx.el.img(
                        src=rx.get_upload_url(img[1]),
                        class_name="rounded-lg object-cover w-full h-full",
                    ),
                    rx.el.div(
                        rx.el.p(img[0], class_name="text-xs text-white truncate"),
                        class_name="absolute bottom-0 left-0 right-0 p-2 bg-black/50",
                    ),
                    class_name="relative overflow-hidden rounded-lg shadow-sm border border-gray-200 aspect-square",
                ),
            ),
            class_name="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4",
        ),
        class_name="w-full max-w-4xl p-6 bg-white border border-gray-200 rounded-xl shadow-sm mt-8",
    )


def final_results_view() -> rx.Component:
    """The final results view with download link and gallery."""
    logging.info("Rendering final results view.")
    return rx.el.div(
        rx.el.div(
            rx.el.h3(
                "Processing Complete", class_name="text-lg font-semibold text-gray-800"
            ),
            rx.el.div(
                rx.el.div(
                    rx.el.p("File", class_name="text-sm text-gray-500"),
                    rx.el.p(State.file_name, class_name="font-medium text-gray-800"),
                    class_name="flex-1",
                ),
                rx.el.div(
                    rx.el.p("Images Processed", class_name="text-sm text-gray-500"),
                    rx.el.p(
                        State.processed_images_count,
                        class_name="font-medium text-gray-800",
                    ),
                    class_name="text-center",
                ),
                rx.el.div(
                    rx.el.p("Objects Detected", class_name="text-sm text-gray-500"),
                    rx.el.p(
                        State.total_objects_detected,
                        class_name="font-medium text-gray-800",
                    ),
                    class_name="text-right",
                ),
                class_name="flex justify-between items-center mt-4 p-4 bg-gray-50 rounded-lg",
            ),
            rx.el.a(
                rx.icon("download", class_name="mr-2"),
                "Download Processed Images",
                href=rx.get_upload_url(State.download_filename),
                class_name="mt-6 w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center",
            ),
            rx.el.button(
                rx.icon("refresh-cw", class_name="mr-2"),
                "Process Another File",
                on_click=State.cancel_upload,
                class_name="mt-2 w-full bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded-lg hover:bg-gray-300 transition-colors flex items-center justify-center",
            ),
            class_name="w-full max-w-lg p-6 bg-white border border-gray-200 rounded-xl shadow-sm",
        ),
        processed_gallery(),
        class_name="flex flex-col items-center gap-8 w-full",
    )


def processing_status_view() -> rx.Component:
    """The view for the processing status."""
    logging.info("Rendering processing status view.")
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.icon("cog", class_name="w-8 h-8 text-blue-600 animate-spin"),
                class_name="p-3 bg-blue-100 rounded-lg",
            ),
            rx.el.div(
                rx.el.p(
                    State.processing_status,
                    class_name="font-semibold text-gray-800 text-sm truncate",
                ),
                rx.el.p(
                    f"{State.processed_images_count}/{State.file_images_count} images processed",
                    class_name="text-xs text-gray-500",
                ),
                class_name="flex-1 min-w-0",
            ),
            class_name="flex items-center gap-4 w-full",
        ),
        rx.el.div(
            rx.el.div(
                class_name="bg-blue-600 h-2 rounded-full transition-all duration-300",
                style={"width": State.processing_progress.to_string() + "%"},
            ),
            class_name="w-full bg-gray-200 rounded-full h-2 mt-2",
        ),
        rx.el.p(State.upload_message, class_name="text-sm text-gray-600 mt-2"),
        class_name="w-full max-w-lg p-4 bg-white border border-gray-200 rounded-xl shadow-sm",
    )


def index() -> rx.Component:
    """The main page of the app."""
    logging.info("Rendering index page.")
    return rx.el.main(
        rx.el.div(
            rx.el.div(
                rx.el.h2(
                    "Upload Your Image Archive",
                    class_name="text-3xl font-bold text-gray-900 tracking-tight",
                ),
                rx.el.p(
                    "Drag and drop a zip file containing your images to start the object detection process.",
                    class_name="text-gray-600 mt-2 max-w-xl",
                ),
                class_name="text-center mb-8",
            ),
            rx.match(
                State.app_status,
                ("upload", upload_form()),
                ("uploading", upload_status()),
                ("processing", processing_status_view()),
                ("results", final_results_view()),
                results_summary(),
            ),
            class_name="container mx-auto flex flex-col items-center justify-center min-h-screen px-4 py-24",
        ),
        class_name="font-['JetBrains_Mono'] bg-gray-50",
    )


app = rx.App(
    theme=rx.theme(appearance="light"),
    head_components=[
        rx.el.link(rel="preconnect", href="https://fonts.googleapis.com"),
        rx.el.link(rel="preconnect", href="https://fonts.gstatic.com", cross_origin=""),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap",
            rel="stylesheet",
        ),
    ],
)
app.add_page(index)