import reflex as rx
from app.models.app_state import AppState


def source_selection():
    return rx.box(
        rx.heading("Data Source", size="3"),
        rx.radio_group(
            ["sklearn", "csv", "xor"],
            value=AppState.data_source,
            on_change=AppState.set_data_source,
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def dataset_config():
    return rx.box(
        rx.heading("Dataset Configuration", size="3"),

        # sklearn dataset options
        rx.vstack(
            rx.select(
                ["iris", "breast_cancer", "digits", "wine"],
                label="Select Dataset",
                value=AppState.dataset_name,
                on_change=AppState.set_dataset_name,
                width="100%",
            ),
            display=rx.cond(AppState.data_source == "sklearn", "block", "none"),
        ),

        # CSV upload options
        rx.vstack(
            rx.upload(
                rx.vstack(
                    rx.button(
                        "Select CSV File",
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.text("Drag and drop files here or click to select files"),
                ),
                border="1px dashed",
                border_color="gray.200",
                border_radius="md",
                padding="4",
                multiple=False,
                accept=".csv",
                max_files=1,
                on_file_drop=AppState.handle_upload,
                mb=2,
            ),
            rx.cond(
                AppState.csv_filepath != "",
                rx.text(f"Selected file: {AppState.csv_filepath.split('/')[-1]}", color="green.500"),
            ),
            rx.input(
                placeholder="Enter target column name",
                value=AppState.target_column,
                on_change=AppState.set_target_column,
                width="100%",
            ),
            display=rx.cond(AppState.data_source == "csv", "block", "none"),
        ),

        # Preprocessing options (common for all data sources)
        rx.vstack(
            rx.select(
                ["minmax", "standard", "none"],
                label="Scaler Type",
                value=AppState.scaler_type,
                on_change=AppState.set_scaler_type,
                width="100%",
                mb=4,
            ),
            rx.button(
                "Load Data",
                on_click=AppState.load_data,
                color_scheme="green",
                size="3",
                width="100%",
            ),
        ),

        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def data_preview():
    return rx.box(
        rx.heading("Data Preview", size="3"),
        rx.cond(
            len(AppState.X_train) > 0,
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.text("Training Set", font_weight="bold"),
                        rx.text(
                            f"Features: {len(AppState.X_train)} samples × {len(AppState.X_train[0]) if AppState.X_train else 0} features"),
                        rx.text(f"Labels: {len(AppState.y_train)} samples"),
                    ),
                    rx.vstack(
                        rx.text("Testing Set", font_weight="bold"),
                        rx.text(
                            f"Features: {len(AppState.X_test)} samples × {len(AppState.X_test[0]) if AppState.X_test else 0} features"),
                        rx.text(f"Labels: {len(AppState.y_test)} samples"),
                    ),
                    width="100%",
                    spacing="8",
                ),

                rx.button(
                    "Continue to Feature Engineering",
                    color_scheme="blue",
                    size="3",
                    on_click=rx.set_value(str(AppState.current_step), 2),
                    width="100%",
                    mt=4,
                ),
            ),
            rx.text("Load data to see preview"),
        ),

        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
    )


def data_selection_page():
    return rx.container(
        rx.vstack(
            rx.heading("Data Selection", size="1"),
            rx.text("Select and configure your dataset"),

            source_selection(),
            dataset_config(),
            data_preview(),

            width="100%",
            spacing="4",
        ),
        padding="6",
    )