import reflex as rx
from web.models.app_state import AppState
from web.components.metrics import metric_card, metrics_grid


def welcome_section():
    return rx.vstack(
        rx.heading("Stochomaly", size="1", mb=2),
        rx.text(
            "Self-learning anomaly detection using stochastic optimization techniques.",
            color="gray.600",
            font_size="lg",
        ),
        rx.spacer(),
        rx.hstack(
            rx.button(
                "Load Dataset",
                on_click=AppState.open_dataset_modal,
                color_scheme="blue",
                size="3",
                margin_right="4",
            ),
            rx.button(
                "View Documentation",
                on_click=rx.redirect("https://github.com/steepcloud/Stochomaly"),
                variant="outline",
                size="3",
            ),
            justify="center",
            width="100%",
        ),
        align_items="center",
        padding="6",
        spacing="6",
        margin_y="4",
    )


def dataset_overview():
    return rx.cond(
        AppState.dataset_loaded,
        rx.box(
            rx.heading("Dataset Overview", size="2", mb="4"),
            metrics_grid({
                "Rows": AppState.dataset_rows,
                "Features": AppState.dataset_cols,
                "Missing Values": f"{AppState.missing_values_pct}%",
                "Anomalies": "Unknown",
            }),
            padding="6",
            border_radius="md",
            border="1px solid",
            border_color="gray.200",
            margin_y="4",
            width="100%",
        ),
        rx.box()
    )


def workflow_section():
    return rx.box(
        rx.heading("Anomaly Detection Workflow", size="2", mb="4"),
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.image(src="/icons/data.svg", height="50px", mb="2"),
                    rx.heading("Data Selection", size="3"),
                    rx.text("Load and explore your dataset"),
                    rx.spacer(),
                    rx.link(
                        rx.button("Start", variant="ghost"),
                        href="/data_selection",
                    ),
                    height="100%",
                ),
                height="100%",
            ),
            rx.card(
                rx.vstack(
                    rx.image(src="/icons/features.svg", height="50px", mb="2"),
                    rx.heading("Feature Engineering", size="3"),
                    rx.text("Transform and extract relevant features"),
                    rx.spacer(),
                    rx.link(
                        rx.button("Start", variant="ghost"),
                        href="/feature_engineering",
                    ),
                    height="100%",
                ),
                height="100%",
            ),
            rx.card(
                rx.vstack(
                    rx.image(src="/icons/model.svg", height="50px", mb="2"),
                    rx.heading("Model Training", size="3"),
                    rx.text("Configure and train detection models"),
                    rx.spacer(),
                    rx.link(
                        rx.button("Start", variant="ghost"),
                        href="/model_training",
                    ),
                    height="100%",
                ),
                height="100%",
            ),
            rx.card(
                rx.vstack(
                    rx.image(src="/icons/eval.svg", height="50px", mb="2"),
                    rx.heading("Evaluation", size="3"),
                    rx.text("Analyze performance and results"),
                    rx.spacer(),
                    rx.link(
                        rx.button("Start", variant="ghost"),
                        href="/evaluation",
                    ),
                    height="100%",
                ),
                height="100%",
            ),
            template_columns="repeat(auto-fit, minmax(250px, 1fr))",
            gap="4",
            width="100%",
        ),
        padding="6",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        margin_y="4",
        width="100%",
    )


def model_summary():
    return rx.cond(
        AppState.model_trained,
        rx.box(
            rx.heading("Latest Model Results", size="2", mb="4"),
            metrics_grid({
                "Accuracy": f"{AppState.model_accuracy:.2%}",
                "Precision": f"{AppState.model_precision:.2%}",
                "Recall": f"{AppState.model_recall:.2%}",
                "F1 Score": f"{AppState.model_f1:.2%}",
            }),
            padding="6",
            border_radius="md",
            border="1px solid",
            border_color="gray.200",
            margin_y="4",
            width="100%",
        ),
        rx.box()
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            welcome_section(),
            dataset_overview(),
            workflow_section(),
            model_summary(),
            width="100%",
            spacing="4",
            padding_y="4",
        ),
        padding="6",
        max_width="1200px",
    )


# Create the page
def create():
    return rx.fragment(
        rx.vstack(
            index(),
            width="100%",
            spacing="0",
        )
    )