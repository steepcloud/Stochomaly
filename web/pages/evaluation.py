import reflex as rx
from web.models.app_state import AppState


def model_selection():
    return rx.box(
        rx.heading("Model Selection", size="3"),
        rx.hstack(
            rx.select(
                ["neural_network", "reinforcement_learning", "isolation_forest", "autoencoder"],
                label="Model Type",
                value=AppState.selected_model_type,
                on_change=AppState.set_selected_model_type,
            ),
            rx.select(
                AppState.available_models,
                label="Select Model",
                value=AppState.selected_model_name,
                on_change=AppState.set_selected_model_name,
                is_disabled=rx.cond(len(AppState.available_models) > 0, False, True),
            ),
        ),
        rx.button(
            "Load Model Metrics",
            on_click=AppState.load_model_metrics,
            color_scheme="blue",
            size="2",
            is_loading=AppState.is_loading_metrics,
            is_disabled=AppState.is_loading_metrics or AppState.selected_model_name == ""
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def performance_metrics():
    return rx.box(
        rx.heading("Performance Metrics", size="3"),
        rx.cond(
            AppState.metrics_loaded,
            rx.hstack(
                # Accuracy metric
                rx.vstack(
                    rx.text("Accuracy", font_weight="bold"),
                    rx.heading(AppState.model_accuracy, size="4"),
                    rx.hstack(
                        rx.cond(
                            AppState.accuracy_change > 0,
                            rx.hstack(
                                rx.icon("arrow_up", color="green.500"),
                                rx.text(f"{abs(AppState.accuracy_change):.2f}% from baseline",
                                        color="green.500", font_size="sm"),
                            ),
                            rx.hstack(
                                rx.icon("arrow_down", color="red.500"),
                                rx.text(f"{abs(AppState.accuracy_change):.2f}% from baseline",
                                        color="red.500", font_size="sm"),
                            ),
                        ),
                    ),
                    align_items="center",
                ),
                # Precision metric
                rx.vstack(
                    rx.text("Precision", font_weight="bold"),
                    rx.heading(AppState.model_precision, size="4"),
                    align_items="center",
                ),
                # Recall metric
                rx.vstack(
                    rx.text("Recall", font_weight="bold"),
                    rx.heading(AppState.model_recall, size="4"),
                    align_items="center",
                ),
                # F1 Score metric
                rx.vstack(
                    rx.text("F1 Score", font_weight="bold"),
                    rx.heading(AppState.model_f1, size="4"),
                    align_items="center",
                ),
                # AUC metric
                rx.vstack(
                    rx.text("AUC", font_weight="bold"),
                    rx.heading(AppState.model_auc, size="4"),
                    align_items="center",
                ),
                width="100%",
                justify="between",
                spacing="4",
            ),
            rx.text("Select and load a model to view performance metrics"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def confusion_matrix():
    return rx.box(
        rx.heading("Confusion Matrix", size="3"),
        rx.cond(
            AppState.metrics_loaded,
            rx.vstack(
                rx.plotly(
                    data=[{
                        "z": AppState.confusion_matrix,
                        "x": ["Predicted Negative", "Predicted Positive"],
                        "y": ["Actual Negative", "Actual Positive"],
                        "type": "heatmap",
                        "colorscale": "Blues",
                        "showscale": False,
                        "hovertemplate": "Prediction: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
                    }],
                    layout={
                        "title": "Confusion Matrix",
                        "xaxis": {"title": "Predicted Label"},
                        "yaxis": {"title": "Actual Label"}
                    },
                    width="100%",
                    height="300px",
                ),
                rx.text("TN: " + str(AppState.confusion_matrix[0][0]) +
                       " | FP: " + str(AppState.confusion_matrix[0][1]) +
                       " | FN: " + str(AppState.confusion_matrix[1][0]) +
                       " | TP: " + str(AppState.confusion_matrix[1][1])),
            ),
            rx.text("Load a model to view the confusion matrix"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def roc_curve():
    return rx.box(
        rx.heading("ROC Curve", size="3"),
        rx.cond(
            AppState.metrics_loaded,
            rx.plotly(
                data=[
                    {
                        "x": AppState.fpr,
                        "y": AppState.tpr,
                        "type": "scatter",
                        "mode": "lines",
                        "name": f"ROC curve (area = {AppState.model_auc:.2f})",
                    },
                    {
                        "x": [0, 1],
                        "y": [0, 1],
                        "type": "scatter",
                        "mode": "lines",
                        "name": "Random",
                        "line": {"dash": "dash"},
                    }
                ],
                layout={
                    "title": "Receiver Operating Characteristic",
                    "xaxis": {"title": "False Positive Rate"},
                    "yaxis": {"title": "True Positive Rate"},
                    "showlegend": True,
                    "legend": {"x": 0.1, "y": 0.9},
                },
                width="100%",
                height="400px",
            ),
            rx.text("Load a model to view the ROC curve"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def model_comparison():
    return rx.box(
        rx.heading("Model Comparison", size="3"),
        rx.hstack(
            rx.checkbox_group(
                AppState.available_comparison_models,
                value=AppState.selected_comparison_models,
                on_change=AppState.set_comparison_models,
            ),
            rx.button(
                "Compare Models",
                on_click=AppState.compare_models,
                color_scheme="green",
                size="2",
                is_disabled=rx.cond(len(AppState.selected_comparison_models) > 1, False, True),
            ),
            width="100%",
            justify="between",
        ),
        rx.cond(
            AppState.comparison_ready,
            rx.vstack(
                rx.plotly(
                    data=[{
                        "x": AppState.comparison_models,
                        "y": AppState.comparison_accuracy,
                        "type": "bar",
                        "name": "Accuracy",
                        "marker": {"color": "rgba(58, 71, 80, 0.6)"}
                    }, {
                        "x": AppState.comparison_models,
                        "y": AppState.comparison_f1,
                        "type": "bar",
                        "name": "F1 Score",
                        "marker": {"color": "rgba(246, 78, 139, 0.6)"}
                    }],
                    layout={
                        "title": "Model Performance Comparison",
                        "xaxis": {"title": "Model"},
                        "yaxis": {"title": "Score"},
                        "barmode": "group",
                    },
                    width="100%",
                    height="400px",
                ),
            ),
            rx.text("Select at least two models to compare"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def feature_importance():
    return rx.box(
        rx.heading("Feature Importance", size="3"),
        rx.cond(
            AppState.metrics_loaded and AppState.feature_importance_available,
            rx.plotly(
                data=[{
                    "x": AppState.feature_importance,
                    "y": AppState.feature_names,
                    "type": "bar",
                    "orientation": "h",
                    "marker": {
                        "color": "rgba(50, 171, 96, 0.6)",
                    },
                }],
                layout={
                    "title": "Feature Importance",
                    "xaxis": {"title": "Importance"},
                    "yaxis": {"title": "Feature"},
                    "margin": {"l": 150},
                },
                width="100%",
                height="400px",
            ),
            rx.text("Feature importance not available for this model"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def prediction_distribution():
    return rx.box(
        rx.heading("Prediction Distribution", size="3"),
        rx.cond(
            AppState.metrics_loaded,
            rx.plotly(
                data=[{
                    "x": AppState.prediction_scores,
                    "type": "histogram",
                    "nbinsx": 30,
                    "marker": {"color": "rgba(0, 107, 164, 0.6)"},
                }],
                layout={
                    "title": "Prediction Score Distribution",
                    "xaxis": {"title": "Prediction Score"},
                    "yaxis": {"title": "Count"},
                },
                width="100%",
                height="300px",
            ),
            rx.text("Load a model to view prediction distribution"),
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb=4,
    )


def evaluation_page():
    return rx.container(
        rx.vstack(
            rx.heading("Model Evaluation", size="1"),
            rx.text("Evaluate and compare model performance"),

            model_selection(),
            rx.hstack(
                rx.box(
                    performance_metrics(),
                    confusion_matrix(),
                    width="50%",
                ),
                rx.box(
                    roc_curve(),
                    width="50%",
                ),
                spacing="4",
            ),
            prediction_distribution(),
            feature_importance(),
            model_comparison(),

            width="100%",
            spacing="4",
        ),
        padding="6",
    )