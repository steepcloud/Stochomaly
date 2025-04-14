import reflex as rx
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union


def metric_card(title: str, value: Union[str, float, int], caption: Optional[str] = None):
    """Display a single metric in a card format."""
    return rx.box(
        rx.vstack(
            rx.text(title, font_size="sm", color="gray.500"),
            rx.heading(f"{value}", size="3"),
            rx.text(caption, font_size="xs", color="gray.500") if caption else None,
            align="center",
            spacing="1",
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        width="100%",
    )


def metrics_grid(metrics: Dict[str, Union[str, float, int]]):
    """Display multiple metrics in a grid layout."""
    return rx.grid(
        *[
            metric_card(name, value) for name, value in metrics.items()
        ],
        template_columns="repeat(auto-fit, minmax(200px, 1fr))",
        gap="4",
        width="100%",
    )


def confusion_matrix(
        matrix: List[List[int]],
        labels: List[str] = None,
        color_scheme: str = "blues"
):
    """Display a confusion matrix."""
    if labels is None:
        labels = [str(i) for i in range(len(matrix))]

    # Create table rows
    rows = []
    matrix_size = len(matrix)
    max_value = max([max(row) for row in matrix])

    # Header row
    header_row = rx.hstack(
        rx.box(width="60px"),  # Empty corner cell
        *[
            rx.box(
                rx.text(label, font_weight="bold"),
                text_align="center",
                padding="2",
                width="60px"
            )
            for label in labels
        ],
        width="100%"
    )
    rows.append(header_row)

    # Data rows
    for i, row in enumerate(matrix):
        cells = [rx.box(rx.text(labels[i], font_weight="bold"), padding="2", width="60px")]
        for j, value in enumerate(row):
            # Calculate color intensity
            intensity = int(100 * value / max_value) if max_value > 0 else 0
            bg_color = f"{color_scheme}.{min(900, intensity * 9 + 100)}"
            text_color = "white" if intensity > 50 else "black"

            cells.append(
                rx.box(
                    rx.text(str(value)),
                    text_align="center",
                    padding="2",
                    width="60px",
                    height="60px",
                    bg=bg_color,
                    color=text_color,
                )
            )
        rows.append(rx.hstack(*cells, width="100%"))

    return rx.box(
        rx.heading("Confusion Matrix", size="3", mb="2"),
        rx.vstack(*rows, width="100%"),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
        overflow="auto",
    )


def learning_curve(
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "Accuracy",
        epochs: List[int] = None
):
    """Display a learning curve showing training and validation metrics."""
    if epochs is None:
        epochs = list(range(1, len(train_scores) + 1))

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_scores, mode='lines+markers', name='Training'))
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, mode='lines+markers', name='Validation'))
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title=metric_name,
        legend=dict(orientation="h", y=-0.2),
        height=250,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return rx.box(
        rx.heading(f"Learning Curve - {metric_name}", size="3", mb="2"),
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def feature_importance(features: List[str], importances: List[float], max_features: int = 10):
    """Display feature importance as a horizontal bar chart."""
    # Sort features by importance and take top N
    sorted_indices = np.argsort(importances)[::-1][:max_features]
    top_features = [features[i] for i in sorted_indices]
    top_importances = [importances[i] for i in sorted_indices]

    # Reverse order for better visualization
    top_features.reverse()
    top_importances.reverse()

    # Create Plotly figure
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h'
    ))
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        height=350,
        margin=dict(l=150, r=40, t=40, b=40)
    )

    return rx.box(
        rx.heading("Feature Importance", size="3", mb="2"),
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )