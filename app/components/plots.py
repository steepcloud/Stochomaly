import reflex as rx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


def scatter_plot(
        x: List[float],
        y: List[float],
        title: str = "Scatter Plot",
        x_label: str = "X",
        y_label: str = "Y",
        color: Optional[List[Any]] = None,
        color_label: str = "Class"
):
    """Create a scatter plot for data visualization.

    Args:
        x: X-axis values
        y: Y-axis values
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        color: Optional values for color-coding points
        color_label: Label for color legend
    """
    fig = px.scatter(
        x=x,
        y=y,
        color=color,
        labels={"x": x_label, "y": y_label, "color": color_label},
        title=title
    )

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def distribution_plot(
        data: List[float],
        title: str = "Distribution",
        x_label: str = "Value",
        bin_count: int = 30
):
    """Create a histogram showing the distribution of values.

    Args:
        data: List of values to plot
        title: Plot title
        x_label: X-axis label
        bin_count: Number of bins for histogram
    """
    fig = px.histogram(
        x=data,
        nbins=bin_count,
        title=title,
        labels={"x": x_label, "y": "Frequency"}
    )

    fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def correlation_heatmap(
        data: Dict[str, List[float]],
        title: str = "Correlation Matrix"
):
    """Create a correlation heatmap for features.

    Args:
        data: Dictionary with feature names as keys and feature values as lists
        title: Plot title
    """
    features = list(data.keys())
    values = np.array(list(data.values()))
    corr_matrix = np.corrcoef(values)

    fig = px.imshow(
        corr_matrix,
        x=features,
        y=features,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title
    )

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
        overflow="auto",
    )


def roc_curve(
        fpr: List[float],
        tpr: List[float],
        auc: float,
        title: str = "ROC Curve"
):
    """Create a ROC curve plot.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under the curve value
        title: Plot title
    """
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"AUC = {auc:.3f}"
    ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Classifier"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def precision_recall_curve(
        precision: List[float],
        recall: List[float],
        auc: float,
        title: str = "Precision-Recall Curve"
):
    """Create a precision-recall curve plot.

    Args:
        precision: Precision values
        recall: Recall values
        auc: Area under the curve value
        title: Plot title
    """
    fig = go.Figure()

    # Add PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"AUC = {auc:.3f}"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)")
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def residual_plot(
        y_true: List[float],
        y_pred: List[float],
        title: str = "Residual Plot"
):
    """Create a residual plot for regression models.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    residuals = [true - pred for true, pred in zip(y_true, y_pred)]

    fig = go.Figure()

    # Add scatter plot of residuals
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        marker=dict(color="blue", opacity=0.6),
        name="Residuals"
    ))

    # Add horizontal reference line at y=0
    fig.add_trace(go.Scatter(
        x=[min(y_pred), max(y_pred)],
        y=[0, 0],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Zero Line"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return rx.box(
        rx.plotly(fig),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )