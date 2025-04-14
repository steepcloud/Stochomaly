import reflex as rx
from app.models.app_state import AppState
from app.models.training_state import TrainingState


def nn_config():
    return rx.container(
        rx.vstack(
            rx.heading("Neural Network Configuration", size="2", mb=4),

            # Model architecture
            rx.box(
                rx.heading("Network Architecture", size="3"),
                rx.hstack(
                    rx.select(
                        ["mlp", "lstm", "gru", "cnn"],
                        label="Model Type",
                        value=AppState.nn_model_type,
                        on_change=AppState.set_nn_model_type,
                    ),
                    rx.input(
                        label="Hidden Layers",
                        type="number",
                        value=AppState.nn_hidden_layers,
                        on_change=AppState.set_nn_hidden_layers,
                        step=1,
                        min_=1,
                        max_=5,
                    ),
                ),
                rx.input(
                    label="Units per Layer",
                    type="number",
                    value=AppState.nn_units,
                    on_change=AppState.set_nn_units,
                    step=8,
                    min_=8,
                    max_=512,
                ),
                mb=4,
            ),

            # Training parameters
            rx.box(
                rx.heading("Training Parameters", size="3"),
                rx.hstack(
                    rx.input(
                        label="Learning Rate",
                        type="number",
                        value=TrainingState.nn_learning_rate,
                        on_change=lambda val: setattr(TrainingState, "nn_learning_rate", val),
                        step=0.0001,
                        min_=0.0001,
                        max_=0.1,
                    ),
                    rx.input(
                        label="Batch Size",
                        type="number",
                        value=TrainingState.nn_batch_size,
                        on_change=lambda val: setattr(TrainingState, "nn_batch_size", val),
                        step=8,
                        min_=8,
                        max_=256,
                    ),
                    rx.input(
                        label="Epochs",
                        type="number",
                        value=TrainingState.nn_epochs,
                        on_change=lambda val: setattr(TrainingState, "nn_epochs", val),
                        step=1,
                        min_=1,
                        max_=100,
                    ),
                ),
                mb=4,
            ),

            # Regularization
            rx.box(
                rx.heading("Regularization", size="3"),
                rx.hstack(
                    rx.input(
                        label="Dropout Rate",
                        type="number",
                        value=AppState.nn_dropout_rate,
                        on_change=AppState.set_nn_dropout_rate,
                        step=0.05,
                        min_=0.0,
                        max_=0.5,
                    ),
                    rx.input(
                        label="L2 Regularization",
                        type="number",
                        value=AppState.nn_l2_reg,
                        on_change=AppState.set_nn_l2_reg,
                        step=0.0001,
                        min_=0.0,
                        max_=0.01,
                    ),
                ),
                mb=4,
            ),

            # Optimization
            rx.box(
                rx.heading("Optimizer", size="3"),
                rx.select(
                    ["adam", "rmsprop", "sgd"],
                    label="Optimizer Type",
                    value=AppState.nn_optimizer,
                    on_change=AppState.set_nn_optimizer,
                ),
                mb=4,
            ),

            # Loss function
            rx.box(
                rx.heading("Loss Function", size="3"),
                rx.select(
                    ["mse"],
                    label="Loss Function",
                    value=AppState.nn_loss_function,
                    on_change=AppState.set_nn_loss_function,
                ),
                mb=4,
            ),

            # Start training button
            rx.button(
                "Start Neural Network Training",
                on_click=AppState.train_neural_network,
                color_scheme="blue",
                size="2",
                is_loading=TrainingState.is_training,
                is_disabled=TrainingState.is_training,
                width="100%",
            ),

            spacing="4",
            width="100%",
            align_items="stretch",
        ),
        padding="6",
        margin_top="2",
    )


def nn_training_progress():
    return rx.box(
        rx.heading("Training Progress", size="3"),
        rx.cond(
            TrainingState.is_training,
            rx.vstack(
                rx.hstack(
                    rx.box(
                        rx.text("Epoch", font_weight="bold"),
                        rx.text(f"{TrainingState.current_epoch}/{TrainingState.total_epochs}", font_size="xl"),
                        rx.text(f"{(TrainingState.current_epoch / TrainingState.total_epochs * 100):.1f}% complete",
                                color="gray.600", font_size="sm"),
                        padding="2",
                    ),
                    rx.box(
                        rx.text("Training Loss", font_weight="bold"),
                        rx.text(f"{TrainingState.loss_history[-1] if TrainingState.loss_history else 0:.4f}",
                                font_size="xl"),
                        padding="2",
                    ),
                    rx.box(
                        rx.text("Validation Loss", font_weight="bold"),
                        rx.text(f"{TrainingState.val_loss_history[-1] if TrainingState.val_loss_history else 0:.4f}",
                                font_size="xl"),
                        padding="2",
                    ),
                    width="100%",
                    justify="between",
                ),
                rx.progress(value=TrainingState.training_progress * 100, width="100%"),

                # Add charts if you have data
                rx.cond(
                    len(TrainingState.loss_history) > 0,
                    rx.vstack(
                        rx.heading("Training History", size="4"),
                        rx.plotly(
                            data=[
                                {
                                    "x": [i for i in range(len(TrainingState.loss_history))],
                                    "y": TrainingState.loss_history,
                                    "type": "scatter",
                                    "mode": "lines",
                                    "name": "Training Loss",
                                },
                                {
                                    "x": [i for i in range(len(TrainingState.val_loss_history))],
                                    "y": TrainingState.val_loss_history,
                                    "type": "scatter",
                                    "mode": "lines",
                                    "name": "Validation Loss",
                                }
                            ],
                            layout={"title": "Loss During Training"},
                            width="100%",
                            height="300px",
                        ),
                    ),
                    rx.text("No training data yet"),
                ),

                width="100%",
            ),
            rx.text("Click 'Start Neural Network Training' to begin training")
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        margin_y="4",
        width="100%",
    )


def model_training_page():
    return rx.container(
        rx.vstack(
            rx.heading("Neural Network Training", size="1"),
            rx.text("Configure and train a neural network for anomaly detection"),

            nn_config(),
            nn_training_progress(),

            width="100%",
            spacing="4",
        ),
        padding="6",
    )