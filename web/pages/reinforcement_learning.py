import reflex as rx
from web.models.app_state import AppState


def rl_config():
    return rx.container(
        rx.vstack(
            rx.heading("Reinforcement Learning Configuration", size="2", mb=4),

            # Agent selection
            rx.box(
                rx.heading("Agent Configuration", size="3"),
                rx.select(
                    ["dqn", "double_dqn", "dueling_dqn", "a2c"],
                    label="Agent Type",
                    value=AppState.rl_agent_type,
                    on_change=AppState.set_rl_agent_type,
                ),
                rx.select(
                    ["epsilon-greedy", "softmax"],
                    label="Exploration Policy",
                    value=AppState.rl_policy,
                    on_change=AppState.set_rl_policy,
                ),
                mb=4,
            ),

            # Epsilon-greedy specific parameters
            rx.box(
                rx.heading("Epsilon-Greedy Parameters", size="3"),
                rx.hstack(
                    rx.input(
                        label="Initial Epsilon",
                        type="number",
                        value=AppState.rl_epsilon,
                        on_change=AppState.set_rl_epsilon,
                        step=0.1,
                        min_=0.0,
                        max_=1.0,
                    ),
                    rx.input(
                        label="Min Epsilon",
                        type="number",
                        value=AppState.rl_epsilon_min,
                        on_change=AppState.set_rl_epsilon_min,
                        step=0.01,
                        min_=0.0,
                        max_=1.0,
                    ),
                    rx.input(
                        label="Epsilon Decay",
                        type="number",
                        value=AppState.rl_epsilon_decay,
                        on_change=AppState.set_rl_epsilon_decay,
                        step=0.01,
                        min_=0.8,
                        max_=0.999,
                    ),
                ),
                display=rx.cond(AppState.rl_policy == "epsilon-greedy", "block", "none"),
                mb=4,
            ),

            # Softmax specific parameters
            rx.box(
                rx.heading("Softmax Parameters", size="3"),
                rx.hstack(
                    rx.input(
                        label="Initial Temperature",
                        type="number",
                        value=AppState.rl_temperature,
                        on_change=AppState.set_rl_temperature,
                        step=0.1,
                        min_=0.1,
                        max_=10.0,
                    ),
                    rx.input(
                        label="Min Temperature",
                        type="number",
                        value=AppState.rl_temperature_min,
                        on_change=AppState.set_rl_temperature_min,
                        step=0.05,
                        min_=0.01,
                        max_=1.0,
                    ),
                    rx.input(
                        label="Temperature Decay",
                        type="number",
                        value=AppState.rl_temperature_decay,
                        on_change=AppState.set_rl_temperature_decay,
                        step=0.01,
                        min_=0.8,
                        max_=0.999,
                    ),
                ),
                display=rx.cond(AppState.rl_policy == "softmax", "block", "none"),
                mb=4,
            ),

            # Learning parameters
            rx.box(
                rx.heading("Learning Parameters", size="3"),
                rx.hstack(
                    rx.input(
                        label="Learning Rate",
                        type="number",
                        value=AppState.rl_learning_rate,
                        on_change=AppState.set_rl_learning_rate,
                        step=0.0001,
                        min_=0.0001,
                        max_=0.01,
                    ),
                    rx.input(
                        label="Discount Factor (Gamma)",
                        type="number",
                        value=AppState.rl_gamma,
                        on_change=AppState.set_rl_gamma,
                        step=0.01,
                        min_=0.8,
                        max_=0.999,
                    ),
                ),
                mb=4,
            ),

            # Environment parameters
            rx.box(
                rx.heading("Environment Parameters", size="3"),
                rx.hstack(
                    rx.select(
                        ["f1", "precision", "recall", "accuracy"],
                        label="Reward Metric",
                        value=AppState.rl_reward_metric,
                        on_change=AppState.set_rl_reward_metric,
                    ),
                    rx.switch(
                        "Enable Dynamic Thresholds",
                        is_checked=AppState.enable_dynamic_threshold,
                        on_change=AppState.set_enable_dynamic_threshold,
                    ),
                ),
                rx.hstack(
                    rx.input(
                        label="Min Threshold",
                        type="number",
                        value=AppState.rl_threshold_min,
                        on_change=AppState.set_rl_threshold_min,
                        step=0.01,
                        min_=0.0,
                        max_=0.5,
                    ),
                    rx.input(
                        label="Max Threshold",
                        type="number",
                        value=AppState.rl_threshold_max,
                        on_change=AppState.set_rl_threshold_max,
                        step=0.01,
                        min_=0.5,
                        max_=1.0,
                    ),
                    rx.input(
                        label="Number of Thresholds",
                        type="number",
                        value=AppState.rl_n_thresholds,
                        on_change=AppState.set_rl_n_thresholds,
                        step=1,
                        min_=5,
                        max_=50,
                    ),
                ),
                rx.input(
                    label="Threshold Adjustment Frequency",
                    type="number",
                    value=AppState.adjustment_frequency,
                    on_change=AppState.set_adjustment_frequency,
                    step=1,
                    min_=1,
                    max_=50,
                    display=rx.cond(AppState.enable_dynamic_threshold, "block", "none"),
                ),
                mb=4,
            ),

            # Training parameters
            rx.box(
                rx.heading("Training Parameters", size="3"),
                rx.hstack(
                    rx.input(
                        label="Episodes",
                        type="number",
                        value=AppState.rl_episodes,
                        on_change=AppState.set_rl_episodes,
                        step=10,
                        min_=10,
                        max_=1000,
                    ),
                    rx.input(
                        label="Max Steps per Episode",
                        type="number",
                        value=AppState.rl_max_steps,
                        on_change=AppState.set_rl_max_steps,
                        step=10,
                        min_=10,
                        max_=500,
                    ),
                ),
                rx.hstack(
                    rx.input(
                        label="Batch Size",
                        type="number",
                        value=AppState.rl_batch_size,
                        on_change=AppState.set_rl_batch_size,
                        step=8,
                        min_=8,
                        max_=256,
                    ),
                    rx.input(
                        label="Memory Size",
                        type="number",
                        value=AppState.rl_memory_size,
                        on_change=AppState.set_rl_memory_size,
                        step=1000,
                        min_=1000,
                        max_=100000,
                    ),
                    rx.input(
                        label="Target Network Update Frequency",
                        type="number",
                        value=AppState.rl_target_update,
                        on_change=AppState.set_rl_target_update,
                        step=1,
                        min_=1,
                        max_=100,
                    ),
                ),
                mb=4,
            ),

            # Start training button
            rx.button(
                "Start RL Training",
                on_click=AppState.train_rl_agent,
                color_scheme="green",
                size="2",
                is_loading=AppState.is_training,
                is_disabled=AppState.is_training,
                width="100%",
            ),

            spacing="4",
            width="100%",
            align_items="stretch",
        ),
        padding="6",
        margin_top="2",
    )


def reinforcement_learning_page():
    return rx.container(
        rx.vstack(
            rx.heading("Reinforcement Learning", size="1"),
            rx.text("Configure and train an RL agent for anomaly detection"),

            rl_config(),

            width="100%",
            spacing="4",
        ),
        padding="6",
    )