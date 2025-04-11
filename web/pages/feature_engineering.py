import reflex as rx
from web.models.app_state import AppState


def dimensionality_reduction_config():
    return rx.box(
        rx.heading("Dimensionality Reduction", size="3"),
        rx.hstack(
            rx.switch(
                "Enable Dimensionality Reduction",
                is_checked=AppState.enable_dim_reduction,
                on_change=AppState.set_enable_dim_reduction,
            ),
            rx.select(
                ["pca", "tsne", "umap"],  # Updated to match your implementations
                label="Reduction Method",
                value=AppState.dim_reduction_method,
                on_change=AppState.set_dim_reduction_method,
                is_disabled=rx.cond(AppState.enable_dim_reduction, False, True),
            ),
        ),
        rx.input(
            label="Target Dimensions",
            type="number",
            value=AppState.target_dimensions,
            on_change=AppState.set_target_dimensions,
            step=1,
            min_=2,
            max_=100,
            is_disabled=rx.cond(AppState.enable_dim_reduction, False, True),
        ),

        # TSNE specific parameters
        rx.hstack(
            rx.input(
                label="Perplexity",
                type="number",
                value=AppState.tsne_perplexity,
                on_change=AppState.set_tsne_perplexity,
                step=1,
                min_=5,
                max_=50,
            ),
            rx.input(
                label="Iterations",
                type="number",
                value=AppState.tsne_iterations,
                on_change=AppState.set_tsne_iterations,
                step=100,
                min_=100,
                max_=5000,
            ),
            is_disabled=rx.cond(
                rx.and_(AppState.enable_dim_reduction, AppState.dim_reduction_method == "tsne"),
                False, True
            ),
        ),

        # UMAP specific parameters
        rx.hstack(
            rx.input(
                label="Neighbors",
                type="number",
                value=AppState.umap_neighbors,
                on_change=AppState.set_umap_neighbors,
                step=1,
                min_=5,
                max_=50,
            ),
            rx.input(
                label="Min Distance",
                type="number",
                value=AppState.umap_min_dist,
                on_change=AppState.set_umap_min_dist,
                step=0.01,
                min_=0.01,
                max_=0.99,
            ),
            is_disabled=rx.cond(
                rx.and_(AppState.enable_dim_reduction, AppState.dim_reduction_method == "umap"),
                False, True
            ),
        ),
        mb=4,
    )


def data_transformation_config():
    return rx.container(
        rx.vstack(
            rx.heading("Data Transformation", size="2", mb=4),

            # Scaling/Normalization
            rx.box(
                rx.heading("Scaling", size="3"),
                rx.select(
                    ["none", "standard", "minmax", "robust"],
                    label="Scaling Method",
                    value=AppState.scaling_method,
                    on_change=AppState.set_scaling_method,
                ),
                mb=4,
            ),

            dimensionality_reduction_config(),

            # Feature Engineering - Pipeline Components
            rx.box(
                rx.heading("Feature Engineering Pipeline", size="3"),
                rx.switch(
                    "Use Enhanced Feature Pipeline",
                    is_checked=AppState.use_enhanced_pipeline,
                    on_change=AppState.set_use_enhanced_pipeline,
                ),

                # Autoencoder parameters
                rx.box(
                    rx.heading("Autoencoder", size="4"),
                    rx.hstack(
                        rx.input(
                            label="Hidden Dimension",
                            type="number",
                            value=AppState.ae_hidden_dim,
                            on_change=AppState.set_ae_hidden_dim,
                            step=8,
                            min_=8,
                            max_=512,
                        ),
                        rx.input(
                            label="Latent Dimension",
                            type="number",
                            value=AppState.ae_latent_dim,
                            on_change=AppState.set_ae_latent_dim,
                            step=1,
                            min_=2,
                            max_=100,
                        ),
                    ),
                    rx.hstack(
                        rx.input(
                            label="Epochs",
                            type="number",
                            value=AppState.ae_epochs,
                            on_change=AppState.set_ae_epochs,
                            step=5,
                            min_=5,
                            max_=200,
                        ),
                        rx.input(
                            label="Batch Size",
                            type="number",
                            value=AppState.ae_batch_size,
                            on_change=AppState.set_ae_batch_size,
                            step=8,
                            min_=8,
                            max_=128,
                        ),
                    ),
                    is_disabled=rx.cond(AppState.use_enhanced_pipeline, False, True),
                    padding="2",
                    border="1px solid",
                    border_color="gray.200",
                    border_radius="md",
                    mt=2,
                ),

                # Isolation Forest parameters
                rx.box(
                    rx.heading("Isolation Forest", size="4"),
                    rx.hstack(
                        rx.input(
                            label="Number of Estimators",
                            type="number",
                            value=AppState.if_n_estimators,
                            on_change=AppState.set_if_n_estimators,
                            step=10,
                            min_=10,
                            max_=200,
                        ),
                        rx.input(
                            label="Contamination",
                            type="number",
                            value=AppState.if_contamination,
                            on_change=AppState.set_if_contamination,
                            step=0.01,
                            min_=0.01,
                            max_=0.5,
                        ),
                    ),
                    is_disabled=rx.cond(AppState.use_enhanced_pipeline, False, True),
                    padding="2",
                    border="1px solid",
                    border_color="gray.200",
                    border_radius="md",
                    mt=2,
                ),

                # LOF parameters
                rx.box(
                    rx.heading("Local Outlier Factor", size="4"),
                    rx.input(
                        label="Number of Neighbors",
                        type="number",
                        value=AppState.lof_n_neighbors,
                        on_change=AppState.set_lof_n_neighbors,
                        step=1,
                        min_=5,
                        max_=50,
                    ),
                    is_disabled=rx.cond(AppState.use_enhanced_pipeline, False, True),
                    padding="2",
                    border="1px solid",
                    border_color="gray.200",
                    border_radius="md",
                    mt=2,
                ),
                mb=4,
            ),

            # Outlier Handling
            rx.box(
                rx.heading("Outlier Treatment", size="3"),
                rx.select(
                    ["none", "clip", "remove", "iqr"],
                    label="Outlier Handling Method",
                    value=AppState.outlier_method,
                    on_change=AppState.set_outlier_method,
                ),
                rx.input(
                    label="Outlier Threshold",
                    type="number",
                    value=AppState.outlier_threshold,
                    on_change=AppState.set_outlier_threshold,
                    step=0.1,
                    min_=1.0,
                    max_=10.0,
                    is_disabled=rx.cond(AppState.outlier_method == "none", True, False),
                ),
                mb=4,
            ),

            # Missing Value Handling
            rx.box(
                rx.heading("Missing Values", size="3"),
                rx.select(
                    ["none", "mean", "median", "most_frequent", "knn"],
                    label="Missing Value Strategy",
                    value=AppState.missing_value_strategy,
                    on_change=AppState.set_missing_value_strategy,
                ),
                mb=4,
            ),

            # Apply transformations button
            rx.button(
                "Apply Transformations",
                on_click=AppState.apply_feature_transformations,
                color_scheme="purple",
                size="2",
                is_loading=AppState.is_transforming,
                is_disabled=AppState.is_transforming,
                width="100%",
            ),

            spacing="4",
            width="100%",
            align_items="stretch",
        ),
        padding="6",
        margin_top="2",
    )


def feature_preview():
    return rx.box(
        rx.heading("Data Preview", size="3"),
        rx.cond(
            AppState.data_transformed,
            rx.vstack(
                rx.text("Transformed dataset shape: {0} rows × {1} features".format(
                    AppState.transformed_rows,
                    AppState.transformed_cols
                )),

                # Feature importance visualization
                rx.cond(
                    AppState.feature_importance_available,
                    rx.vstack(
                        rx.heading("Feature Importance", size="4"),
                        rx.plotly(
                            data=[{
                                "x": AppState.feature_names,
                                "y": AppState.feature_importance,
                                "type": "bar",
                            }],
                            layout={"title": "Feature Importance"},
                            width="100%",
                            height="300px",
                        ),
                    ),
                    rx.text("Feature importance not available")
                ),

                # Display sample of transformed data in a table
                rx.heading("Data Sample", size="4"),
                rx.data_table(
                    data=AppState.data_sample,
                    pagination=True,
                    search=True,
                    sort=True,
                ),

                width="100%",
            ),
            rx.text("Apply transformations to preview the processed data")
        ),
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        margin_y="4",
        width="100%",
    )


def feature_engineering_page():
    return rx.container(
        rx.vstack(
            rx.heading("Feature Engineering", size="1"),
            rx.text("Configure and apply data transformations and feature selection"),

            data_transformation_config(),
            feature_preview(),

            width="100%",
            spacing="4",
        ),
        padding="6",
    )