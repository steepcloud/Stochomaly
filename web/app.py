import reflex as rx
from web.pages import (
    index,
    data_selection,
    feature_engineering,
    model_training,
    reinforcement_learning,
    evaluation
)

app = rx.App()

# Add routes to all pages
app.add_page("/", index.index_page)
app.add_page("/data", data_selection.data_selection_page)
app.add_page("/features", feature_engineering.feature_engineering_page)
app.add_page("/training", model_training.model_training_page)
app.add_page("/reinforcement", reinforcement_learning.reinforcement_learning_page)
app.add_page("/evaluation", evaluation.evaluation_page)