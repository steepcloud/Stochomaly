import reflex as rx

app = rx.App()

# routes to all pages
app.add_page("/", "pages.index.create")
app.add_page("/data_selection", "pages.data_selection.data_selection_page")
app.add_page("/feature_engineering", "pages.feature_engineering.feature_engineering_page")
app.add_page("/model_training", "pages.model_training.model_training_page")
app.add_page("/reinforcement_learning", "pages.reinforcement_learning.reinforcement_learning_page")
app.add_page("/evaluation", "pages.evaluation.evaluation_page")