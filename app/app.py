import reflex as rx

app = rx.App()

# routes to all pages
app.add_page("/", "web.pages.index.create")
app.add_page("/data_selection", "web.pages.data_selection.data_selection_page")
app.add_page("/feature_engineering", "web.pages.feature_engineering.feature_engineering_page")
app.add_page("/model_training", "web.pages.model_training.model_training_page")
app.add_page("/reinforcement_learning", "web.pages.reinforcement_learning.reinforcement_learning_page")
app.add_page("/evaluation", "web.pages.evaluation.evaluation_page")