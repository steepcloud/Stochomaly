import reflex as rx

config = rx.Config(
    app_name="app",
    api_url="/api",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
    frontend_port=3000,
    backend_port=8000,
)