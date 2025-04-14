import reflex as rx


def form_container(title: str, description: str = "", children=None):
    """A standard container for form sections with title and optional description.

    Args:
        title: The form section title
        description: Optional description text
        children: Form elements to include
    """
    return rx.box(
        rx.heading(title, size="3"),
        rx.text(description) if description else None,
        *children if children else [],
        padding="4",
        border_radius="md",
        border="1px solid",
        border_color="gray.200",
        mb="4",
    )


def number_input(
    label: str,
    value_var,
    on_change,
    min_value=None,
    max_value=None,
    step=None,
    help_text=None
):
    """A standard number input with label and optional help text.

    Args:
        label: Input label
        value_var: State variable to bind
        on_change: State handler for changes
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Step increment
        help_text: Optional help text below input
    """
    return rx.vstack(
        rx.hstack(
            rx.text(label, font_weight="medium"),
            rx.spacer(),
            rx.text(help_text, font_size="xs", color="gray.500") if help_text else None,
            width="100%",
        ),
        rx.input(
            value=value_var,
            type="number",
            on_change=on_change,
            min_=min_value,
            max_=max_value,
            step=step,
            width="100%",
        ),
        align_items="start",
        width="100%",
        spacing="1",
    )


def slider_input(
    label: str,
    value_var,
    on_change,
    min_value=0,
    max_value=100,
    step=1,
    help_text=None,
    show_value=True
):
    """A slider with label and optional displayed value.

    Args:
        label: Slider label
        value_var: State variable to bind
        on_change: State handler for changes
        min_value: Minimum value
        max_value: Maximum value
        step: Step increment
        help_text: Optional help text
        show_value: Whether to show current value
    """
    return rx.vstack(
        rx.hstack(
            rx.text(label, font_weight="medium"),
            rx.spacer(),
            rx.text(help_text, font_size="xs", color="gray.500") if help_text else None,
            width="100%",
        ),
        rx.hstack(
            rx.slider(
                min_=min_value,
                max_=max_value,
                step=step,
                value=value_var,
                on_change=on_change,
                width="100%",
            ),
            rx.text(value_var) if show_value else None,
            width="100%",
        ),
        align_items="start",
        width="100%",
        spacing="1",
    )


def action_button(
    text: str,
    on_click,
    color_scheme="blue",
    size="md",
    is_loading=None,
    is_disabled=None,
    width="auto"
):
    """A standard action button for forms with loading state.

    Args:
        text: Button text
        on_click: Click handler
        color_scheme: Button color scheme
        size: Button size
        is_loading: Loading state variable
        is_disabled: Disabled state variable
        width: Button width
    """
    return rx.button(
        text,
        on_click=on_click,
        color_scheme=color_scheme,
        size=size,
        is_loading=is_loading,
        is_disabled=is_disabled or is_loading,
        width=width,
    )


def parameter_group(title: str, children=None):
    """A group of related parameters with a title.

    Args:
        title: Group title
        children: Parameter inputs
    """
    return rx.box(
        rx.text(title, font_weight="medium", mb="2"),
        rx.vstack(
            *children if children else [],
            pl="4",
            spacing="3",
            width="100%",
            align_items="start",
        ),
        mb="4",
    )