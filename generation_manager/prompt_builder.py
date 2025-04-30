"""
This module provides functionality for rendering prompts using Jinja2 templates.

It sets up a Jinja2 environment and offers a function to render templates with given context.
Errors during rendering are handled gracefully, and a function to list available templates is also provided.
"""

from importlib import resources
from typing import Optional
from jinja2 import Environment, FileSystemLoader, TemplateError, UndefinedError, StrictUndefined

# Set up Jinja2 environment
with resources.path('generation_manager', 'prompts') as prompts_path:
    env = Environment(
        loader=FileSystemLoader(prompts_path),
        undefined=StrictUndefined,
    )

def render_prompt(template_name: str, context: dict = None) -> Optional[str]:
    """
    Render a prompt template with the given context.

    Args:
        template_name (str): The name of the template file to render (e.g. 'prompt_name.html')
        context (dict, optional): A dictionary of variables to pass to the template. Defaults to None.

    Returns:
        str: The rendered template as a string or None.
    """
    try:
        # Render the template
        template = env.get_template(template_name)
        return template.render(context or {})
    
    except UndefinedError as e:
        # Handle undefined variables in the template context
        print(f"render_prompt: Undefined variable in template '{template_name}': {e}")
        return None
    
    except TemplateError:
        # Raise a RuntimeError if there's an issue with template rendering
        print(f"render_prompt: Issue rendering template: '{template_name}'")
        return None

def list_available_templates() -> list[str]:
    """
    Returns a list of all available template filenames in the prompts directory.

    Returns:
        list[str]: List of template filenames.
    """
    try:
        return env.list_templates()
    except Exception as e:
        print(f"list_available_templates: Issue listing templates: {e}")
        return []
