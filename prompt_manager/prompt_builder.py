"""
This module provides functionality for rendering prompts using Jinja2 templates.
It sets up a Jinja2 environment and offers a function to render templates with given context.
"""

from importlib import resources
from typing import Optional
from jinja2 import Environment, FileSystemLoader, TemplateError

# Set up Jinja2 environment
with resources.path('prompt_manager', 'prompts') as prompts_path:
    env = Environment(loader=FileSystemLoader(prompts_path))

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
    
    except TemplateError:
        # Raise a RuntimeError if there's an issue with template rendering
        print(f"render_prompt: Issue rendering template: '{template_name}'")
        return None
