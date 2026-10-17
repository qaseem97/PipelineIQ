# Python coding standards

Target Python 3.11 and the standard library only. Use dataclasses for models, type hints on every public function, and f-strings. Keep functions short and give them clear names that match the design exactly; tests import these names, so a renamed class or method breaks the build.

# Package layout

Code lives in the app package. Always include app/__init__.py. Import between modules with absolute imports such as `from app.models import Task`, never relative imports like `from models import Task`.

# Validation and errors

Validate input at the service layer before changing any state. Raise the exception type named in the design with a helpful message. Strip whitespace from text input before checking that it is non-empty. Never fail silently by returning None where the design expects an error.

# Common bugs to avoid

Do not use a mutable default argument such as `def f(items=[])`; use `field(default_factory=list)` in dataclasses. Return copies of internal lists so callers cannot modify internal state by accident. When sorting with several keys, put None values last explicitly. Compare dates as date objects, not strings, unless they are ISO formatted.

# Fixing failing tests

When tests fail, read the assertion message and traceback first. Fix the code so it meets the requirement; do not change behaviour that other passing tests rely on. Return the complete content of every file, not only the changed lines.
