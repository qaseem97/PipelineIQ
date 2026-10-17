# Layering

Use a small layered design: models (dataclasses holding data), a repository or store (keeps records in memory, looks them up by id), and a service (business rules and validation). Callers and tests use the service. Put each layer in its own module under app/.

# Interfaces

Every interface lists its full Python signature with type hints and return type, and describes its behaviour for valid input and for every error case. Name the exception type for each error. Prefer ValueError for invalid input and KeyError or a custom NotFoundError for unknown identifiers, and use the same choice consistently across the design.

# Identifiers

Generate ids inside the system (an incrementing integer or uuid4 string). Never require the caller to invent ids. Return the created object or its id from create methods so tests can use it.

# Traceability

Every module and interface lists the requirement ids it satisfies. Before finishing, check that each requirement id appears at least once. A requirement with no interface is a design gap.

# Determinism

Avoid hidden dependencies on the current time or randomness in business rules. If a rule depends on today's date (for example overdue items or late fees), accept the date as a parameter with a sensible default so it can be tested.
