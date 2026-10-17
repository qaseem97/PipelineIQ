# Writing testable requirements

Each requirement describes one capability from the user's point of view. Start the description with "The system must ...". Avoid words that cannot be measured, such as fast, easy, user-friendly, flexible or robust; replace them with concrete conditions.

# Acceptance criteria

Every requirement needs two to four acceptance criteria. Each criterion is a single observable condition that an automated test can check, written as Given / When / Then or as an input and expected result. Include at least one criterion for invalid input, for example an empty name, a negative amount, a duplicate record or an unknown identifier, and state exactly what should happen (an error is raised, the record is rejected, nothing changes).

# Scope for a first version

Prefer three to six requirements that together form a usable core. Leave out authentication, persistence to disk, networking, notifications and user interfaces unless the idea is specifically about them. Data can live in memory for the first version.

# Common entities

For systems that manage records (tasks, books, orders, members), the usual core is: create a record with validation, list or view records, update the key state of a record, remove a record, and one domain rule that makes the product useful (for example sorting by priority, calculating a fee or preventing double booking).

# Dates and money

When requirements involve dates, use ISO format (YYYY-MM-DD) and say how past or invalid dates are treated. When they involve money, say whether amounts are whole units or decimals and how rounding works.
