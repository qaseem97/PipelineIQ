# Test naming and traceability

Name every test after the requirement it verifies: test_req_<number>_<behaviour>, for example test_req_2_rejects_empty_title. Each requirement id needs at least one test, and each acceptance criterion should map to a test.

# Structure

Use pytest functions, not unittest classes. Create a fresh service object in each test (or in a fixture) so tests do not share state. Follow Arrange, Act, Assert, with one behaviour per test.

# Testing errors

Check error cases with `with pytest.raises(ValueError):` (or the exception the design names). Do not assert on the exact wording of error messages unless the requirement specifies it.

# Imports

Import only public interfaces from the app package, for example `from app.service import TaskService`. Use exactly the class and method names in the design and the code under test. Do not import private helpers or test internal attributes.

# Dates

When behaviour depends on the current date, pass an explicit date to the code instead of relying on today's date, so tests are deterministic.

# Common mistakes

Do not assume ids start at a particular number; use the id returned by the create call. Do not depend on the order of results unless the requirement defines an order. Keep tests independent of each other.
