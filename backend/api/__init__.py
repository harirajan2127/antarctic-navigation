"""FastAPI route modules for the DSS REST API.

- ``navigation_routes`` : the legacy ``/api/v1/routes/optimize`` endpoint
  (mounted by ``app.main``).
- ``health``, ``datasets``, ``sea_ice``, ``icebergs``, ``routes``,
  ``analytics`` : routers mounted by the top-level ``main.py`` application
  under the ``/api`` prefix.
"""