"""Service layer for the DSS REST API.

Wraps the legacy ``app.services`` data loaders/forecasters/predictors behind
stable, demo-aware contracts that the ``/api`` routers expose. Also contains
standalone utilities (distances, dataset inventory, analytics aggregation,
navigation engine orchestration).
"""
__all__ = []