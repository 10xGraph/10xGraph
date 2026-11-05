"""Check that the checkpointer and the store declare the same tenant columns."""

import logging
import os

from tenxgraph.core.exceptions import GraphError


logger = logging.getLogger("tenxgraph.graph")

_PRODUCTION_MODES = frozenset({"production", "prod"})


def _is_production() -> bool:
    """``MODE=production`` (or ``prod``), the same switch the API server reads."""
    return os.environ.get("MODE", "").strip().lower() in _PRODUCTION_MODES


def _tenant_signature(backend: object) -> dict[str, str | None]:
    """``{name: type}`` of a backend's tenant columns (type ``None`` when it does not say)."""
    columns = getattr(backend, "columns", None)
    tenant = getattr(columns, "tenant", None)
    if isinstance(tenant, tuple) and all(hasattr(spec, "type") for spec in tenant):
        return {spec.name: spec.type for spec in tenant}
    names = getattr(backend, "tenant_column_names", ())
    if not isinstance(names, tuple | list | frozenset | set):
        return {}  # not a declaration (for example a test double)
    return dict.fromkeys(name for name in names if isinstance(name, str))


def check_tenant_columns(checkpointer: object, store: object | None) -> None:
    """Refuse (production) or warn (otherwise) when checkpointer and store isolate differently.

    Threads and memories of one request must be isolated by the same tenant columns: if the
    checkpointer filters by ``org_id`` and the store does not (or uses other columns, or other
    types), memories of two tenants with the same ``user_id`` mix while their threads do not.
    ``MODE=production`` makes this a ``GraphError``; any other mode logs a warning. Nothing is
    checked without a store.
    """
    if store is None:
        return
    cp_columns = _tenant_signature(checkpointer)
    store_columns = _tenant_signature(store)
    if not cp_columns and not store_columns:
        return
    same_names = set(cp_columns) == set(store_columns)
    same_types = all(
        cp_columns[name] is None
        or store_columns[name] is None
        or cp_columns[name] == store_columns[name]
        for name in cp_columns
        if name in store_columns
    )
    if same_names and same_types:
        return
    message = (
        f"The checkpointer ({type(checkpointer).__name__}) declares tenant columns "
        f"{cp_columns or 'none'} but the store ({type(store).__name__}) declares "
        f"{store_columns or 'none'}. Threads and memories would be isolated differently. "
        "Declare the same tenant_columns on both."
    )
    if _is_production():
        raise GraphError(
            message=message + " Refusing to compile with MODE=production.",
            error_code="GRAPH_TENANT_001",
            context={"checkpointer": cp_columns, "store": store_columns},
        )
    logger.warning(message)
