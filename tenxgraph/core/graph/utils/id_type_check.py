"""Check that the checkpointer's id columns can hold the ids the bound generator makes."""

from tenxgraph.utils.id_generator import BaseIDGenerator, IDType


def check_id_type(checkpointer: object, generator: BaseIDGenerator) -> None:
    """Raise ``ValueError`` when ``generator`` makes ids the checkpointer cannot store.

    Only a checkpointer that declares an id type (``declared_id_type``, ``PgCheckpointer``) is
    checked; the others store any id. Text ids need a text setup, integer ids an ``int`` or
    ``bigint`` one, and ``bigint`` ids (Snowflake) need ``bigint``.
    """
    declared = getattr(checkpointer, "declared_id_type", None)
    if not isinstance(declared, IDType):
        return
    makes = generator.id_type
    if makes == IDType.STRING and declared != IDType.STRING:
        fix = 'id_type="string"'
    elif makes != IDType.STRING and declared == IDType.STRING:
        fix = f'id_type="{"bigint" if makes == IDType.BIGINT else "int"}"'
    elif makes == IDType.BIGINT and declared == IDType.INTEGER:
        fix = 'id_type="bigint"'
    else:
        return
    raise ValueError(
        f"The id generator {type(generator).__name__} makes {makes.value} ids but "
        f"{type(checkpointer).__name__} stores {declared.value} ids. Pass {fix} to "
        f"{type(checkpointer).__name__}(...), or use an id generator whose ids it stores "
        "(StateGraph(id_generator=...))."
    )
