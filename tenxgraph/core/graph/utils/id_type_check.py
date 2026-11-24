"""Check that the checkpointer's id columns can hold the ids the bound generator makes."""

from tenxgraph.utils.id_generator import BaseIDGenerator, IDType


def check_id_type(checkpointer: object, generator: BaseIDGenerator) -> None:
    """Raise ``ValueError`` when ``generator`` makes ids the checkpointer cannot store.

    Only a checkpointer that declares an id type (``declared_id_type``, ``PgCheckpointer``) is
    checked; the others store any id. Text ids need text columns, integer ids ``int`` or
    ``bigint`` ones, and ``bigint`` ids (Snowflake) need ``bigint``.
    """
    declared = getattr(checkpointer, "declared_id_type", None)
    if not isinstance(declared, IDType):
        return
    makes = generator.id_type
    wrong = (makes == IDType.STRING) != (declared == IDType.STRING) or (
        makes == IDType.BIGINT and declared == IDType.INTEGER
    )
    if not wrong:
        return
    name = type(checkpointer).__name__
    if getattr(checkpointer, "_id_type_given", True):
        fix = "bigint" if makes == IDType.BIGINT else "int" if makes == IDType.INTEGER else "string"
        advice = (
            f'Pass id_type="{fix}" to {name}(...) (deprecated: drop it and let the id type '
            "follow the id generator), or use an id generator whose ids it stores "
            "(StateGraph(id_generator=...))."
        )
    else:
        advice = (
            f"{name} fixed its id type when its tables were set up, with another id "
            "generator bound. Bind the same id generator before the first use."
        )
    raise ValueError(
        f"The id generator {type(generator).__name__} makes {makes.value} ids but "
        f"{name} stores {declared.value} ids. {advice}"
    )
