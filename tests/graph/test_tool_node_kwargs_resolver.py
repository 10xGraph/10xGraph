import inspect

import pytest

from tenxgraph.core.graph.tool_node.kwargs_resolver import KwargsResolverMixin


class _Resolver(KwargsResolverMixin):
    pass


class _Container:
    def __init__(self, values=None):
        self.values = values or {}

    def has(self, name):
        return name in self.values

    def get(self, name):
        return self.values[name]


def test_should_skip_parameter_for_varargs_and_kwargs():
    resolver = _Resolver()

    def sample(a, *args, **kwargs):
        return a

    params = inspect.signature(sample).parameters
    assert resolver._should_skip_parameter(params["args"]) is True
    assert resolver._should_skip_parameter(params["kwargs"]) is True
    assert resolver._should_skip_parameter(params["a"]) is False


def test_handle_injectable_parameter_prefers_injectable_value():
    resolver = _Resolver()

    def sample(dep):
        return dep

    param = inspect.signature(sample).parameters["dep"]
    value = resolver._handle_injectable_parameter(
        "dep",
        param,
        {"dep": "from_injectable"},
        _Container({"dep": "from_container"}),
    )

    assert value == "from_injectable"


def test_handle_injectable_parameter_falls_back_to_container_when_none():
    resolver = _Resolver()

    def sample(dep):
        return dep

    param = inspect.signature(sample).parameters["dep"]
    value = resolver._handle_injectable_parameter(
        "dep",
        param,
        {"dep": None},
        _Container({"dep": "from_container"}),
    )

    assert value == "from_container"


def test_handle_injectable_parameter_raises_for_missing_required():
    resolver = _Resolver()

    def sample(dep):
        return dep

    param = inspect.signature(sample).parameters["dep"]
    with pytest.raises(TypeError, match="Required injectable parameter 'dep' not found"):
        resolver._handle_injectable_parameter("dep", param, {}, _Container())


def test_handle_injectable_parameter_returns_none_for_defaulted_param():
    resolver = _Resolver()

    def sample(dep="default"):
        return dep

    param = inspect.signature(sample).parameters["dep"]
    assert resolver._handle_injectable_parameter("dep", param, {}, _Container()) is None


def test_get_parameter_value_from_args_then_container_then_default():
    resolver = _Resolver()

    def sample(a, b, c="ok"):
        return a, b, c

    sig = inspect.signature(sample)
    container = _Container({"b": 20})

    assert resolver._get_parameter_value("a", sig.parameters["a"], {"a": 10}, {}, container) == 10
    assert resolver._get_parameter_value("b", sig.parameters["b"], {}, {}, container) == 20
    assert resolver._get_parameter_value("c", sig.parameters["c"], {}, {}, container) is None


def test_get_parameter_value_raises_for_missing_required_param():
    resolver = _Resolver()

    def sample(required):
        return required

    sig = inspect.signature(sample)
    with pytest.raises(TypeError, match="Missing required parameter 'required' for function"):
        resolver._get_parameter_value("required", sig.parameters["required"], {}, {}, _Container())


def test_prepare_kwargs_skips_variadics_and_omits_none_values():
    resolver = _Resolver()

    def sample(a, dep="x", *args, **kwargs):
        return a, dep, args, kwargs

    sig = inspect.signature(sample)
    result = resolver._prepare_kwargs(
        sig,
        args={"a": 1},
        injectable_params={"dep": None},
        dependency_container=_Container(),
    )

    assert result == {"a": 1}
