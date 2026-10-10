"""KwargsResolverMixin — resolves kwargs for tool functions with DI support."""

from __future__ import annotations

import inspect
import typing as t


class KwargsResolverMixin:
    def _should_skip_parameter(self, param: inspect.Parameter) -> bool:
        return param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

    def _handle_injectable_parameter(
        self,
        p_name: str,
        param: inspect.Parameter,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        if p_name in injectable_params:
            injectable_value = injectable_params[p_name]
            if injectable_value is not None:
                return injectable_value

        if dependency_container and dependency_container.has(p_name):
            return dependency_container.get(p_name)

        if param.default is inspect._empty:
            raise TypeError(f"Required injectable parameter '{p_name}' not found")

        return None

    def _get_parameter_value(
        self,
        p_name: str,
        param: inspect.Parameter,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> t.Any | None:
        if p_name in injectable_params:
            return self._handle_injectable_parameter(
                p_name, param, injectable_params, dependency_container
            )

        value_sources = [
            lambda: args.get(p_name),
            lambda: (
                dependency_container.get(p_name)
                if dependency_container and dependency_container.has(p_name)
                else None
            ),
        ]

        for source in value_sources:
            value = source()
            if value is not None:
                return value

        if param.default is not inspect._empty:
            return None

        raise TypeError(f"Missing required parameter '{p_name}' for function")

    def _prepare_kwargs(
        self,
        sig: inspect.Signature,
        args: dict,
        injectable_params: dict,
        dependency_container,
    ) -> dict:
        kwargs: dict = {}
        for p_name, p in sig.parameters.items():
            if self._should_skip_parameter(p):
                continue
            value = self._get_parameter_value(
                p_name, p, args, injectable_params, dependency_container
            )
            if value is not None:
                kwargs[p_name] = value
        return kwargs
