from typing import Dict, Type, Generic, TypeVar

# Registers hold a generic type
T = TypeVar('T')


class Register(Generic[T]):
    """
    Wraps a dictionary to provide a case-insensitive register
    """
    def __init__(self, initial_values: Dict[str, T] = None):
        # Create a copy of the passed values to prevent modification
        initial_values = initial_values.copy() if initial_values else {}
        self.__register: Dict[str, T] = {}
        for key, value in initial_values.items():
            self[key] = value

    def keys(self):
        return self.__register.keys()

    def values(self):
        return self.__register.values()

    def __getitem__(self, key: str) -> T:
        key = key.lower()
        if key not in self.__register:
            raise KeyError(f"Key {key} not found in register")
        return self.__register[key]

    def __setitem__(self, key: str, value: T):
        key = key.lower()
        if key in self.__register:
            raise ValueError(f"Key {key} already exists in register")
        self.__register[key] = value

    def __str__(self):
        return f"Register({self.__register})"


class TypedRegisterRegister:
    """
    Wraps a dictionary to provide typesafe access to a register
    """
    def __init__(self):
        self.__register: Dict[Type, Register[any]] = {}

    def keys(self):
        return self.__register.keys()

    def values(self):
        return self.__register.values()

    def __getitem__(self, key: Type[T]) -> Register[T]:
        if key not in self.__register:
            raise KeyError(f"Key {key} not found in register")
        return self.__register[key]

    def __setitem__(self, key: Type[T], value: Register[T]):
        if key in self.__register:
            raise ValueError(f"Key {key} already exists in register")
        self.__register[key] = value

    def __str__(self):
        return f"TypedRegisterRegister({self.__register})"
