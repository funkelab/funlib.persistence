class Freezable:
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self!r} is frozen, you can't add attributes to it")
        object.__setattr__(self, key, value)

    def freeze(self):
        self.__isfrozen = True

    def thaw(self):
        self.__isfrozen = False
