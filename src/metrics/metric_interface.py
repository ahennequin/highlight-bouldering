class MetricInterface:
    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("Must implement compute method")
