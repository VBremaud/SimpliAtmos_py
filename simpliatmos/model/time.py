class Time:
    def __init__(self, param):
        self.t = 0.0
        self.ite = 0
        self.dt = param.dt if param.dt > 0 else 0.01
        self.param = param
        self._c = 0.0  # Kahan correction term
        self.ite0 = 0
        self.t0 = 0.0

    @property
    def finished(self):
        return (
            (self.t >= self.param.tend)
            or (self.ite >= self.param.maxite)
        )

    def pushforward(self):
        # Kahan summation to reduce numerical drift
        y = self.dt - self._c
        t_new = self.t + y
        self._c = (t_new - self.t) - y
        self.t = t_new
        self.ite += 1

    def tostring(self):
        return f"t={self.t:.2f}"

    @property
    def save_to_file(self):
        return (self.param.nhis > 0) and (
            self.ite % self.param.nhis == 0 or self.finished
        )
