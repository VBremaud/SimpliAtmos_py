# simpliatmos/io/io.py

import xarray as xr
import numpy as np
import os

class IO:
    def __init__(self, param, mesh, state, time):
        self.param = param
        self.mesh = mesh
        self.state = state
        self.time = time

        self.filename = param.output_file or "output.nc"
        self.fields = {
            "u": (state.U.x, "m/s"),
            "v": (state.U.y, "m/s"),
            "b": (state.b, "m2/s"),
        }

        self._init_storage()

    def _init_storage(self):
        self.times = []
        self.data = {name: [] for name in self.fields}

        self.coords = {
            "x": ("x", self.mesh.x),
            "y": ("y", self.mesh.y),
        }

    def write(self, time):
        self.times.append(time.t)
        for name, (var, _) in self.fields.items():
            self.data[name].append(var.copy())  # sécurité

        if time.finished:
            self._write_netcdf()

    def _write_netcdf(self):
        data_vars = {}

        for name, vals in self.data.items():
            unit = self.fields[name][1]
            arr = np.array(vals)  # (nt, ny, nx)
            data_vars[name] = (("time", "y", "x"), arr, {"units": unit})

        coords = self.coords.copy()
        coords["time"] = ("time", np.array(self.times))

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs["description"] = "SimpliAtmos output"
        ds.to_netcdf(self.filename)
        print(f"\n[NetCDF] Fichier sauvegardé : {self.filename}")
