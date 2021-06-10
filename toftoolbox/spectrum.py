from __future__ import annotations
from typing import Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def aoq_from_t(t, t0, c0):
    return ((t-t0)/c0)**2

def t_from_aoq(aoq, t0, c0):
    return t0 + c0 * np.sqrt(aoq)


class Spectrum:
    def __init__(self, t: np.ndarray, y: np.ndarray, invert_y: bool = False, **kwargs):
        self.t = t
        self.dt = t[1] - t[0]
        self.y = y
        if invert_y:
            self.y *= -1

        self.t0 = None
        self.c0 = None

        self.label = None
        self.t_breed = None

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @classmethod
    def from_RandS_csv(cls, filename: str, invert_y: bool = False, **kwargs) -> Spectrum:
        df = pd.read_csv(filename)
        t = df.iloc[:, 0]
        y = df.iloc[:, 1]
        return cls(t, y, invert_y, **kwargs)

    @property
    def aoq(self) -> np.ndarray:
        assert self.t0 is not None
        assert self.c0 is not None
        return aoq_from_t(self.t, self.t0, self.c0)

    def set_param(self, t0: float, c0: float) -> None:
        self.t0 = t0
        self.c0 = c0

    def set_param_by_fit(self, ts: np.ndarray, aoqs: np.ndarray, return_fig: bool = True) -> Union[None, Any]:
        bounds = ((-np.inf, 0), (ts.min(), np.inf)) #(mins, maxs)
        popt, pcov = curve_fit(aoq_from_t, ts, aoqs, bounds=bounds)

        self.set_param(*popt)

        if return_fig:
            fig, ax = plt.subplots()
            ax.plot(ts, aoqs, "rs")
            t_ = np.linspace(ts.min(), ts.max(), 100)
            ax.plot(t_, aoq_from_t(t_, *popt), "k-")
            ax.set(
                xlabel="$t$ (s)",
                ylabel="$A/Q$",
            )
            return fig


    def plot_data_vs_t(self, ax: plt.Axes = None) -> Any:
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(
                xlabel = "$t$ (s)",
                ylabel = "Intensity (a.u.)",
            )
        ax.plot(self.t, self.y, label=self.label)
        return ax.get_figure()

    def plot_data_vs_aoq(self, ax: plt.Axes = None) -> Any:
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(
                xlabel = "$A/Q$",
                ylabel = "Intensity (a.u.)",
            )
        ax.plot(self.aoq, self.y, label=self.label)
        return ax.get_figure()

    def extract_histogram(self, a:Union[int, float], qs:np.ndarray, peakwidth: float) -> np.ndarray:
        hw = int((peakwidth / 2) // self.dt)
        hw = max(hw, 1)
        hist = []
        for q in qs:
            idx = np.argmin(np.abs(self.aoq - a/q))
            hist.append(np.sum(self.y[idx-hw:idx+hw]))
        return np.asarray(hist)

    def plot_histogram(self, a:Union[int, float], qs:np.ndarray, peakwidth: float, ax: plt.Axes = None, barplot: bool = True) -> Any:
        hist = self.extract_histogram(a, qs, peakwidth)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(
                xlabel = f"$Q\,(A = {a})$",
                ylabel = "Intensity (a.u.)",
            )
        if barplot:
            ax.bar(qs, hist, label=self.label)
        else:
            ax.plot(qs, hist, "s:", label=self.label)



class SpectraCollection:
    def __init__(self, spectra, sortby="t_breed") -> None:
        for s in spectra:
            assert getattr(s, sortby) is not None
        self.spectra = sorted(spectra, key=lambda x: getattr(x, sortby))
        self.sortval = [getattr(x, sortby) for x in self.spectra]

    def __len__(self):
        return len(self.spectra)

    def _empty_figure(self, single_plot: bool = False) -> Any:
        if single_plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            axs = [ax] * len(self)

        else:
            fig, axs = plt.subplots(len(self), figsize=(10, len(self)*3), sharex=True)
        return fig,axs

    def plot_data_vs_t(self, single_plot: bool = False) -> Any:
        fig, axs = self._empty_figure(single_plot)
        axs[-1].set(
            xlabel = "$t$ (s)",
        )

        for s, ax in zip(self.spectra, axs):
            s.plot_data_vs_t(ax=ax)
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend()

        return fig

    def plot_data_vs_aoq(self, single_plot: bool = False) -> Any:
        fig, axs = self._empty_figure(single_plot)
        axs[-1].set(
            xlabel = "$A/Q$",
        )

        for s, ax in zip(self.spectra, axs):
            s.plot_data_vs_aoq(ax=ax)
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend()

        return fig

    def plot_histogram(self, a: Union[int, float], qs: np.ndarray, peakwidth: float, barplot: bool = True, single_plot: bool = False) -> Any:
        fig, axs = self._empty_figure(single_plot)
        axs[-1].set(
            xlabel =  f"$Q\,(A = {a})$",
        )

        for s, ax in zip(self.spectra, axs):
            s.plot_histogram(a, qs, peakwidth, barplot=barplot, ax=ax)
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend()

        return fig
