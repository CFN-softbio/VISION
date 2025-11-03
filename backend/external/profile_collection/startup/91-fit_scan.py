# Classes and functions to make it easy to do a dscan with realtime fitting to
# a custom function.

print(f'Loading {__file__}')

import glob
from bluesky.callbacks import CallbackBase


class SavingCallback(CallbackBase):
    """This call back saves data in a data element.

    Access each element by calling
    scallback.data[key] etc...
    """

    def __init__(self, *args, **kwargs):
        super(SavingCallback, self).__init__(*args, **kwargs)
        self.data = dict()

    def descriptor(self, doc):
        for key, val in doc["data_keys"].items():
            self.data[key] = list()

    def event(self, doc):
        for key, val in doc["data"].items():
            self.data[key].append(val)


class PrintCallback(CallbackBase):
    """For debugging only
    Exposes raw documents
    """

    def start(self, doc):
        print("Start document: ")
        print(doc)
        print("END")

    def event(self, doc):
        print("Events document: ")
        print(doc)
        print("END")

    def descriptor(self, doc):
        print("Descriptor document: ")
        print(doc)
        print("END")

    def stop(self, doc):
        print("Stop document: ")
        print(doc)
        print("END")


def remove_last_Pilatus_series():
    """Delete the last Pilatus image tiff. If the last image is a series of files
    with sequential IDs, they are all deleted."""

    # chars = caget('XF:11BMB-ES{Det:SAXS}:TIFF1:FullFileName_RBV')
    # chars = caget('XF:11BMB-ES{Det:SAXS}:TIFF1:FullFileName_RBV')

    chars = caget("XF:11BMB-ES{}:TIFF1:FullFileName_RBV".format(pilatus_Epicsname))
    chars = caget("XF:11BMB-ES{}:TIFF1:FullFileName_RBV".format(pilatus_Epicsname))
    filename = "".join(chr(char) for char in chars)[:-1]
    filenames = filename[:-11] + "*" + ".tiff"
    for filename in glob.glob(filenames):
        # print(filename)
        os.remove(filename)


if False:
    # %run -i /opt/ipython_profiles/profile_collection/startup/91-fit_scan.py

    # Define a 'fake' detector, for testing purposes
    from bluesky.examples import Reader

    def fake_detector_response_peak():
        pos = armz.user_readback.value
        A = 1000.0
        x0 = -40.0
        sigma = 0.1
        I = A * np.exp(-((pos - x0) ** 2) / (2 * sigma**2)) + 10.0

        return np.random.poisson(I)

    def fake_detector_response_edge():
        pos = armz.user_readback.value
        A = 1000.0
        x0 = -17.0
        sigma = 0.05
        I = A / (1 + np.exp(-(pos - x0) / (-sigma))) + 10.0

        return np.random.poisson(I)

    # det = Reader( 'det', {'intensity': lambda: 1.0*( (DETx.user_readback.value - (-40.0))**2 )/(2.*(0.1)**2) } )
    det = Reader("intensity", {"intensity": fake_detector_response_edge})
    detselect(det)
    # detselect(det, suffix='')

    # fit_scan(DETx, 1, 3, detector_suffix='')
    # fit_scan(armz, [-5,0], 5, detector_suffix='')


class MotorWait(CallbackBase):
    def __init__(self, motor, wait_time):
        self.motor = motor
        self.wait_time = wait_time

    def start(self, doc):
        self.settle_time_original = self.motor.settle_time
        self.motor.settle_time = self.wait_time

    def stop(self, doc):
        self.motor.settle_time = self.settle_time_original


class LiveTable_Custom(LiveTable):
    """Simple customization of LiveTable that preserves the x and y data."""

    # Based on:
    # /opt/conda_envs/collection-17Q1.0/lib/python3.5/site-packages/bluesky/callbacks/core.py
    # class LiveTable(CallbackBase)

    def __init__(self, fields, y_name, x_name):
        super().__init__(fields)

        self.y_name = y_name
        self.x_name = x_name
        self.ydata = []
        self.xdata = []

    def start(self, doc):
        super().start(doc)

    def event(self, doc):
        if self.y_name not in doc["data"]:
            return

        y = doc["data"][self.y_name]
        x = doc["data"][self.x_name]

        self.ydata.append(y)
        self.xdata.append(x)

        super().event(doc)

    def stop(self, doc):
        super().stop(doc)


class LiveStat(CallbackBase):
    """
    Calculate simple statistics for an (x,y) curve.
    """

    # Note: Follows the style/naming of class LiveFit(CallbackBase),
    # where possible, so that it can be used in similar contexts.

    def __init__(self, stat, y_name, x_name, update_every=1):
        self.stat = stat
        self.y_name = y_name
        self.x_name = x_name
        self.update_every = update_every

        self.ydata = []
        self.xdata = []

        class Result(object):
            pass

        self.result = Result()  # Dummy object to replicate the hiearchy expected for LiveFit
        self.result.values = {}

    def event(self, doc):
        if self.y_name not in doc["data"]:
            return

        y = doc["data"][self.y_name]
        x = doc["data"][self.x_name]

        self.ydata.append(y)
        self.xdata.append(x)

        if self.update_every is not None:
            i = doc["seq_num"]
            if (i - 1) % self.update_every == 0:
                if type(self.stat) is list:
                    for stat in self.stat:
                        self.update_fit(stat)
                else:
                    self.update_fit(self.stat)

        super().event(doc)

    def update_fit(self, stat):
        xs = np.asarray(self.xdata)
        ys = np.asarray(self.ydata)

        if stat is "max":
            idx = np.argmax(ys)
            x0 = xs[idx]
            y0 = ys[idx]

            self.result.values["x_max"] = x0
            self.result.values["y_max"] = y0

        elif stat is "min":
            idx = np.argmin(ys)
            x0 = xs[idx]
            y0 = ys[idx]

            self.result.values["x_min"] = x0
            self.result.values["y_min"] = y0

        elif stat is "COM":
            x0 = np.sum(xs * ys) / np.sum(ys)
            y0 = np.interp(x0, xs, ys)

            self.result.values["x_COM"] = x0
            self.result.values["y_COM"] = y0

        elif stat is "HM":
            """Half-maximum, using the point(s) closest to HM."""
            idx_max = np.argmax(ys)
            half_max = 0.5 * ys[idx_max]

            l = None
            r = None

            left = ys[:idx_max]
            right = ys[idx_max:]

            if len(left) > 0 and left.min() < half_max:
                idx_hm = np.abs(left - half_max).argmin()
                l = xs[:idx_max][idx_hm]
            if len(right) > 0 and right.min() < half_max:
                idx_hm = np.abs(right - half_max).argmin()
                r = xs[idx_max:][idx_hm]

            if l is None:
                x0 = r
            elif r is None:
                x0 = l
            else:
                x0 = np.average([l, r])

            if x0 is None:
                x0 = np.average(xs)

            y0 = np.interp(x0, xs, ys)
            self.result.values["x_HM"] = x0
            self.result.values["y_HM"] = y0

        elif stat is "HMi":
            """Half-maximum, with averaging of values near HW."""
            idx_max = np.argmax(ys)
            half_max = 0.5 * ys[idx_max]

            l = None
            r = None

            left = ys[:idx_max]
            right = ys[idx_max:]

            if len(left) > 0 and left.min() < half_max and left.max() > half_max:
                idx = np.where(left < half_max)[0][-1]
                l = np.average([xs[:idx_max][idx], xs[:idx_max][idx + 1]])
            if len(right) > 0 and right.min() < half_max and right.max() > half_max:
                idx = np.where(right < half_max)[0][0]
                r = np.average([xs[idx_max:][idx - 1], xs[idx_max:][idx]])

            if l is None:
                x0 = r
            elif r is None:
                x0 = l
            else:
                x0 = np.average([l, r])

            if x0 is None:
                x0 = np.average(xs)

            y0 = np.interp(x0, xs, ys)
            self.result.values["x_HM"] = x0
            self.result.values["y_HM"] = y0

        else:
            print("ERROR: Statistic type {} is not recognized.".format(stat))

        # print('Update_fit: ({:g}, {:g})'.format(x0, y0))
        self.result.values["x0"] = x0
        self.result.values["y0"] = y0


class LiveStatPlot(LivePlot):
    def __init__(
        self,
        livestat,
        *,
        scan_range=None,
        legend_keys=None,
        xlim=None,
        ylim=None,
        ax=None,
        **kwargs,
    ):
        kwargs_update = {
            "color": "b",
            "linewidth": 0,
            "marker": "o",
            "markersize": 10.0,
        }
        kwargs_update.update(kwargs)

        super().__init__(
            livestat.y_name,
            livestat.x_name,
            legend_keys=legend_keys,
            xlim=xlim,
            ylim=xlim,
            ax=ax,
            **kwargs_update,
        )

        self.livestat = livestat

        self.scan_range = scan_range

    def get_scan_range(self, overscan=0.0):
        if self.scan_range is None:
            x_start = np.min(self.livestat.xdata)
            x_stop = np.max(self.livestat.xdata)
        else:
            x_start = np.min(self.scan_range)
            x_stop = np.max(self.scan_range)

        span = abs(x_stop - x_start)

        x_start -= span * overscan
        x_stop += span * overscan

        return x_start, x_stop, span

    def start(self, doc):
        self.livestat.start(doc)
        super().start(doc)

        for line in self.ax.lines:
            if hasattr(line, "custom_tag_x0") and line.custom_tag_x0:
                line.remove()

        # A line that denotes the current fit position for x0 (e.g. center of gaussian)
        x_start, x_stop, span = self.get_scan_range(overscan=0.0)
        self.x0_line = self.ax.axvline(
            (x_start + x_stop) * 0.5, color="b", alpha=0.5, dashes=[5, 5], linewidth=2.0
        )
        self.x0_line.custom_tag_x0 = True

    def event(self, doc):
        self.livestat.event(doc)

        # Slight kludge (to over-ride possible 'greying out' from LivePlot_Custom.start)
        self.current_line.set_alpha(1.0)
        self.current_line.set_linewidth(2.5)
        self.x0_line.set_alpha(0.5)
        self.x0_line.set_linewidth(2.0)

        if self.livestat.result is not None:
            x0 = self.livestat.result.values["x0"]
            y0 = self.livestat.result.values["y0"]

            self.x_data = [x0]
            self.y_data = [y0]

            self.update_plot()
        # Intentionally override LivePlot.event. Do not call super().

        self.update_plot()

    def update_plot(self):
        super().update_plot()

        self.x0_line.set_xdata([self.x_data[0]])

    def descriptor(self, doc):
        self.livestat.descriptor(doc)
        super().descriptor(doc)

    def stop(self, doc):
        self.livestat.stop(doc)
        # Intentionally override LivePlot.stop. Do not call super().


class LivePlot_Custom(LivePlot):
    def __init__(
        self,
        y,
        x=None,
        *,
        legend_keys=None,
        xlim=None,
        ylim=None,
        ax=None,
        fig=None,
        **kwargs,
    ):
        kwargs_update = {
            "color": "k",
            "linewidth": 3.5,
        }
        kwargs_update.update(kwargs)

        rcParams_update = {
            "figure.figsize": (11, 7),
            "figure.facecolor": "white",
            "font.size": 14,
            "axes.labelsize": 16,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.borderpad": 0.1,
            "legend.labelspacing": 0.1,
            "legend.columnspacing": 1.5,
        }
        # For more rcParam options: http://matplotlib.org/users/customizing.html
        plt.matplotlib.rcParams.update(rcParams_update)

        super().__init__(
            y,
            x,
            legend_keys=legend_keys,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            fig=fig,
            **kwargs_update,
        )
        # super().setup()

        # self.ax.figure.canvas.manager.toolbar.pan()

    def start(self, doc):
        print(f"*** LivePlot_Custom.start() got {doc}")
        # Make all the 'older' lines greyed-out
        for line in self.ax.lines:
            alpha = line.get_alpha()
            if alpha is None:
                alpha = 1.0
            alpha = max(alpha * 0.75, 0.1)
            line.set_alpha(alpha)

            lw = line.get_linewidth()
            if lw is None:
                lw = 1.0
            lw = max(lw * 0.5, 0.2)
            line.set_linewidth(lw)

        super().start(doc)
        self.ax.figure.canvas.mpl_connect("scroll_event", self.scroll_event)

    def update_plot(self):
        ymin = min(self.y_data)
        ymax = max(self.y_data)
        yspan = ymax - ymin

        # If the data is 'reasonable' (strictly positive and close to zero),
        # then force the plotting range to something sensible
        if ymin >= 0 and yspan > 0 and ymin / yspan < 0.25:
            self.ax.set_ylim([0, ymax * 1.2])

        super().update_plot()

    def scroll_event(self, event):
        """Gets called when the mousewheel/scroll-wheel is used. This activates
        zooming."""

        if event.inaxes != self.ax:
            return

        current_plot_limits = self.ax.axis()
        x = event.xdata
        y = event.ydata

        # The following function converts from the wheel-mouse steps
        # into a zoom-percentage. Using a base of 4 and a divisor of 2
        # means that each wheel-click is a 50% zoom. However, the speed
        # of zooming can be altered by changing these numbers.

        # 50% zoom:
        step_percent = 4.0 ** (-event.step / 2.0)
        # Fast zoom:
        # step_percent = 100.0**( -event.step/2.0 )
        # Slow zoom:
        # step_percent = 2.0**( -event.step/4.0 )

        xi = x - step_percent * (x - current_plot_limits[0])
        xf = x + step_percent * (current_plot_limits[1] - x)
        yi = y - step_percent * (y - current_plot_limits[2])
        yf = y + step_percent * (current_plot_limits[3] - y)

        self.ax.axis((xi, xf, yi, yf))

        self.ax.figure.canvas.draw()

    def add_line(self, x_data, y_data, **kwargs):
        (self.current_line,) = self.ax.plot(x_data, y_data, **kwargs)
        self.lines.append(self.current_line)
        self.legend = self.ax.legend(loc=0, title=self.legend_title)  # .draggable()


class LiveFitPlot_Custom(LiveFitPlot):
    """
    Add a plot to an instance of LiveFit.

    Note: If your figure blocks the main thread when you are trying to
    scan with this callback, call `plt.ion()` in your IPython session.

    Parameters
    ----------
    livefit : LiveFit
        an instance of ``LiveFit``
    legend_keys : list, optional
        The list of keys to extract from the RunStart document and format
        in the legend of the plot. The legend will always show the
        scan_id followed by a colon ("1: ").  Each
    xlim : tuple, optional
        passed to Axes.set_xlim
    ylim : tuple, optional
        passed to Axes.set_ylim
    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.
    All additional keyword arguments are passed through to ``Axes.plot``.
    """

    def __init__(
        self,
        livefit,
        *,
        legend_keys=None,
        xlim=None,
        ylim=None,
        ax=None,
        scan_range=None,
        **kwargs,
    ):
        kwargs_update = {
            "color": "b",
            "linewidth": 2.5,
        }
        kwargs_update.update(kwargs)

        super().__init__(
            livefit,
            legend_keys=legend_keys,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            **kwargs_update,
        )

        self.y_guess = 0
        self.scan_range = scan_range

    def get_scan_range(self, overscan=0.0):
        if self.scan_range is None:
            x_start = np.min(self.livefit.independent_vars_data[self.__x_key])
            x_stop = np.max(self.livefit.independent_vars_data[self.__x_key])
        else:
            x_start = np.min(self.scan_range)
            x_stop = np.max(self.scan_range)

        span = abs(x_stop - x_start)

        x_start -= span * overscan
        x_stop += span * overscan

        return x_start, x_stop, span

    def event(self, doc):
        # Slight kludge (to over-ride possible 'greying out' from LivePlot_Custom.start)
        self.current_line.set_alpha(1.0)
        self.current_line.set_linewidth(2.5)
        self.x0_line.set_alpha(0.5)
        self.x0_line.set_linewidth(2.0)

        self.livefit.event(doc)
        if self.livefit.result is not None:
            # self.y_data = self.livefit.result.best_fit
            # self.x_data = self.livefit.independent_vars_data[self.__x_key]

            x_start, x_stop, span = self.get_scan_range(overscan=0.25)

            self.x_data = np.linspace(x_start, x_stop, num=200, endpoint=True, retstep=False)
            self.y_data = self.livefit.result.eval(x=self.x_data)

            self.update_plot()

        # Intentionally override LivePlot.event. Do not call super().

    def start(self, doc):
        super().start(doc)

        for line in self.ax.lines:
            if hasattr(line, "custom_tag_x0") and line.custom_tag_x0:
                line.remove()

        # A line that denotes the current fit position for x0 (e.g. center of gaussian)
        x_start, x_stop, span = self.get_scan_range(overscan=0.0)
        self.x0_line = self.ax.axvline(
            (x_start + x_stop) * 0.5, color="b", alpha=0.5, dashes=[5, 5], linewidth=2.0
        )
        self.x0_line.custom_tag_x0 = True

    def update_plot(self):
        x0 = self.livefit.result.values["x0"]
        self.x0_line.set_xdata([x0])
        super().update_plot()


class LiveFit_Custom(LiveFit):
    """
    Fit a model to data using nonlinear least-squares minimization.

    Parameters
    ----------
    model_name : string
        The name of the model to be used in fitting
    y : string
        name of the field in the Event document that is the dependent variable
    independent_vars : dict
        map the independent variable name(s) in the model to the field(s)
        in the Event document; e.g., ``{'x': 'motor'}``
    init_guess : dict, optional
        initial guesses for other values, if expected by model;
        e.g., ``{'sigma': 1}``
    update_every : int or None, optional
        How often to recompute the fit. If `None`, do not compute until the
        end. Default is 1 (recompute after each new point).

    Attributes
    ----------
    result : lmfit.ModelResult
    """

    def __init__(
        self,
        model_name,
        y,
        independent_vars,
        scan_range,
        update_every=1,
        background=None,
    ):
        self.x_start = min(scan_range)
        self.x_stop = max(scan_range)
        self.x_span = abs(self.x_stop - self.x_start)

        substitutions = {
            "gaussian": "gauss",
            "lorentzian": "lorentz",
            "squarewave": "square",
            "tophat": "square",
            "rectangular": "square",
            "errorfunction": "erf",
        }
        if model_name in substitutions.keys():
            model_name = substitutions[model_name]

        lm_model = self.get_model(model_name)
        init_guess = self.get_initial_guess(model_name)

        # Add additional models (if any)
        if background is not None:
            if type(background) is list:
                for back in background:
                    lm_model += self.get_model(back)
                    init_guess.update(self.get_initial_guess(back))
            else:
                lm_model += self.get_model(background)
                init_guess.update(self.get_initial_guess(background))

        super().__init__(
            lm_model,
            y,
            independent_vars,
            init_guess=init_guess,
            update_every=update_every,
        )

    def get_model(self, model_name):
        if model_name is "gauss":

            def model_function(x, x0, prefactor, sigma):
                return prefactor * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

        elif model_name is "lorentz":

            def model_function(x, x0, prefactor, gamma):
                return prefactor * (gamma**2) / ((x - x0) ** 2 + (gamma**2))

        elif model_name is "doublesigmoid":

            def model_function(x, x0, prefactor, sigma, fwhm):
                left = prefactor / (1 + np.exp(-(x - (x0 - fwhm * 0.5)) / sigma))
                right = prefactor / (1 + np.exp(-(x - (x0 + fwhm * 0.5)) / sigma))
                return prefactor * (left - right)

        elif model_name is "square":

            def model_function(x, x0, prefactor, fwhm):
                sigma = fwhm * 0.02
                left = prefactor / (1 + np.exp(-(x - (x0 - fwhm * 0.5)) / sigma))
                right = prefactor / (1 + np.exp(-(x - (x0 + fwhm * 0.5)) / sigma))
                return prefactor * (left - right)

        elif model_name is "sigmoid":

            def model_function(x, x0, prefactor, sigma):
                return prefactor / (1 + np.exp(-(x - x0) / sigma))

        elif model_name is "sigmoid_r":

            def model_function(x, x0, prefactor, sigma):
                return prefactor / (1 + np.exp(+(x - x0) / sigma))

        elif model_name is "step":

            def model_function(x, x0, prefactor, sigma):
                return prefactor / (1 + np.exp(-(x - x0) / sigma))

        elif model_name is "step_r":

            def model_function(x, x0, prefactor, sigma):
                return prefactor / (1 + np.exp(+(x - x0) / sigma))

        elif model_name is "tanh":

            def model_function(x, x0, prefactor, sigma):
                return prefactor * 0.5 * (np.tanh((x - x0) / sigma) + 1.0)

        elif model_name is "tanh_r":

            def model_function(x, x0, prefactor, sigma):
                return prefactor * 0.5 * (np.tanh(-(x - x0) / sigma) + 1.0)

        elif model_name is "erf":
            import scipy

            def model_function(x, x0, prefactor, sigma):
                return prefactor * 0.5 * (scipy.special.erf((x - x0) / sigma) + 1.0)

        elif model_name is "erf_r":
            import scipy

            def model_function(x, x0, prefactor, sigma):
                return prefactor * 0.5 * (scipy.special.erf(-(x - x0) / sigma) + 1.0)

        elif model_name is "constant":

            def model_function(x, offset):
                return x * 0 + offset

        elif model_name is "linear":

            def model_function(x, m, b):
                return m * x + b

        else:
            print("ERROR: Model {:s} unknown.".format(model_name))

        lm_model = lmfit.Model(model_function)

        return lm_model

    def get_initial_guess(self, model_name):
        return getattr(self, "initial_guess_{:s}".format(model_name))()

    def initial_guess_gauss(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 1000, min=0),
            "sigma": lmfit.Parameter("sigma", self.x_span * 0.25, min=0, max=self.x_span * 4),
        }
        return init_guess

    def initial_guess_lorentz(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 1, min=0),
            "gamma": lmfit.Parameter("gamma", self.x_span * 0.25, min=0, max=self.x_span * 4),
        }
        return init_guess

    def initial_guess_doublesigmoid(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 100, min=0),
            "sigma": lmfit.Parameter("sigma", self.x_span * 0.25, min=0, max=self.x_span),
            "fwhm": lmfit.Parameter("fwhm", self.x_span * 0.25, min=0, max=self.x_span),
        }
        return init_guess

    def initial_guess_square(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 100, min=0),
            "fwhm": lmfit.Parameter("fwhm", self.x_span * 0.25, min=0, max=self.x_span),
        }
        return init_guess

    def initial_guess_sigmoid(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 100, min=0),
            "sigma": lmfit.Parameter("sigma", self.x_span * 0.25, min=0, max=self.x_span * 4),
        }
        return init_guess

    def initial_guess_sigmoid_r(self):
        return self.initial_guess_sigmoid()

    def initial_guess_step(self):
        init_guess = {
            "x0": lmfit.Parameter(
                "x0",
                (self.x_start + self.x_stop) * 0.5,
                min=self.x_start - self.x_span * 0.1,
                max=self.x_stop + self.x_span * 0.1,
            ),
            "prefactor": lmfit.Parameter("prefactor", 100, min=0),
            "sigma": lmfit.Parameter("sigma", self.x_span * 0.002, min=0, max=self.x_span * 0.005),
        }
        return init_guess

    def initial_guess_step_r(self):
        return self.initial_guess_step()

    def initial_guess_tanh(self):
        return self.initial_guess_sigmoid()

    def initial_guess_tanh_r(self):
        return self.initial_guess_tanh()

    def initial_guess_erf(self):
        return self.initial_guess_sigmoid()

    def initial_guess_erf_r(self):
        return self.initial_guess_erf()

    def initial_guess_linear(self):
        init_guess = {"m": 0, "b": 0}
        return init_guess

    def initial_guess_constant(self):
        init_guess = {"offset": 0}
        return init_guess


import lmfit


def fit_scan(
    motor,
    span,
    num=11,
    detectors=None,
    detector_suffix="",
    exposure_time=0.5,
    toggle_beam=True,
    fit=None,
    background=None,
    per_step=None,
    wait_time=None,
    md={},
    save_flg=0,
):
    """
    Scans the specified motor, and attempts to fit the data as requested.

    Parameters
    ----------
    motor : motor
        The axis/stage/motor that you want to move.
    span : float
        The total size of the scan range (centered about the current position).
        If a two-element list is instead specified, this is interpreted as the
        distances relative to the current position for the start and end.
    num : int
        The number of scan points.
    fit : None or string
        If None, then fitting is not done. Otherwise, the model specified by the
        supplied string is used.
            peaks: gauss, lorentz, doublesigmoid, square
            edges: sigmoid, step
            stats: max, min, COM (center-of-mass), HM (half-max)
    background : None or string
        A baseline/background underlying the fit function can be specified.
        (In fact, a sequence of summed background functions can be supplied.)
            constant, linear
    md : dict, optional
        metadata
    """

    # TODO: Normalize per ROI pixel and per count_time?
    # TODO: save scan data with save_flg=1.

    if toggle_beam:
        beam.on()

    if not beam.is_on():
        print("WARNING: Experimental shutter is not open.")

    initial_position = motor.user_readback.value

    if type(span) is list:
        start = initial_position + span[0]
        stop = initial_position + span[1]
    else:
        start = initial_position - span / 2.0
        stop = initial_position + span / 2.0
    span = abs(stop - start)
    # positions, dp = np.linspace(start, stop, num, endpoint=True, retstep=True)

    if detectors is None:
        # detselect(pilatus_name, suffix='_stats4_total')
        detectors = get_beamline().detector
        plot_y = get_beamline().PLOT_Y

        # plot_y = pilatus2M.stats4.total
        # print("plot_y is {}".format(plot_y))

    else:
        plot_y = "{}{}".format(detectors[0].name, detector_suffix)

    # Get axes for plotting
    title = "fit_scan: {} vs. {}".format(detectors[0].name, motor.name)
    # if len(plt.get_fignums())>0:
    # Try to use existing figure
    # fig = plt.gcf() # Most recent figure

    fig = None
    for i in plt.get_fignums():
        title_cur = plt.figure(i).canvas.manager.window.windowTitle()
        if title_cur == title:
            fig = plt.figure(i)
            break

    if fig is None:
        # New figure
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(11, 7), facecolor="white")
        fig.canvas.manager.toolbar.pan()

    fig.canvas.manager.set_window_title(title)
    ax = fig.gca()

    subs = []

    livetable = LiveTable([motor] + list(detectors))
    # subs.append(livetable)
    # liveplot = LivePlot_Custom(plot_y, motor.name, ax=ax)
    liveplot = LivePlot(plot_y, motor.name, ax=ax)
    subs.append(liveplot)

    if wait_time is not None:
        subs.append(MotorWait(motor, wait_time))

    if fit in ["max", "min", "COM", "HM", "HMi"] or type(fit) is list:
        livefit = LiveStat(fit, plot_y, motor.name)

        livefitplot = LiveStatPlot(livefit, ax=ax, scan_range=[start, stop])

        subs.append(livefitplot)

    elif fit is not None:
        # Perform a fit

        # livefit = LiveFit(lm_model, plot_y, {'x': motor.name}, init_guess)
        livefit = LiveFit_Custom(
            fit,
            plot_y,
            {"x": motor.name},
            scan_range=[start, stop],
            background=background,
        )

        # livefitplot = LiveFitPlot(livefit, color='k')
        livefitplot = LiveFitPlot_Custom(livefit, ax=ax, scan_range=[start, stop])

        subs.append(livefitplot)

    md["plan_header_override"] = "fit_scan"
    md["scan"] = "fit_scan"
    md["measure_type"] = "fit_scan_{}".format(motor.name)
    md["fit_function"] = fit
    md["fit_background"] = background

    # cms.SAXS.detector.setExposureTime(exposure_time)
    RE(cms.SAXS.detector.setExposureTime(exposure_time))

    # exposure_time_last = md['exposure_time']
    # md['exposure_time'] = exposure_time

    # Perform the scan

    bec.disable_plots()
    RE(scan(list(detectors), motor, start, stop, num, per_step=per_step, md=md), subs)
    bec.enable_plots()
    # RE(scan(list(detectors), motor, start, stop, num, per_step=per_step, md=md), [liveplot, livefit, livefitplot])
    # RE(scan(list(detectors), motor, start, stop, num, per_step=per_step, md=md), [livefit])

    # md['exposure_time'] = exposure_time_last
    # if plot_y=='pilatus300_stats4_total' or plot_y=='pilatus300_stats3_total':
    if plot_y == "pilatus2M_stats4_total" or plot_y == "pilatus2M_stats3_total":
        remove_last_Pilatus_series()

    # check save_flg and save the scan data thru databroker
    if save_flg == 1:
        header = db[-1]
        dtable = header.table()
        filename = "{}/{}".format(RE.md["experiment_alias_directory"], header.start["scan_id"])
        dtable.to_csv(filename)

    if toggle_beam:
        beam.off()

    if fit is None:
        # Return to start position
        # motor.user_setpoint.set(initial_position)
        # mov(motor, initial_position)
        motor.move(initial_position)

    else:
        print(livefit.result.values)
        x0 = livefit.result.values["x0"]
        # mov(motor, x0)
        motor.move(x0)
        return livefit.result


def fit_edge(
    motor,
    span,
    num=11,
    detectors=None,
    detector_suffix="",
    plot=True,
    toggle_beam=True,
    wait_time=None,
    md={},
):
    """
    Optimized fit_scan for finding a (decreasing) step-edge.

    Parameters
    ----------
    motor : motor
        The axis/stage/motor that you want to move.
    span : float
        The total size of the scan range (centered about the current position).
        If a two-element list is instead specified, this is interpreted as the
        distances relative to the current position for the start and end.
    num : int
        The number of scan points.
    md : dict, optional
        metadata
    """

    if toggle_beam:
        beam.on()

    if not beam.is_on():
        print("WARNING: Experimental shutter is not open.")

    cms.setMonitor(monitor=["stats1", "stats2", "stats3", "stats4"])

    initial_position = motor.user_readback.value

    if type(span) is list:
        start = initial_position + span[0]
        stop = initial_position + span[1]
    else:
        start = initial_position - span / 2.0
        stop = initial_position + span / 2.0
    span = abs(stop - start)

    if detectors is None:
        detectors = get_beamline().detector
        plot_y = get_beamline().PLOT_Y
    else:
        plot_y = "{}{}".format(detectors[0].name, detector_suffix)

    subs = []
    livetable = LiveTable_Custom([motor] + list(detectors), plot_y, motor.name)
    # scallback = SavingCallback()
    # subs.append(scallback)
    # access data with scallback.data['keyname']
    # (gives a list)
    subs.append(livetable)

    if plot:
        # Get axes for plotting
        title = "fit_scan: {} vs. {}".format(detectors[0].name, motor.name)
        fig = None
        for i in plt.get_fignums():
            title_cur = plt.figure(i).canvas.manager.window.windowTitle()
            if title_cur == title:
                fig = plt.figure(i)
                break

        if fig is None:
            # New figure
            # fig, ax = plt.subplots()
            fig = plt.figure(figsize=(11, 7), facecolor="white")
            fig.canvas.manager.toolbar.pan()

        fig.canvas.manager.set_window_title(title)
        ax = fig.gca()

        liveplot = LivePlot_Custom(plot_y, motor.name, ax=ax)
        # liveplot = LivePlot(plot_y, motor.name, ax=ax)
        subs.append(liveplot)

    if wait_time is not None:
        subs.append(MotorWait(motor, wait_time))

    md["plan_header_override"] = "fit_edge"
    md["scan"] = "fit_edge"

    # Perform the scan
    bec.disable_table()
    bec.disable_plots()
    RE(scan(list(detectors), motor, start, stop, num, md=md), subs)
    # RE(scan(list(detectors), motor, start, stop, num, md=md), [liveplot, livetable] )
    bec.enable_plots()
    bec.enable_table()

    # if plot_y=='pilatus300_stats4_total' or plot_y=='pilatus300_stats3_total':
    if plot_y == "pilatus2M_stats4_total" or plot_y == "pilatus2M_stats3_total":
        remove_last_Pilatus_series()

    x0_guess = np.average(livetable.xdata)

    # Determine x0 from half-max (HM) analysis
    if True:
        # TODO: Handle case where more than one pair of points cross the HM
        if len(livetable.xdata) > 3:
            y_max = np.max(livetable.ydata)
            HM = (y_max - np.min(livetable.ydata)) / 2.0

            for ip, (x2, y2) in enumerate(zip(livetable.xdata, livetable.ydata)):
                if ip > 0:
                    x1 = livetable.xdata[ip - 1]
                    yx1 = livetable.ydata[ip - 1]

                    if x1 > HM and x2 < HM:
                        # This pair of points crosses the HM
                        m = (y2 - y1) / (x2 - x1)
                        b = y2 - m * x2
                        xHM = (HM - b) / m
                        x0_guess = xHM

    # Fit to sigmoid_r
    if True:
        y_max = np.max(livetable.ydata)

        x_span = abs(np.max(livetable.xdata) - np.min(livetable.xdata))

        def model(v, x):
            return v["prefactor"] / (1 + np.exp(+(x - v["x0"]) / v["sigma"]))

        def func2minimize(params, x, data):
            v = params.valuesdict()
            m = model(v, x)
            return m - data

        params = lmfit.Parameters()
        if y_max > 0:
            params.add("prefactor", value=y_max * 0.95, min=y_max * 0.90, max=y_max * 1.02)
        else:
            params.add("prefactor", value=y_max * 0.95, min=0, max=1)
        params.add(
            "x0",
            value=x0_guess,
            min=np.min(livetable.xdata) + x_span * 0.05,
            max=np.max(livetable.xdata) - x_span * 0.05,
        )
        params.add("sigma", value=0.014, min=x_span * 1e-4, max=x_span * 0.08)

        # 1st fit: only vary x0
        params["prefactor"].set(vary=False)
        params["sigma"].set(vary=False)
        lm_result = lmfit.minimize(func2minimize, params, args=(livetable.xdata, livetable.ydata))
        # lmfit.report_fit(lm_result.params)

        # 2nd fit: vary everything
        params["prefactor"].set(vary=True)
        params["sigma"].set(vary=True)
        lm_result = lmfit.minimize(func2minimize, params, args=(livetable.xdata, livetable.ydata))
        # lmfit.report_fit(lm_result.params)

        if plot:
            xe = 0.25
            fit_x = np.linspace(
                np.min(livetable.xdata) - xe * x_span,
                np.max(livetable.xdata) + xe * x_span,
                num=500,
            )
            fit_y = model(lm_result.params.valuesdict(), fit_x)
            # liveplot.add_line(fit_x, fit_y, color='b', linewidth=2.5)

        # Detect bad fits
        avg_deviation = np.sum(np.abs(lm_result.residual / y_max)) / len(livetable.ydata)
        print("  avg deviation {:.1f}%".format(avg_deviation * 100))
        # avg_deviation of <1% is good.
        # avg_deviation of 1-4% is normal.
        # avg_deviation of 8% is not good (fit okay, but HM slightly off).
        # avg_deviation of 10% is bad fit.

        # TODO: Change the plotting if the fit is bad. (I.e. since we're using HM instead of fit, show that.)

    if avg_deviation > 0.06:
        x0 = x0_guess
    else:
        x0 = lm_result.params["x0"].value

    print("Moving to x = {:.3f}".format(x0))
    motor.move(x0)

    if toggle_beam:
        beam.off()

    lm_result_values = {k: v.value for k, v in lm_result.params.items()}
    return lm_result_values


def _test_fit_scan(
    motor,
    span,
    num=11,
    detectors=None,
    detector_suffix="",
    fit="HMi",
    background=None,
    per_step=None,
    wait_time=None,
    md={},
):
    """
    Scans the specified motor, and attempts to fit the data as requested.

    Parameters
    ----------
    motor : motor
        The axis/stage/motor that you want to move.
    span : float
        The total size of the scan range (centered about the current position).
        If a two-element list is instead specified, this is interpreted as the
        distances relative to the current position for the start and end.
    num : int
        The number of scan points.
    fit : None or string
        If None, then fitting is not done. Otherwise, the model specified by the
        supplied string is used.
            peaks: gauss, lorentz, doublesigmoid, square
            edges: sigmoid, step
            stats: max, min, COM (center-of-mass), HM (half-max)
    background : None or string
        A baseline/background underlying the fit function can be specified.
        (In fact, a sequence of summed background functions can be supplied.)
            constant, linear
    md : dict, optional
        metadata
    """

    # TODO: Normalize per ROI pixel and per count_time?

    if not beam.is_on():
        print("WARNING: Experimental shutter is not open.")

    initial_position = motor.user_readback.value

    if type(span) is list:
        start = initial_position + span[0]
        stop = initial_position + span[1]
    else:
        start = initial_position - span / 2.0
        stop = initial_position + span / 2.0
    span = abs(stop - start)
    # positions, dp = np.linspace(start, stop, num, endpoint=True, retstep=True)

    if detectors is None:
        detectors = get_beamline().detector
        plot_y = get_beamline().PLOT_Y
    else:
        plot_y = "{}{}".format(detectors[0].name, detector_suffix)

    # Get axes for plotting
    title = "fit_scan: {} vs. {}".format(detectors[0].name, motor.name)
    # if len(plt.get_fignums())>0:
    # Try to use existing figure
    # fig = plt.gcf() # Most recent figure

    fig = None
    for i in plt.get_fignums():
        title_cur = plt.figure(i).canvas.manager.window.windowTitle()
        if title_cur == title:
            fig = plt.figure(i)
            break

    if fig is None:
        # New figure
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(11, 7), facecolor="white")
        fig.canvas.manager.toolbar.pan()

    fig.canvas.manager.set_window_title(title)
    ax = fig.gca()

    subs = []

    livetable = LiveTable([motor] + list(detectors))
    # livetable = LiveTable([motor] + list(detectors))
    subs.append(livetable)
    liveplot = LivePlot_Custom(plot_y, motor.name, ax=ax)
    subs.append(liveplot)

    if wait_time is not None:
        subs.append(MotorWait(motor, wait_time))

    if fit in ["max", "min", "COM", "HM", "HMi"] or type(fit) is list:
        livefit = LiveStat(fit, plot_y, motor.name)

        livefitplot = LiveStatPlot(livefit, ax=ax, scan_range=[start, stop])

        subs.append(livefitplot)

    elif fit is not None:
        # Perform a fit

        # livefit = LiveFit(lm_model, plot_y, {'x': motor.name}, init_guess)
        livefit = LiveFit_Custom(
            fit,
            plot_y,
            {"x": motor.name},
            scan_range=[start, stop],
            background=background,
        )

        # livefitplot = LiveFitPlot(livefit, color='k')
        livefitplot = LiveFitPlot_Custom(livefit, ax=ax, scan_range=[start, stop])

        subs.append(livefitplot)

    md["plan_header_override"] = "fit_scan"
    md["scan"] = "fit_scan"
    md["fit_function"] = fit
    md["fit_background"] = background

    # Perform the scan
    def _fit_scan():
        yield from scan(list(detectors), motor, start, stop, num, per_step=per_step, md=md)
        yield from read(list(detectors))
        # threshold=1
        # print(reading)
        # if reading['det']['value']>threshold:
        # print(reading['det']['value'])
        # print('the scan is DONE')

    RE(_fit_scan())

    #    RE(scan(list(detectors), motor, start, stop, num, per_step=per_step, md=md), subs )

    # if plot_y=='pilatus300_stats4_total' or plot_y=='pilatus300_stats3_total':
    if plot_y == "pilatus2M_stats4_total" or plot_y == "pilatus2M_stats3_total":
        remove_last_Pilatus_series()

    if fit is None:
        # Return to start position
        # motor.user_setpoint.set(initial_position)
        # mov(motor, initial_position)
        motor.move(initial_position)

    else:
        print(livefit.result.values)
        x0 = livefit.result.values["x0"]
        # mov(motor, x0)
        motor.move(x0)

        return livefit.result


def setMonitor(monitor=["stats1", "stats2", "stats3", "stats4"]):
    if monitor == None:
        monitor = ["stats3", "stats4"]

    pilatus2M.read_attrs = ["tiff"] + monitor


# TODO:
# - Fit parameters on graph
# - Correctly guess orientation of sigmoids/etc.
# - Have an 'auto' mode that just picks the right fit-function?
# - Do it silently (for autonomous modes); maybe save to disk
# - Allow fit to be 'redone' (with a different function) at the end.
## Maybe need a global pointer to the last fit? (Or contained within beamline?)

# HMi


# TODO:
# version of fit_scan for use in scripts (only fits at end, does lots of double-checks for sanity...)
# terminate the fit_scan when reaching < threshold
