#!/usr/bin/env python3
"""Self-contained plotting functions for the analytical tornado model.

The public API consists of five functions:

``plot_vorticity_3D``
    Plot the axisymmetric vorticity and three-dimensional velocity field.
``plot_vorticity_2D``
    Plot vorticity, azimuthal wind, and meridional flow in an ``(r,z)`` plane.
``plot_azimuthal_wind_2D``
    Compare the viscous azimuthal-wind solutions for ``b != 0`` and ``b = 0``.
``plot_azimuthal_wind_profile``
    Compare radial azimuthal-wind profiles at one height.
``plot_azimuthal_wind_vertical_profile``
    Compare vertical azimuthal-wind profiles at one radius.

All physical, numerical, and display parameters are passed directly to these
functions.  No notebook state or settings dataclass is required.

Each public function also accepts ``x_label_fontsize``,
``y_label_fontsize``, ``x_tick_fontsize``, ``y_tick_fontsize``,
``contour_label_fontsize``, ``curve_linewidth``, and ``curve_color``.
When no contour labels are present, ``contour_label_fontsize`` controls
the scalar-field colorbar label or the direct time labels on profile curves.
Leaving a style argument as ``None`` preserves that plot's original default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, is_color_like
from scipy.integrate import solve_ivp
from scipy.sparse import diags, lil_matrix

__all__ = [
    "plot_vorticity_3D",
    "plot_vorticity_2D",
    "plot_azimuthal_wind_2D",
    "plot_azimuthal_wind_profile",
    "plot_azimuthal_wind_vertical_profile",
]


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; received {value}.")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _require_integer_at_least(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _validate_plot_style(
    *,
    x_label_fontsize: float | None,
    y_label_fontsize: float | None,
    x_tick_fontsize: float | None,
    y_tick_fontsize: float | None,
    contour_label_fontsize: float | None,
    curve_linewidth: float | None,
    curve_color: str | None,
) -> None:
    optional_sizes = {
        "x_label_fontsize": x_label_fontsize,
        "y_label_fontsize": y_label_fontsize,
        "x_tick_fontsize": x_tick_fontsize,
        "y_tick_fontsize": y_tick_fontsize,
        "contour_label_fontsize": contour_label_fontsize,
        "curve_linewidth": curve_linewidth,
    }
    for name, value in optional_sizes.items():
        if value is not None:
            _require_positive(name, value)
    if curve_color is not None and not is_color_like(curve_color):
        raise ValueError(
            f"curve_color must be a valid Matplotlib color; "
            f"received {curve_color!r}."
        )


def _apply_axis_text_style(
    axis,
    *,
    x_label_fontsize: float | None,
    y_label_fontsize: float | None,
    x_tick_fontsize: float | None,
    y_tick_fontsize: float | None,
) -> None:
    if x_label_fontsize is not None:
        axis.xaxis.label.set_fontsize(x_label_fontsize)
    if y_label_fontsize is not None:
        axis.yaxis.label.set_fontsize(y_label_fontsize)
    if x_tick_fontsize is not None:
        axis.tick_params(axis="x", labelsize=x_tick_fontsize)
    if y_tick_fontsize is not None:
        axis.tick_params(axis="y", labelsize=y_tick_fontsize)


def _set_colorbar_label(
    colorbar,
    label: str,
    fontsize: float | None,
) -> None:
    if fontsize is None:
        colorbar.set_label(label)
    else:
        colorbar.set_label(label, fontsize=fontsize)


def _validate_vorticity_inputs(
    circulation: float,
    a: float,
    diffusivity: float,
    radius_max: float,
    height_max: float,
) -> None:
    _require_finite("circulation", circulation)
    _require_positive("a", a)
    _require_positive("diffusivity", diffusivity)
    _require_positive("radius_max", radius_max)
    _require_positive("height_max", height_max)


def _vorticity_model_fields(
    radius: np.ndarray,
    height: np.ndarray,
    circulation: float,
    a: float,
    diffusivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return zeta, radial velocity, vertical velocity, and swirl velocity."""
    exponent = -a * radius**2 / (4.0 * diffusivity)
    vorticity = (
        circulation
        * a
        / (4.0 * np.pi * diffusivity)
        * np.exp(exponent)
    )
    radial_velocity = -0.5 * a * radius
    vertical_velocity = a * height

    numerator = -np.expm1(exponent)
    azimuthal_velocity = np.zeros_like(radius, dtype=float)
    np.divide(
        circulation * numerator,
        2.0 * np.pi * radius,
        out=azimuthal_velocity,
        where=radius != 0.0,
    )
    return (
        vorticity,
        radial_velocity,
        vertical_velocity,
        azimuthal_velocity,
    )


def _finish_single_figure(
    fig,
    *,
    output_path: str | Path | None,
    dpi: int,
    show: bool,
):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_vorticity_3D(
    *,
    circulation: float,
    a: float,
    diffusivity: float,
    radius_max: float,
    height_max: float,
    nr: int = 10,
    ntheta: int = 16,
    nz: int = 10,
    scatter_size: float = 15.0,
    scatter_alpha: float = 0.30,
    quiver_length: float = 0.30,
    quiver_alpha: float = 0.80,
    cmap: str = "viridis",
    x_label_fontsize: float | None = None,
    y_label_fontsize: float | None = None,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    contour_label_fontsize: float | None = None,
    curve_linewidth: float | None = None,
    curve_color: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 100,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot the vorticity field and full velocity vector in 3D.

    Parameters
    ----------
    circulation
        Circulation-strength parameter in the analytical vortex solution.
    a
        Constant radial-convergence/vertical-stretching rate.
    diffusivity
        Positive diffusion parameter in the Gaussian vortex solution.
    radius_max, height_max
        Displayed cylindrical-domain limits.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.  It is returned even when it is displayed or saved.
    """
    _validate_vorticity_inputs(
        circulation, a, diffusivity, radius_max, height_max
    )
    _require_integer_at_least("nr", nr, 2)
    _require_integer_at_least("ntheta", ntheta, 4)
    _require_integer_at_least("nz", nz, 2)
    _require_positive("scatter_size", scatter_size)
    if not 0.0 < scatter_alpha <= 1.0:
        raise ValueError("scatter_alpha must lie in (0,1].")
    _require_positive("quiver_length", quiver_length)
    if not 0.0 < quiver_alpha <= 1.0:
        raise ValueError("quiver_alpha must lie in (0,1].")
    _require_integer_at_least("dpi", dpi, 1)
    _validate_plot_style(
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        contour_label_fontsize=contour_label_fontsize,
        curve_linewidth=curve_linewidth,
        curve_color=curve_color,
    )

    radius_values = np.linspace(0.0, radius_max, nr)
    theta_values = np.linspace(
        0.0, 2.0 * np.pi, ntheta, endpoint=False
    )
    height_values = np.linspace(0.0, height_max, nz)
    radius, theta, height = np.meshgrid(
        radius_values,
        theta_values,
        height_values,
        indexing="ij",
    )

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    (
        vorticity,
        radial_velocity,
        vertical_velocity,
        azimuthal_velocity,
    ) = _vorticity_model_fields(
        radius, height, circulation, a, diffusivity
    )
    x_velocity = (
        radial_velocity * np.cos(theta)
        - azimuthal_velocity * np.sin(theta)
    )
    y_velocity = (
        radial_velocity * np.sin(theta)
        + azimuthal_velocity * np.cos(theta)
    )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        x.ravel(),
        y.ravel(),
        height.ravel(),
        c=vorticity.ravel(),
        cmap=cmap,
        alpha=scatter_alpha,
        s=scatter_size,
    )
    colorbar = fig.colorbar(scatter, ax=axis, shrink=0.65, pad=0.10)
    _set_colorbar_label(
        colorbar,
        r"Vertical vorticity $\zeta$",
        contour_label_fontsize,
    )

    quiver = axis.quiver(
        x.ravel(),
        y.ravel(),
        height.ravel(),
        x_velocity.ravel(),
        y_velocity.ravel(),
        vertical_velocity.ravel(),
        length=quiver_length,
        normalize=True,
        color=curve_color or "crimson",
        alpha=quiver_alpha,
    )
    if curve_linewidth is not None:
        quiver.set_linewidth(curve_linewidth)
    axis.set(
        title="Axisymmetric vortex: vorticity and velocity",
        xlabel=r"$x$",
        ylabel=r"$y$",
        zlabel=r"$z$",
        xlim=(-radius_max, radius_max),
        ylim=(-radius_max, radius_max),
        zlim=(0.0, height_max),
    )
    _apply_axis_text_style(
        axis,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
    )
    fig.tight_layout()
    return _finish_single_figure(
        fig,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )


def plot_vorticity_2D(
    *,
    circulation: float,
    a: float,
    diffusivity: float,
    radius_max: float,
    height_max: float,
    nr: int = 161,
    nz: int = 161,
    vorticity_levels: int = 40,
    wind_levels: int = 12,
    flow_style: str = "stream",
    vector_stride: int = 8,
    quiver_scale: float | None = None,
    cmap: str = "Reds",
    x_label_fontsize: float | None = None,
    y_label_fontsize: float | None = None,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    contour_label_fontsize: float | None = None,
    curve_linewidth: float | None = None,
    curve_color: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 100,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot vorticity, azimuthal wind, and meridional flow in ``(r,z)``.

    ``flow_style`` may be ``"stream"`` or ``"quiver"``.  The azimuthal wind
    is drawn as blue contours and the ``(u,w)`` meridional flow in black.
    """
    _validate_vorticity_inputs(
        circulation, a, diffusivity, radius_max, height_max
    )
    _require_integer_at_least("nr", nr, 3)
    _require_integer_at_least("nz", nz, 3)
    _require_integer_at_least("vorticity_levels", vorticity_levels, 2)
    _require_integer_at_least("wind_levels", wind_levels, 2)
    _require_integer_at_least("vector_stride", vector_stride, 1)
    _require_integer_at_least("dpi", dpi, 1)
    if flow_style not in {"stream", "quiver"}:
        raise ValueError("flow_style must be 'stream' or 'quiver'.")
    if quiver_scale is not None:
        _require_positive("quiver_scale", quiver_scale)
    _validate_plot_style(
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        contour_label_fontsize=contour_label_fontsize,
        curve_linewidth=curve_linewidth,
        curve_color=curve_color,
    )

    radius_values = np.linspace(0.0, radius_max, nr)
    height_values = np.linspace(0.0, height_max, nz)
    radius, height = np.meshgrid(radius_values, height_values)
    (
        vorticity,
        radial_velocity,
        vertical_velocity,
        azimuthal_velocity,
    ) = _vorticity_model_fields(
        radius, height, circulation, a, diffusivity
    )

    fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
    filled = axis.contourf(
        radius,
        height,
        vorticity,
        levels=vorticity_levels,
        cmap=cmap,
        extend="both",
    )
    colorbar = fig.colorbar(filled, ax=axis)
    _set_colorbar_label(
        colorbar,
        r"Vertical vorticity $\zeta (s^{-1})$",
        contour_label_fontsize,
    )

    wind_range = float(np.ptp(azimuthal_velocity))
    wind_scale = max(1.0, float(np.max(np.abs(azimuthal_velocity))))
    if wind_range > 100.0 * np.finfo(float).eps * wind_scale:
        contours = axis.contour(
            radius,
            height,
            azimuthal_velocity,
            levels=wind_levels,
            colors=curve_color or "blue",
            linewidths=curve_linewidth or 1.2,
            alpha=0.9,
        )
        axis.clabel(
            contours,
            inline=True,
            fmt="%.3g",
            fontsize=contour_label_fontsize or 8,
        )

    if flow_style == "stream":
        axis.streamplot(
            radius_values,
            height_values,
            radial_velocity,
            vertical_velocity,
            color=curve_color or "black",
            density=1.5,
            linewidth=curve_linewidth or 0.6,
            arrowsize=1.1,
        )
    else:
        stride = vector_stride
        flow_quiver = axis.quiver(
            radius[::stride, ::stride],
            height[::stride, ::stride],
            radial_velocity[::stride, ::stride],
            vertical_velocity[::stride, ::stride],
            color=curve_color or "black",
            alpha=0.7,
            scale=quiver_scale,
            pivot="mid",
            width=0.0025,
        )
        if curve_linewidth is not None:
            flow_quiver.set_linewidth(curve_linewidth)

    axis.set(
        title=(
            r"Vorticity $\zeta$ (shaded), azimuthal wind $v$ "
            r"(contours), and meridional flow $(u,w)$"
        ),
        xlabel=r"Radius $r$ (m)",
        ylabel=r"Height $z$ (m)",
        xlim=(0.0, radius_max),
        ylim=(0.0, height_max),
    )
    _apply_axis_text_style(
        axis,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
    )
    axis.grid(linestyle=":", alpha=0.2)
    fig.tight_layout()
    return _finish_single_figure(
        fig,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )


def _validate_wind_inputs(
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    Rf: float,
    H: float,
    times: Sequence[float],
    nr: int,
    n_solver: int,
    solver_radius: float | None,
    buffer_widths: float,
    rtol: float,
    atol: float,
    max_step: float,
    dpi: int,
) -> np.ndarray:
    for name, value in {"A0": A0, "b": b}.items():
        _require_finite(name, value)
    if a < 0.0 or not np.isfinite(a):
        raise ValueError("a must be finite and nonnegative.")
    if nu < 0.0 or not np.isfinite(nu):
        raise ValueError("nu must be finite and nonnegative.")
    if k < 0.0 or not np.isfinite(k):
        raise ValueError("k must be finite and nonnegative.")
    for name, value in {"R0": R0, "Rf": Rf, "H": H}.items():
        _require_positive(name, value)
    _require_integer_at_least("nr", nr, 3)
    _require_integer_at_least("n_solver", n_solver, 5)
    _require_positive("buffer_widths", buffer_widths)
    _require_positive("rtol", rtol)
    _require_positive("atol", atol)
    if max_step <= 0.0 or np.isnan(max_step):
        raise ValueError("max_step must be positive.")
    _require_integer_at_least("dpi", dpi, 1)
    if solver_radius is not None:
        _require_positive("solver_radius", solver_radius)
        if solver_radius <= Rf:
            raise ValueError("solver_radius must be larger than Rf.")

    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("times must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain only finite values.")
    if not np.isclose(time_values[0], 0.0):
        raise ValueError("times must start at t=0 for the stated initial value.")
    if np.any(time_values < 0.0) or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be nonnegative and strictly increasing.")
    return time_values


def _direct_vertical_diffusion_time(t: float, a: float) -> float:
    if a == 0.0:
        return t
    return -np.expm1(-2.0 * a * t) / (2.0 * a)


def _b_zero_amplitude(
    radius: np.ndarray,
    t: float,
    *,
    A0: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
) -> np.ndarray:
    """Closed-form viscous Fourier amplitude for the ``b=0`` solution."""
    if a == 0.0:
        exp_at = 1.0
        diffusion_time = t
    else:
        exp_at = np.exp(a * t)
        diffusion_time = np.expm1(a * t) / a

    denominator = R0**2 + 4.0 * nu * diffusion_time
    vertical_damping = np.exp(
        -nu * k**2 * _direct_vertical_diffusion_time(t, a)
    )
    return (
        A0
        * radius
        * exp_at
        * R0**4
        / denominator**2
        * np.exp(-exp_at * radius**2 / denominator)
        * vertical_damping
    )


def _inviscid_b_amplitude(
    radius: np.ndarray,
    t: float,
    *,
    A0: float,
    b: float,
    a: float,
    R0: float,
    k: float,
) -> np.ndarray:
    """Closed-form complex Fourier amplitude when ``nu=0``."""
    if a == 0.0:
        exp_at = 1.0
        phase_displacement = b * radius * t
    else:
        exp_at = np.exp(a * t)
        displacement_factor = (
            2.0
            * np.exp(-a * t)
            * np.expm1(1.5 * a * t)
            / (3.0 * a)
        )
        phase_displacement = b * radius * displacement_factor

    envelope = (
        A0
        * radius
        * exp_at
        * np.exp(-exp_at * radius**2 / R0**2)
    )
    return envelope * np.exp(-1j * k * phase_displacement)


def _maximum_radial_width(
    *, R0: float, nu: float, a: float, final_time: float
) -> float:
    if nu == 0.0:
        return R0
    if a == 0.0:
        return np.sqrt(R0**2 + 4.0 * nu * final_time)
    final_width_squared = (
        4.0 * nu / a
        + (R0**2 - 4.0 * nu / a) * np.exp(-a * final_time)
    )
    return max(R0, np.sqrt(final_width_squared))


def _choose_solver_grid(
    *,
    R0: float,
    Rf: float,
    nu: float,
    a: float,
    final_time: float,
    n_solver: int,
    solver_radius: float | None,
    buffer_widths: float,
) -> np.ndarray:
    width = _maximum_radial_width(
        R0=R0, nu=nu, a=a, final_time=final_time
    )
    if solver_radius is None:
        outer_radius = max(
            Rf + buffer_widths * width,
            (buffer_widths + 2.0) * width,
        )
    else:
        outer_radius = solver_radius

    radius = np.linspace(0.0, outer_radius, n_solver)
    if radius[1] - radius[0] > width / 20.0:
        print(
            "Warning: fewer than 20 radial solver points span the "
            "characteristic Gaussian width; increase n_solver.",
            file=sys.stderr,
        )
    return radius


def _build_radial_matrix(
    radius_solver: np.ndarray,
    *,
    a: float,
    nu: float,
):
    """Discretize ``nu*(drr+dr/r-1/r^2)+a*r*dr/2+a/2``."""
    dr = radius_solver[1] - radius_solver[0]
    radius = radius_solver[1:-1]
    matrix = lil_matrix(
        (radius.size, radius.size), dtype=np.complex128
    )
    alpha = 0.5 * a

    for index, radial_position in enumerate(radius):
        lower = (
            nu
            * (
                1.0 / dr**2
                - 1.0 / (2.0 * radial_position * dr)
            )
            - alpha * radial_position / (2.0 * dr)
        )
        diagonal = (
            nu * (-2.0 / dr**2 - 1.0 / radial_position**2)
            + alpha
        )
        upper = (
            nu
            * (
                1.0 / dr**2
                + 1.0 / (2.0 * radial_position * dr)
            )
            + alpha * radial_position / (2.0 * dr)
        )
        if index > 0:
            matrix[index, index - 1] = lower
        matrix[index, index] = diagonal
        if index < radius.size - 1:
            matrix[index, index + 1] = upper
    return matrix.tocsr(), radius


def _solve_b_amplitude(
    radius_solver: np.ndarray,
    times: np.ndarray,
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    if nu == 0.0:
        return np.stack(
            [
                _inviscid_b_amplitude(
                    radius_solver,
                    t,
                    A0=A0,
                    b=b,
                    a=a,
                    R0=R0,
                    k=k,
                )
                for t in times
            ]
        )

    radial_matrix, unknown_radius = _build_radial_matrix(
        radius_solver, a=a, nu=nu
    )
    initial = (
        A0
        * unknown_radius
        * np.exp(-unknown_radius**2 / R0**2)
    ).astype(np.complex128)

    amplitudes = np.zeros(
        (times.size, radius_solver.size), dtype=np.complex128
    )
    amplitudes[0, 1:-1] = initial
    if times.size == 1:
        return amplitudes

    def time_diagonal(t: float) -> np.ndarray:
        vertical_wavenumber = k * np.exp(-a * t)
        return (
            -nu * vertical_wavenumber**2
            - 1j * b * vertical_wavenumber * unknown_radius
        )

    def right_hand_side(t: float, amplitude: np.ndarray) -> np.ndarray:
        return radial_matrix @ amplitude + time_diagonal(t) * amplitude

    def jacobian(t: float, _amplitude: np.ndarray):
        return radial_matrix + diags(
            time_diagonal(t), offsets=0, format="csr"
        )

    solution = solve_ivp(
        right_hand_side,
        (float(times[0]), float(times[-1])),
        initial,
        method="BDF",
        t_eval=times,
        jac=jacobian,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(
            f"Radial azimuthal-wind solver failed: {solution.message}"
        )
    amplitudes[:, 1:-1] = solution.y.T
    return amplitudes


def _interpolate_complex(
    x_new: np.ndarray,
    x: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(x_new, x, values.real) + 1j * np.interp(
        x_new, x, values.imag
    )


def _wind_fields(
    frame: int,
    *,
    times: np.ndarray,
    radius_display: np.ndarray,
    height_display: np.ndarray,
    radius_solver: np.ndarray,
    b_amplitudes: np.ndarray,
    A0: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = times[frame]
    vertical_wavenumber = k * np.exp(-a * t)
    vertical_mode = np.exp(
        1j * vertical_wavenumber * height_display
    )[:, None]

    b_amplitude = _interpolate_complex(
        radius_display, radius_solver, b_amplitudes[frame]
    )
    b_field = np.real(vertical_mode * b_amplitude[None, :])

    zero_amplitude = _b_zero_amplitude(
        radius_display,
        t,
        A0=A0,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
    )
    zero_field = np.real(vertical_mode * zero_amplitude[None, :])
    return b_field, zero_field


def _prepare_wind_solution(
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    Rf: float,
    H: float,
    times: Sequence[float],
    nr: int,
    n_solver: int,
    solver_radius: float | None,
    buffer_widths: float,
    rtol: float,
    atol: float,
    max_step: float,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_values = _validate_wind_inputs(
        A0=A0,
        b=b,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
        Rf=Rf,
        H=H,
        times=times,
        nr=nr,
        n_solver=n_solver,
        solver_radius=solver_radius,
        buffer_widths=buffer_widths,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dpi=dpi,
    )
    radius_display = np.linspace(0.0, Rf, nr)
    radius_solver = _choose_solver_grid(
        R0=R0,
        Rf=Rf,
        nu=nu,
        a=a,
        final_time=float(time_values[-1]),
        n_solver=n_solver,
        solver_radius=solver_radius,
        buffer_widths=buffer_widths,
    )
    b_amplitudes = _solve_b_amplitude(
        radius_solver,
        time_values,
        A0=A0,
        b=b,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    return time_values, radius_display, radius_solver, b_amplitudes


def _add_meridional_quiver(
    axes,
    *,
    a: float,
    b: float,
    Rf: float,
    H: float,
    quiver_nr: int,
    quiver_nz: int,
    quiver_length_fraction: float,
    quiver_color: str,
    quiver_alpha: float,
    curve_linewidth: float | None,
) -> None:
    radius_values = np.linspace(0.0, Rf, quiver_nr + 2)[1:-1]
    height_values = np.linspace(0.0, H, quiver_nz + 2)[1:-1]
    radius, height = np.meshgrid(radius_values, height_values)

    radial_velocity = -0.5 * a * radius
    b_vertical_velocity = a * height + b * radius
    zero_vertical_velocity = a * height
    vector_fields = [
        (radial_velocity, b_vertical_velocity),
        (radial_velocity, zero_vertical_velocity),
        (
            np.zeros_like(radial_velocity),
            b_vertical_velocity - zero_vertical_velocity,
        ),
    ]
    speed_reference = max(
        float(np.max(np.hypot(u_component, w_component)))
        for u_component, w_component in vector_fields
    )
    if speed_reference == 0.0:
        return

    maximum_arrow_length = quiver_length_fraction * min(Rf, H)
    common_scale = speed_reference / maximum_arrow_length
    for axis, (u_component, w_component) in zip(axes, vector_fields):
        quiver = axis.quiver(
            radius,
            height,
            u_component,
            w_component,
            angles="xy",
            scale_units="xy",
            scale=common_scale,
            color=quiver_color,
            alpha=quiver_alpha,
            width=0.0025,
            headwidth=3.5,
            headlength=4.5,
            pivot="mid",
            zorder=3,
        )
        if curve_linewidth is not None:
            quiver.set_linewidth(curve_linewidth)


def _filename_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_azimuthal_wind_2D(
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    Rf: float,
    H: float,
    times: Sequence[float],
    nr: int = 241,
    nz: int = 241,
    vmax: float | None = None,
    difference_vmax: float | None = None,
    show_quiver: bool = True,
    quiver_nr: int = 14,
    quiver_nz: int = 14,
    quiver_length_fraction: float = 0.08,
    quiver_color: str = "black",
    quiver_alpha: float = 0.65,
    n_solver: int = 701,
    solver_radius: float | None = None,
    buffer_widths: float = 4.0,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step: float = np.inf,
    x_label_fontsize: float | None = None,
    y_label_fontsize: float | None = None,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    contour_label_fontsize: float | None = None,
    curve_linewidth: float | None = None,
    curve_color: str | None = None,
    figsize: tuple[float, float] = (17.0, 5.4),
    dpi: int = 140,
    output_dir: str | Path | None = None,
    file_prefix: str = "azimuthal_wind",
    show: bool = True,
):
    """Plot ``v_b``, ``v_0``, and ``v_b-v_0`` in ``(r,z)`` at each time.

    The initial condition is

    ``v(r,z,0) = A0*r*exp(-r**2/R0**2)*cos(k*z)``.

    For ``nu>0``, the ``b`` solution is obtained from the deterministic
    complex Fourier-amplitude equation using a radial finite-difference
    discretization and BDF time integration.  The ``b=0`` solution is the
    closed-form viscous solution.  For ``nu=0``, both branches are exact
    characteristic solutions.

    Returns a list of figures when ``output_dir`` is ``None``; otherwise,
    returns the paths of the saved PNG files.
    """
    _require_integer_at_least("nz", nz, 3)
    if vmax is not None:
        _require_positive("vmax", vmax)
    if difference_vmax is not None:
        _require_positive("difference_vmax", difference_vmax)
    _require_integer_at_least("quiver_nr", quiver_nr, 1)
    _require_integer_at_least("quiver_nz", quiver_nz, 1)
    _require_positive(
        "quiver_length_fraction", quiver_length_fraction
    )
    if not 0.0 < quiver_alpha <= 1.0:
        raise ValueError("quiver_alpha must lie in (0,1].")
    _validate_plot_style(
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        contour_label_fontsize=contour_label_fontsize,
        curve_linewidth=curve_linewidth,
        curve_color=curve_color,
    )

    (
        time_values,
        radius_display,
        radius_solver,
        b_amplitudes,
    ) = _prepare_wind_solution(
        A0=A0,
        b=b,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
        Rf=Rf,
        H=H,
        times=times,
        nr=nr,
        n_solver=n_solver,
        solver_radius=solver_radius,
        buffer_widths=buffer_widths,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dpi=dpi,
    )
    height_display = np.linspace(0.0, H, nz)

    fields = [
        _wind_fields(
            frame,
            times=time_values,
            radius_display=radius_display,
            height_display=height_display,
            radius_solver=radius_solver,
            b_amplitudes=b_amplitudes,
            A0=A0,
            a=a,
            nu=nu,
            R0=R0,
            k=k,
        )
        for frame in range(time_values.size)
    ]
    if vmax is None:
        velocity_limit = max(
            float(np.max(np.abs(field)))
            for pair in fields
            for field in pair
        )
        velocity_limit = velocity_limit or 1.0
    else:
        velocity_limit = vmax
    if difference_vmax is None:
        difference_limit = max(
            float(np.max(np.abs(b_field - zero_field)))
            for b_field, zero_field in fields
        )
        difference_limit = difference_limit or 0.05 * velocity_limit
    else:
        difference_limit = difference_vmax

    velocity_norm = Normalize(
        vmin=-velocity_limit, vmax=velocity_limit
    )
    difference_norm = Normalize(
        vmin=-difference_limit, vmax=difference_limit
    )
    save_directory = Path(output_dir) if output_dir is not None else None
    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)

    results = []
    for frame, t in enumerate(time_values):
        b_field, zero_field = fields[frame]
        difference = b_field - zero_field
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        b_mesh = axes[0].pcolormesh(
            radius_display,
            height_display,
            b_field,
            shading="auto",
            cmap="RdBu_r",
            norm=velocity_norm,
        )
        axes[1].pcolormesh(
            radius_display,
            height_display,
            zero_field,
            shading="auto",
            cmap="RdBu_r",
            norm=velocity_norm,
        )
        difference_mesh = axes[2].pcolormesh(
            radius_display,
            height_display,
            difference,
            shading="auto",
            cmap="PuOr_r",
            norm=difference_norm,
        )
        if show_quiver:
            _add_meridional_quiver(
                axes,
                a=a,
                b=b,
                Rf=Rf,
                H=H,
                quiver_nr=quiver_nr,
                quiver_nz=quiver_nz,
                quiver_length_fraction=quiver_length_fraction,
                quiver_color=curve_color or quiver_color,
                quiver_alpha=quiver_alpha,
                curve_linewidth=curve_linewidth,
            )

        axes[0].set_title(rf"$b={b:g}$")
        axes[1].set_title(r"$b=0$")
        axes[2].set_title(r"$\Delta v=v_b-v_0$")
        for axis in axes:
            axis.set(
                xlabel=r"Radius $r$",
                xlim=(0.0, Rf),
                ylim=(0.0, H),
            )
            _apply_axis_text_style(
                axis,
                x_label_fontsize=x_label_fontsize,
                y_label_fontsize=y_label_fontsize,
                x_tick_fontsize=x_tick_fontsize,
                y_tick_fontsize=y_tick_fontsize,
            )
        axes[0].set_ylabel(r"Height $z$")
        velocity_colorbar = fig.colorbar(
            b_mesh,
            ax=axes[:2],
            location="right",
            shrink=0.94,
            pad=0.02,
        )
        _set_colorbar_label(
            velocity_colorbar,
            r"$v_b$ and $v_0$",
            contour_label_fontsize,
        )
        difference_colorbar = fig.colorbar(
            difference_mesh,
            ax=axes[2],
            location="right",
            shrink=0.94,
            pad=0.02,
        )
        _set_colorbar_label(
            difference_colorbar,
            r"$\Delta v$",
            contour_label_fontsize,
        )
        fig.suptitle(rf"Azimuthal wind at $t={t:g}$")

        if save_directory is None:
            if show:
                plt.show()
            results.append(fig)
        else:
            path = save_directory / (
                f"{file_prefix}_2D_t{_filename_number(t)}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            results.append(path)
    return results


def _radial_wind_profiles(
    *,
    time_values: np.ndarray,
    radius_display: np.ndarray,
    radius_solver: np.ndarray,
    b_amplitudes: np.ndarray,
    profile_z: float,
    A0: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_profiles = np.empty(
        (time_values.size, radius_display.size), dtype=float
    )
    zero_profiles = np.empty_like(b_profiles)
    for frame, t in enumerate(time_values):
        vertical_wavenumber = k * np.exp(-a * t)
        vertical_phase = np.exp(1j * vertical_wavenumber * profile_z)
        b_amplitude = _interpolate_complex(
            radius_display, radius_solver, b_amplitudes[frame]
        )
        b_profiles[frame] = np.real(vertical_phase * b_amplitude)
        zero_amplitude = _b_zero_amplitude(
            radius_display,
            t,
            A0=A0,
            a=a,
            nu=nu,
            R0=R0,
            k=k,
        )
        zero_profiles[frame] = np.real(
            vertical_phase * zero_amplitude
        )
    return b_profiles, zero_profiles, b_profiles - zero_profiles


def _label_profile_curve(
    axis,
    radius: np.ndarray,
    profile: np.ndarray,
    *,
    label: str,
    color,
    curve_index: int,
    curve_count: int,
    fontsize: float,
) -> None:
    amplitude = float(np.max(np.abs(profile)))
    if amplitude > 0.0:
        candidates = np.flatnonzero(np.abs(profile) >= 0.08 * amplitude)
    else:
        candidates = np.arange(radius.size)
    fraction = (
        0.32 + 0.63 * curve_index / (curve_count - 1)
        if curve_count > 1
        else 0.65
    )
    candidate_index = min(
        int(round(fraction * (candidates.size - 1))),
        candidates.size - 1,
    )
    point = int(candidates[candidate_index])
    axis.annotate(
        label,
        xy=(radius[point], profile[point]),
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.68,
        },
        zorder=4,
    )


def plot_azimuthal_wind_profile(
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    Rf: float,
    H: float,
    profile_z: float,
    times: Sequence[float],
    nr: int = 241,
    n_solver: int = 701,
    solver_radius: float | None = None,
    buffer_widths: float = 4.0,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step: float = np.inf,
    x_label_fontsize: float | None = None,
    y_label_fontsize: float | None = None,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    contour_label_fontsize: float | None = None,
    curve_linewidth: float | None = None,
    curve_color: str | None = None,
    figsize: tuple[float, float] = (17.0, 5.3),
    dpi: int = 140,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot all radial profiles of ``v_b``, ``v_0``, and their difference.

    Every requested time is plotted in each of the three panels and labeled
    directly on its curve.  ``profile_z`` must lie in ``[0,H]``.
    """
    _require_finite("profile_z", profile_z)
    if not 0.0 <= profile_z <= H:
        raise ValueError("profile_z must lie in the interval [0,H].")
    _validate_plot_style(
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        contour_label_fontsize=contour_label_fontsize,
        curve_linewidth=curve_linewidth,
        curve_color=curve_color,
    )

    (
        time_values,
        radius_display,
        radius_solver,
        b_amplitudes,
    ) = _prepare_wind_solution(
        A0=A0,
        b=b,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
        Rf=Rf,
        H=H,
        times=times,
        nr=nr,
        n_solver=n_solver,
        solver_radius=solver_radius,
        buffer_widths=buffer_widths,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dpi=dpi,
    )
    b_profiles, zero_profiles, difference = _radial_wind_profiles(
        time_values=time_values,
        radius_display=radius_display,
        radius_solver=radius_solver,
        b_amplitudes=b_amplitudes,
        profile_z=profile_z,
        A0=A0,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    if curve_color is None:
        colors = plt.get_cmap("viridis")(
            np.linspace(0.05, 0.95, time_values.size)
        )
    else:
        colors = [curve_color] * time_values.size
    profile_linewidth = curve_linewidth or 1.8
    profile_label_fontsize = contour_label_fontsize or 8
    panel_data = (b_profiles, zero_profiles, difference)
    panel_titles = (
        rf"$b={b:g}$",
        r"$b=0$",
        r"Difference: $v_b-v_0$",
    )
    y_labels = (r"$v_b$", r"$v_0$", r"$\Delta v$")

    main_limit = max(
        float(np.max(np.abs(b_profiles))),
        float(np.max(np.abs(zero_profiles))),
    )
    main_limit = main_limit or 1.0
    difference_limit = float(np.max(np.abs(difference)))
    difference_limit = difference_limit or 0.05 * main_limit

    for axis, profiles, title, y_label in zip(
        axes, panel_data, panel_titles, y_labels
    ):
        for frame, (t, color) in enumerate(zip(time_values, colors)):
            axis.plot(
                radius_display,
                profiles[frame],
                color=color,
                linewidth=profile_linewidth,
                zorder=2,
            )
            _label_profile_curve(
                axis,
                radius_display,
                profiles[frame],
                label=rf"$t={t:g}$",
                color=color,
                curve_index=frame,
                curve_count=time_values.size,
                fontsize=profile_label_fontsize,
            )
        axis.axhline(0.0, color="0.45", linewidth=0.8, zorder=1)
        axis.set(
            xlabel=r"Radius $r$",
            ylabel=y_label,
            title=title,
            xlim=(0.0, Rf),
        )
        _apply_axis_text_style(
            axis,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
        )
        axis.grid(alpha=0.22)

    axes[0].set_ylim(0, 1.08 * main_limit)
    axes[1].set_ylim(0, 1.08 * main_limit)
    axes[2].set_ylim(
        -1.08 * difference_limit, 1.08 * difference_limit
    )
    fig.suptitle(
        rf"Radial azimuthal-wind profiles at $z={profile_z:g}$"
    )
    return _finish_single_figure(
        fig,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )


def plot_azimuthal_wind_vertical_profile(
    *,
    A0: float,
    b: float,
    a: float,
    nu: float,
    R0: float,
    k: float,
    Rf: float,
    H: float,
    profile_r: float,
    times: Sequence[float],
    nz: int = 241,
    n_solver: int = 701,
    solver_radius: float | None = None,
    buffer_widths: float = 4.0,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step: float = np.inf,
    x_label_fontsize: float | None = None,
    y_label_fontsize: float | None = None,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    contour_label_fontsize: float | None = None,
    curve_linewidth: float | None = None,
    curve_color: str | None = None,
    figsize: tuple[float, float] = (17.0, 5.3),
    dpi: int = 140,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot ``v`` horizontally against ``z`` at one radius and many times.

    The three panels contain the ``b != 0`` solution, the ``b = 0``
    solution, and ``v_b-v_0``.  ``times`` lists the profiles to display
    and need not contain zero: the numerical solution is automatically
    integrated from the initial condition at ``t=0``.
    """
    _require_finite("profile_r", profile_r)
    if not 0.0 <= profile_r <= Rf:
        raise ValueError("profile_r must lie in the interval [0,Rf].")
    _require_integer_at_least("nz", nz, 3)
    _validate_plot_style(
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        x_tick_fontsize=x_tick_fontsize,
        y_tick_fontsize=y_tick_fontsize,
        contour_label_fontsize=contour_label_fontsize,
        curve_linewidth=curve_linewidth,
        curve_color=curve_color,
    )

    display_times = np.asarray(times, dtype=float)
    if display_times.ndim != 1 or display_times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(display_times)):
        raise ValueError("times must contain only finite values.")
    if np.any(display_times < 0.0):
        raise ValueError("times must be nonnegative.")
    if np.any(np.diff(display_times) <= 0.0):
        raise ValueError("times must be strictly increasing.")

    if np.isclose(display_times[0], 0.0):
        solver_times = display_times.copy()
        solver_times[0] = 0.0
        first_display_frame = 0
    else:
        solver_times = np.insert(display_times, 0, 0.0)
        first_display_frame = 1

    (
        time_values,
        _radius_display,
        radius_solver,
        all_b_amplitudes,
    ) = _prepare_wind_solution(
        A0=A0,
        b=b,
        a=a,
        nu=nu,
        R0=R0,
        k=k,
        Rf=Rf,
        H=H,
        times=solver_times,
        nr=3,
        n_solver=n_solver,
        solver_radius=solver_radius,
        buffer_widths=buffer_widths,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dpi=dpi,
    )
    solved_display_times = time_values[first_display_frame:]
    b_amplitudes = all_b_amplitudes[first_display_frame:]
    if not np.allclose(solved_display_times, display_times):
        raise RuntimeError("The solver did not return the requested times.")

    height = np.linspace(0.0, H, nz)
    b_profiles = np.empty((display_times.size, nz), dtype=float)
    zero_profiles = np.empty_like(b_profiles)
    for frame, t in enumerate(display_times):
        vertical_wavenumber = k * np.exp(-a * t)
        vertical_mode = np.exp(1j * vertical_wavenumber * height)

        b_amplitude = _interpolate_complex(
            np.array([profile_r]),
            radius_solver,
            b_amplitudes[frame],
        )[0]
        b_profiles[frame] = np.real(vertical_mode * b_amplitude)

        zero_amplitude = _b_zero_amplitude(
            np.array([profile_r]),
            t,
            A0=A0,
            a=a,
            nu=nu,
            R0=R0,
            k=k,
        )[0]
        zero_profiles[frame] = np.real(
            vertical_mode * zero_amplitude
        )

    difference = b_profiles - zero_profiles
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )
    if curve_color is None:
        colors = plt.get_cmap("viridis")(
            np.linspace(0.05, 0.95, display_times.size)
        )
    else:
        colors = [curve_color] * display_times.size
    profile_linewidth = curve_linewidth or 1.8
    profile_label_fontsize = contour_label_fontsize or 8
    panel_data = (b_profiles, zero_profiles, difference)
    panel_titles = (
        rf"$b={b:g}$",
        r"$b=0$",
        r"Difference: $v_b-v_0$",
    )
    x_labels = (r"$v_b$", r"$v_0$", r"$\Delta v$")

    main_limit = max(
        float(np.max(np.abs(b_profiles))),
        float(np.max(np.abs(zero_profiles))),
    )
    main_limit = main_limit or 1.0
    difference_limit = float(np.max(np.abs(difference)))
    difference_limit = difference_limit or 0.05 * main_limit

    for axis, profiles, title, x_label in zip(
        axes, panel_data, panel_titles, x_labels
    ):
        for frame, (t, color) in enumerate(
            zip(display_times, colors)
        ):
            axis.plot(
                profiles[frame],
                height,
                color=color,
                linewidth=profile_linewidth,
                zorder=2,
            )
            _label_profile_curve(
                axis,
                profiles[frame],
                height,
                label=rf"$t={t:g}$",
                color=color,
                curve_index=frame,
                curve_count=display_times.size,
                fontsize=profile_label_fontsize,
            )
        axis.axvline(0.0, color="0.45", linewidth=0.8, zorder=1)
        axis.set(
            xlabel=x_label,
            ylabel=r"Height $z$",
            title=title,
            ylim=(0.0, H),
        )
        _apply_axis_text_style(
            axis,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            x_tick_fontsize=x_tick_fontsize,
            y_tick_fontsize=y_tick_fontsize,
        )
        axis.grid(alpha=0.22)

    #axes[0].set_xlim(-1.08 * main_limit, 1.08 * main_limit)
    #axes[1].set_xlim(-1.08 * main_limit, 1.08 * main_limit)
    axes[0].set_xlim(0, 1.08 * main_limit)
    axes[1].set_xlim(0, 1.08 * main_limit)
    axes[2].set_xlim(
        -1.08 * difference_limit, 1.08 * difference_limit
    )
    fig.suptitle(
        rf"Vertical azimuthal-wind profiles at $r={profile_r:g}$"
    )
    return _finish_single_figure(
        fig,
        output_path=output_path,
        dpi=dpi,
        show=show,
    )
