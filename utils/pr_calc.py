import numpy as np
from scipy import interpolate
from scipy.integrate import quad

def nonuniform_convolution(t_f, t_g, f, g, t_target, integral_way="gauss"):
    """
    Perform convolution of two functions f and g with non-uniformly spaced time points t.
    """
    # Create an interpolation function for both f and g
    f_interp = interpolate.CubicSpline(t_f, f, bc_type='natural', extrapolate=True)
    g_interp = interpolate.CubicSpline(t_g, g, bc_type='natural', extrapolate=True)
    conv_result = np.zeros(len(t_target))

    f_min = min(t_f)
    g_min = min(t_g)

    for i, t_i in enumerate(t_target):
        integral = 0
        integrand = lambda tau: f_interp(tau) * g_interp(t_i - tau) if f_min <= tau and g_min <= t_i - tau else 0
        for j in range(1, len(t_target)):
            # if t_i < t_target[j]:
            #     break
            if integral_way == "gauss":
                integral += quad(integrand, t_target[j-1], t_target[j], limit=100)[0]
            elif integral_way == "trapz":
                integral += (integrand(t_target[j]) + integrand(t_target[j-1])) * (t_target[j] - t_target[j-1]) / 2
            else:
                raise ValueError("integral_way should be gauss or trapz")
        conv_result[i] = integral
    return conv_result
