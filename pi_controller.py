# ==========================================
# TwinEV v2 — PI Controller
# ==========================================
# Sits between RL agent and actual power
# delivery. Smoothly tracks RL setpoint
# without grid spikes or oscillations.
# ==========================================

import numpy as np


class PIController:
    """
    Proportional-Integral controller.

    The RL agent outputs a TARGET power setpoint (kW).
    The PI controller smoothly drives actual output
    toward that setpoint, respecting hard limits.

    Parameters
    ----------
    Kp          : Proportional gain
    Ki          : Integral gain
    dt          : Time step in hours (default 1.0 for hourly)
    output_min  : Minimum power output (kW)
    output_max  : Maximum power output (kW)
    windup_limit: Anti-windup clamp on integral term (kW)
    """

    def __init__(
        self,
        Kp           = 0.6,
        Ki           = 0.15,
        dt           = 1.0,
        output_min   = 0.0,
        output_max   = 150.0,
        windup_limit = 50.0
    ):
        self.Kp           = Kp
        self.Ki           = Ki
        self.dt           = dt
        self.output_min   = output_min
        self.output_max   = output_max
        self.windup_limit = windup_limit

        # Internal state
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_output = 0.0

    # ------------------------------------------
    # STEP
    # ------------------------------------------

    def step(self, setpoint: float, actual: float) -> float:
        """
        Compute one control step.

        Parameters
        ----------
        setpoint : RL-requested power (kW)
        actual   : Current measured power (kW)

        Returns
        -------
        output   : Adjusted power command (kW), clamped to limits
        """

        error = setpoint - actual

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup clamping
        self._integral += error * self.dt
        self._integral  = np.clip(
            self._integral,
            -self.windup_limit,
             self.windup_limit
        )
        I = self.Ki * self._integral

        # Raw output
        raw_output = actual + P + I

        # Hard clamp to physical limits
        output = float(np.clip(raw_output, self.output_min, self.output_max))

        # Tracking error (for reward function)
        self.tracking_error = abs(setpoint - output)

        self._prev_error  = error
        self._prev_output = output

        return output

    # ------------------------------------------
    # RESET (called at episode start)
    # ------------------------------------------

    def reset(self):
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_output = 0.0
        self.tracking_error = 0.0

    # ------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------

    def get_state(self) -> dict:
        return {
            'integral'       : round(self._integral, 4),
            'prev_error'     : round(self._prev_error, 4),
            'prev_output'    : round(self._prev_output, 4),
            'tracking_error' : round(self.tracking_error, 4),
        }


# ==========================================
# QUICK TEST
# ==========================================

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    pi = PIController(Kp=0.6, Ki=0.15, output_max=150.0)

    # Simulate: RL says ramp from 50 to 120 kW over 24 steps
    setpoints = [50] * 8 + [120] * 8 + [80] * 8
    actual    = 50.0
    outputs   = []
    sps       = []

    for sp in setpoints:
        out    = pi.step(setpoint=sp, actual=actual)
        actual = out + np.random.normal(0, 2)   # simulate noise
        outputs.append(out)
        sps.append(sp)

    plt.figure(figsize=(10, 4))
    plt.plot(sps,     label='RL Setpoint',      linewidth=2, linestyle='--')
    plt.plot(outputs, label='PI Output (actual)', linewidth=2)
    plt.title('PI Controller Tracking Test')
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_pi_test.png', dpi=150)
    plt.show()

    print("PI Controller test complete.")
    print(pi.get_state())
