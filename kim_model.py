import numpy as np

class KimModel:
    def __init__(self, inner_rpm, outer_rpm, delta_x, delta_y, delta_z, duration_hours):
        """
        Initialize the 3D clinostat model.
        
        Parameters:
        - inner_rpm: Inner frame rotation speed (RPM) - corresponds to θ₁̇
        - outer_rpm: Outer frame rotation speed (RPM) - corresponds to θ₂̇
        - delta_x, delta_y, delta_z: Position deviations from clinostat center (meters)
        - duration_hours: Simulation duration (hours)
        """
        self.inner_rpm = inner_rpm  # Inner frame angular velocity (θ₁̇)
        self.outer_rpm = outer_rpm  # Outer frame angular velocity (θ₂̇)
        self.delta_x = delta_x      # Δx
        self.delta_y = delta_y      # Δy
        self.delta_z = delta_z      # Δz
        self.duration_hours = duration_hours
        self.pi_over_30 = np.pi / 30  # Conversion factor from RPM to rad/s
        self.g = np.array([[0], [-9.8], [0]])  # Shape: (3, 1)

    def rpm_to_rad_sec(self, rpm):
        """Convert RPM to radians per second."""
        return rpm * self.pi_over_30

    def calculate_acceleration(self):
        """
        Calculate total acceleration in Local 2 frame over time.
        
        Returns:
        - time_array: Time points (seconds)
        - ax, ay, az: Acceleration components in Local 2 frame (m/s²)
        """
        # Convert RPM to rad/s
        outer_rad_sec = self.rpm_to_rad_sec(self.outer_rpm)  # θ₂̇ (outer frame)
        inner_rad_sec = self.rpm_to_rad_sec(self.inner_rpm)  # θ₁̇ (inner frame)

        # Time array in seconds
        time_array = np.linspace(0, self.duration_hours * 3600, num=int(self.duration_hours * 3600))

        # Angles as function of time
        theta_2 = outer_rad_sec * time_array  # θ₂ (outer frame angle)
        theta_1 = inner_rad_sec * time_array  # θ₁ (inner frame angle)

        # Total angular velocity w = w₁ + w₂
        w = np.array([
            inner_rad_sec * np.ones_like(time_array),          # w_x = θ₁̇
            outer_rad_sec * np.cos(theta_1),                   # w_y = θ₂̇ cos(θ₁)
            outer_rad_sec * np.sin(theta_1)                    # w_z = θ₂̇ sin(θ₁)
        ])  # Shape: (3, len(time_array))

        # Angular acceleration (derivative of w)
        w_dot = np.array([
            np.zeros_like(time_array),                         # ẇ_x = 0 (constant θ₁̇)
            -inner_rad_sec * outer_rad_sec * np.sin(theta_1),  # ẇ_y = -θ₁̇ θ₂̇ sin(θ₁)
            inner_rad_sec * outer_rad_sec * np.cos(theta_1)    # ẇ_z = θ₁̇ θ₂̇ cos(θ₁)
        ])  # Shape: (3, len(time_array))

        # Position in global frame
        r = np.array([
            self.delta_x * np.cos(theta_2) + self.delta_z * np.sin(theta_2),
            self.delta_y * np.cos(theta_1) + self.delta_x * np.sin(theta_1) * np.sin(theta_2) - self.delta_z * np.sin(theta_1) * np.cos(theta_2),
            self.delta_y * np.sin(theta_1) - self.delta_x * np.cos(theta_1) * np.sin(theta_2) + self.delta_z * np.cos(theta_1) * np.cos(theta_2)
        ])

        # Acceleration components
        w_cross_r = np.cross(w.T, r.T).T
        w_cross_w_cross_r = np.cross(w.T, w_cross_r.T).T
        w_dot_cross_r = np.cross(w_dot.T, r.T).T
        a = -(w_dot_cross_r + w_cross_w_cross_r)  # a(t) = -{ẇ × r + w × (w × r)}

        # Rotation matrices (transposed)
        R_y_T = np.array([
            [np.cos(theta_1), np.zeros_like(theta_1), -np.sin(theta_1)],
            [np.zeros_like(theta_1), np.ones_like(theta_1), np.zeros_like(theta_1)],
            [np.sin(theta_1), np.zeros_like(theta_1), np.cos(theta_1)]
        ])  # R_y^T(θ₁)

        R_x_T = np.array([
            [np.ones_like(theta_2), np.zeros_like(theta_2), np.zeros_like(theta_2)],
            [np.zeros_like(theta_2), np.cos(theta_2), np.sin(theta_2)],
            [np.zeros_like(theta_2), -np.sin(theta_2), np.cos(theta_2)]
        ])  # R_x^T(θ₂)

        # Transform accelerations to Local 2 frame
        a_prime = np.einsum('ijk,jk->ik', R_y_T, np.einsum('ijk,jk->ik', R_x_T, a))  # a(t)''
        g_prime = np.einsum('ijk,jk->ik', R_y_T, np.einsum('ijk,jk->ik', R_x_T, self.g))  # g(t)''

        # Total acceleration in Local 2 frame
        a_tot_prime = a_prime + g_prime  # a(t)_{tot}''

        return time_array, a_tot_prime[0], a_tot_prime[1], a_tot_prime[2]
