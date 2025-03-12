import numpy as np

class KimModel:
    def __init__(self, inner_rpm, outer_rpm, delta_x, delta_y, delta_z, duration_hours):
        self.inner_rpm = inner_rpm
        self.outer_rpm = outer_rpm
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        self.duration_hours = duration_hours
        self.pi_over_30 = np.pi / 30
        self.g = np.array([0, -9.8, 0])

    def rpm_to_rad_sec(self, rpm):
        return rpm * self.pi_over_30

    def calculate_acceleration(self):
        inner_rad_sec = self.rpm_to_rad_sec(self.inner_rpm)
        outer_rad_sec = self.rpm_to_rad_sec(self.outer_rpm)
        time_array = np.linspace(0, self.duration_hours * 3600, num=int(self.duration_hours * 3600))

        theta_1 = outer_rad_sec * time_array
        theta_2 = inner_rad_sec * time_array

        w1 = np.array([outer_rad_sec, 0, 0])
        w2 = np.array([np.zeros_like(theta_1), inner_rad_sec * np.cos(theta_1), inner_rad_sec * np.sin(theta_1)])
        w = w1[:, np.newaxis] + w2

        w_dot = np.array([np.zeros_like(theta_1),
                         -outer_rad_sec * inner_rad_sec * np.sin(theta_1),
                         outer_rad_sec * inner_rad_sec * np.cos(theta_1)])

        r = np.array([
            self.delta_x * np.cos(theta_2) + self.delta_z * np.sin(theta_2),
            self.delta_y * np.cos(theta_1) + self.delta_x * np.sin(theta_1) * np.sin(theta_2) - self.delta_z * np.sin(theta_1) * np.cos(theta_2),
            self.delta_y * np.sin(theta_1) - self.delta_x * np.cos(theta_1) * np.sin(theta_2) + self.delta_z * np.cos(theta_1) * np.cos(theta_2)
        ])

        w_cross_r = np.cross(w.T, r.T).T
        w_cross_w_cross_r = np.cross(w.T, w_cross_r.T).T
        w_dot_cross_r = np.cross(w_dot.T, r.T).T
        a = -w_dot_cross_r + w_cross_w_cross_r

        R_y_T = np.array([
            [np.cos(theta_1), np.zeros_like(theta_1), np.sin(theta_1)],
            [np.zeros_like(theta_1), np.ones_like(theta_1), np.zeros_like(theta_1)],
            [-np.sin(theta_1), np.zeros_like(theta_1), np.cos(theta_1)]
        ])

        R_x_T = np.array([
            [np.ones_like(theta_2), np.zeros_like(theta_2), np.zeros_like(theta_2)],
            [np.zeros_like(theta_2), np.cos(theta_2), -np.sin(theta_2)],
            [np.zeros_like(theta_2), np.sin(theta_2), np.cos(theta_2)]
        ])

        a_prime = np.einsum('ijk,jk->ik', R_y_T, np.einsum('ijk,jk->ik', R_x_T, a))
        g_prime = np.einsum('ijk,jk->ik', R_y_T, np.einsum('ijk,jk->ik', R_x_T, self.g[:, np.newaxis]))
        a_tot_prime = a_prime + g_prime

        return time_array, a_tot_prime[0], a_tot_prime[1], a_tot_prime[2]