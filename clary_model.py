import math
import numpy as np


class Sim:
    def __init__(self):
        self.pi_over_30 = math.pi / 30

    def g_vector_x(self, time_in_seconds, inner_rad_sec, outer_rad_sec):
        return math.sin(outer_rad_sec * time_in_seconds) * math.cos(inner_rad_sec * time_in_seconds)

    def g_vector_y(self, time_in_seconds, outer_rad_sec):
        return math.cos(outer_rad_sec * time_in_seconds)

    def g_vector_z(self, time_in_seconds, inner_rad_sec, outer_rad_sec):
        return math.sin(outer_rad_sec * time_in_seconds) * math.sin(inner_rad_sec * time_in_seconds)

    def rpm_to_rad_sec(self, rpm):
        return rpm * self.pi_over_30

    def g_vector_data(self, start_time_in_seconds, end_time_in_seconds, inner_rpm, outer_rpm):
        inner_rad_sec = self.rpm_to_rad_sec(inner_rpm)
        outer_rad_sec = self.rpm_to_rad_sec(outer_rpm)
        time_array = list(range(start_time_in_seconds, end_time_in_seconds + 1))
        x_array = [self.g_vector_x(t, inner_rad_sec, outer_rad_sec) for t in time_array]
        y_array = [self.g_vector_y(t, outer_rad_sec) for t in time_array]
        z_array = [self.g_vector_z(t, inner_rad_sec, outer_rad_sec) for t in time_array]
        return time_array, x_array, y_array, z_array


class DataProcessor:
    def __init__(self, inner_v, outer_v, max_seg, start_analysis, end_analysis):
        self.inner_v = inner_v
        self.outer_v = outer_v
        self.min_seg = 0
        self.max_seg = max_seg
        self.end_time = int(max_seg * 3600)
        self.start_analysis = start_analysis
        self.end_analysis = end_analysis
        self.start_seg = int(start_analysis * 3600) if start_analysis is not None else None
        self.end_seg = int(end_analysis * 3600) if end_analysis is not None else None
        self.time, self.x, self.y, self.z = self._get_sim_accel_data()

    def _get_sim_accel_data(self):
        vector_sim = Sim()
        return vector_sim.g_vector_data(0, self.end_time, self.inner_v, self.outer_v)

    def _get_time_avg(self):
        x_time_avg = np.cumsum(self.x) / np.arange(1, len(self.x) + 1)
        y_time_avg = np.cumsum(self.y) / np.arange(1, len(self.y) + 1)
        z_time_avg = np.cumsum(self.z) / np.arange(1, len(self.z) + 1)
        return x_time_avg, y_time_avg, z_time_avg

    def _get_magnitude(self, x_time_avg, y_time_avg, z_time_avg):
        return np.sqrt(x_time_avg**2 + y_time_avg**2 + z_time_avg**2)

    def _get_mag_seg(self, mag_list):
        mag_seg_list = mag_list[self.min_seg:self.end_time]
        avg_mag_full = np.mean(mag_seg_list)
        avg_mag_analysis = np.mean(mag_list[self.start_seg:self.end_seg]) if self.start_seg is not None and self.end_seg is not None else None
        return avg_mag_full, avg_mag_analysis

    def get_distribution(self):
        path = PathVisualization(self.inner_v, self.x, self.y, self.z)
        return path.get_distribution()


class PathVisualization:
    def __init__(self, id_, x, y, z):
        self.id_ = id_
        self.x = x
        self.y = y
        self.z = z
        self.path_coords = list(zip(x, y, z))
        self.num_points = 1000

    def _create_sphere(self):
        golden_r = (np.sqrt(5.0) + 1.0) / 2.0
        golden_a = (2.0 - golden_r) * (2.0 * np.pi)
        xs, ys, zs = [], [], []

        for i in range(self.num_points):
            y = 1 - (i / (self.num_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_a * i
            xs.append(np.cos(theta) * radius)
            ys.append(y)
            zs.append(np.sin(theta) * radius)

        return list(zip(xs, ys, zs))

    def _split_sphere(self, sphere_coords):
        octants = {'posI': [], 'posII': [], 'posIII': [], 'posIV': [], 'negI': [], 'negII': [], 'negIII': [], 'negIV': []}
        for x, y, z in sphere_coords:
            if z > 0:
                if y > 0:
                    octants['posI' if x > 0 else 'posII'].append((x, y, z))
                else:
                    octants['posIV' if x > 0 else 'posIII'].append((x, y, z))
            else:
                if y > 0:
                    octants['negI' if x > 0 else 'negII'].append((x, y, z))
                else:
                    octants['negIV' if x > 0 else 'negIII'].append((x, y, z))
        return octants

    def _get_path_octant(self, path_row):
        x, y, z = path_row
        if z > 0:
            if y > 0:
                return 'posI' if x > 0 else 'posII'
            return 'posIV' if x > 0 else 'posIII'
        if y > 0:
            return 'negI' if x > 0 else 'negII'
        return 'negIV' if x > 0 else 'negIII'

    def _get_distance_between(self, path_coords, sphere_coords):
        px, py, pz = path_coords
        sx, sy, sz = sphere_coords
        return np.sqrt((px - sx)**2 + (py - sy)**2 + (pz - sz)**2)

    def _get_distribution_num(self, sphere_coords):
        octants = self._split_sphere(sphere_coords)
        path_map = {}
        repeat_time = -1

        for path_row in self.path_coords:
            repeat_time += 1
            path_octant = self._get_path_octant(path_row)
            sphere_coords_split = octants[path_octant]
            dist_dict = {sphere_row: self._get_distance_between(path_row, sphere_row) for sphere_row in sphere_coords_split}
            ranked_dist = sorted(dist_dict.items(), key=lambda x: x[1])[:3]
            segment_vertices = tuple(r[0] for r in ranked_dist)
            path_map[segment_vertices] = path_map.get(segment_vertices, []) + [repeat_time]

        return len(path_map)

    def get_distribution(self):
        sphere_coords = self._create_sphere()
        return self._get_distribution_num(sphere_coords)

    def format_time(self, time):
        return [t / 3600 for t in time]