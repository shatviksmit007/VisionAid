import math
import numpy as np


class CameraBearingGeometry:
    """
    Converts image coordinates into approximate horizontal bearing.

    The model assumes a pinhole camera and requires an approximate
    horizontal field of view (HFOV).

    Angles are measured relative to the optical axis:
        negative = left
        positive = right
        zero     = center
    """

    def __init__(
        self,
        hfov_deg=70.0,
        collision_half_angle_deg=15.0,
    ):
        self.hfov_deg = float(hfov_deg)
        self.collision_half_angle_deg = float(
            collision_half_angle_deg
        )

        self.hfov_rad = math.radians(self.hfov_deg)

        # Principal point is initially assumed to be image center.
        self.cx_norm = 0.5

        # Focal length in normalized image coordinates.
        self.fx_norm = 0.5 / math.tan(self.hfov_rad / 2.0)

    def set_principal_point(self, cx_norm):
        self.cx_norm = float(np.clip(cx_norm, 0.0, 1.0))

    def bearing_deg(self, x_norm):
        """
        Convert normalized image x coordinate to horizontal bearing.
        """
        x_norm = float(np.clip(x_norm, 0.0, 1.0))

        x_camera = x_norm - self.cx_norm

        angle = math.atan2(
            x_camera,
            self.fx_norm,
        )

        return math.degrees(angle)

    def in_collision_corridor(self, x_norm):
        """
        Return whether an image x coordinate lies inside
        the horizontal collision corridor.
        """
        bearing = self.bearing_deg(x_norm)

        return (
            abs(bearing)
            <= self.collision_half_angle_deg
        )

    def corridor_x_bounds(self):
        """
        Return normalized image x coordinates corresponding
        to the left/right collision-angle boundaries.
        """
        theta = math.radians(
            self.collision_half_angle_deg
        )

        left = (
            self.cx_norm
            - self.fx_norm * math.tan(theta)
        )

        right = (
            self.cx_norm
            + self.fx_norm * math.tan(theta)
        )

        return (
            float(np.clip(left, 0.0, 1.0)),
            float(np.clip(right, 0.0, 1.0)),
        )

    def get_info(self):
        left, right = self.corridor_x_bounds()

        return {
            "hfov_deg": self.hfov_deg,
            "collision_half_angle_deg":
                self.collision_half_angle_deg,
            "cx_norm": self.cx_norm,
            "fx_norm": self.fx_norm,
            "left_x": left,
            "right_x": right,
        }
