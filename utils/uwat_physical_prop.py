from utils.modules import PhysicalProp
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
import utils.utils as utils
from typing import Tuple
import torch
from scipy.ndimage import zoom

# Basler Camera used for imaging
from experiment.instruments.camera.basler_ace2.basler_ace2 import BaslerAce2
from experiment.instruments.camera.basler_ace2.basler_config import BaslerConfig
from experiment.toolkits.configs import BaslerAce2Properties

# Meadowlarks SLM
from experiment.instruments.spatial_light_modulator.meadowlark_p1920.meadowlark_p1920 import MeadowlarkP1920
from experiment.toolkits.configs import MeadowlarkP1920Properties


@dataclass
class BaslerCameraProperties:

    index: int
    pixel_format: int
    exposure_time: int
    gain: int
    roi: Tuple[int, int, int, int] # Y0, X0, Height, Width
    desired_size: Tuple[int, int]

    flag_flip_x: bool = True # Flip the image along the x axis
    flag_flip_y: bool = True # Flip the image along the y axis

@dataclass
class MeadowlarkSLMProperties:

    board_id: int
    lut_address: str


class UWatPhysicalProp(PhysicalProp):

    slm: MeadowlarkP1920
    camera: BaslerAce2

    def __init__(self, 
                 camera_properties: BaslerCameraProperties, 
                 slm_properties: MeadowlarkSLMProperties, 
                 device: str):
        torch.nn.Module.__init__(self)
        self._setup_slm(slm_properties)
        self._setup_camera(camera_properties)
        self.device = device

    def forward(self, slm_phase, num_grab_images=1):
        """
        this forward pass gets slm_phase to display and returns the amplitude image at the target plane.

        :param slm_phase: A pytorch tensor of shape (1, 1, H, W), 0-2pi
        :param num_grab_images:
        :return: A pytorch tensor shape of (1, 1, H, W)
        """
        # Upload the phase
        slm_phase_8bit = utils.phasemap_8bit(slm_phase, True)
        self.slm.upload_phase_image(slm_phase_8bit)

        # Acquire image and crop
        y0, x0, h, w = self.camera_properties.roi
        image_np = self.camera.get_one_result().Array / (2**self.camera_properties.pixel_format)
        if self.camera_properties.flag_flip_x:
            image_np = image_np[:, ::-1]
        if self.camera_properties.flag_flip_y:
            image_np = image_np[::-1, :]
        image_np_cropped = image_np[y0:y0+h, x0:x0+w]

        # Interpolate image
        N, M = image_np_cropped.shape # Image size
        H, W = self.camera_properties.desired_size # Desired size (phase size)
        image_np_interpolated = zoom(image_np_cropped, (H / N, W / M), order=3)
        image_np_interpolated = np.maximum(0, image_np_interpolated)

        # Transfer to torch tensor
        image_interpolated = torch.tensor(image_np_interpolated, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        
        # Uniformize the intensity
        image_interpolated /= image_interpolated.max()

        # Take square root of intensity and return
        return image_interpolated.sqrt()

    def _setup_slm(self, slm_properties: MeadowlarkSLMProperties):
        self.slm_properties = slm_properties

        self.slm = MeadowlarkP1920(board_id = slm_properties.board_id)
        self.slm.open_connection()

        self.slm.upload_lookup_table(slm_properties.lut_address)

    def _setup_camera(self, camera_properties: BaslerAce2Properties):
        self.camera_properties = camera_properties
        assert camera_properties.pixel_format in [8, 12], "The pixel format should be either 8 or 12."

        self.camera = BaslerAce2(
            BaslerConfig(
                camera_index = camera_properties.index,
                pixel_format = "Mono8" if camera_properties.pixel_format == 8 else "Mono12"
            )
        )
        self.camera.open_connection()

        self.camera.set_exposure_time(camera_properties.exposure_time)
        self.camera.set_gain(camera_properties.gain)

    def disconnect(self):
        self.camera.disconnect()
        self.slm.disconnect()