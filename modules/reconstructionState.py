from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ReconstructionState:
    hologram: np.ndarray
    wavelength: float
    distance: float
    roi: Optional[tuple] = None
    mask: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None

@dataclass(frozen=True)
class ReconstructionParams:
    BASE_PIXEL_SIZE = 2.2e-6

    wavelength: float
    mtransv: float # Microscope Objective Magnification [x]
    decmt: bool # Decimate (1/2) pixel sampling?
    dx: float = field(init=False) # pixel pitch
    nout: int # Hologram size

    def __post_init__(self):
        # Pixel subsampling in case of color camera (false -> mono camera: Guppy Pro 503b, true -> color camera: Guppy Pro 503c)
        object.__setattr__(self, "dx", 2 * self.BASE_PIXEL_SIZE if self.decmt else self.BASE_PIXEL_SIZE)
        
    def zi_to_zo(self, zi: float) -> float:
        fl = 16
        zo = 1000*zi*fl/(self.mtransv*(zi + fl*self.mtransv))
        return zo

    def zo_to_zi(self, zo: float) -> float:
        fl = 16
        zi = (zo*fl*self.mtransv**2)/(-zo*self.mtransv + fl)
        return zi