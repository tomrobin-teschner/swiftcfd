from enum import Enum, auto

class PrimitiveVariables(Enum):
    velocity_x = auto()
    velocity_y = auto()
    pressure = auto()
    temperature = auto()

    def description(self):
        descriptions = {
            PrimitiveVariables.velocity_x: "Velocity in the x-direction",
            PrimitiveVariables.velocity_y: "Velocity in the y-direction",
            PrimitiveVariables.pressure: "Fluid pressure",
            PrimitiveVariables.temperature: "Fluid temperature",
        }
        return descriptions[self]

    def units(self):
        units = {
            PrimitiveVariables.velocity_x: "m/s",
            PrimitiveVariables.velocity_y: "m/s",
            PrimitiveVariables.pressure: "Pa",
            PrimitiveVariables.temperature: "K",
        }
        return units[self]
    
    def name(self):
        names = {
            PrimitiveVariables.velocity_x: "u",
            PrimitiveVariables.velocity_y: "v",
            PrimitiveVariables.pressure: "p",
            PrimitiveVariables.temperature: "T",
        }
        return names[self]
