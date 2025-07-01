from enum import Enum

class ResourceType(Enum):
    """Resource type enumeration"""
    DSP = "DSP"
    NPU = "NPU"

class TaskPriority(Enum):
    """Task priority levels (from high to low)"""
    CRITICAL = 0     # Highest priority - safety critical tasks
    HIGH = 1         # High priority - real-time tasks
    NORMAL = 2       # Normal priority - regular tasks
    LOW = 3          # Low priority - background tasks
    
    def __lt__(self, other):
        """Enable priority comparison"""
        return self.value < other.value

class RuntimeType(Enum):
    """Runtime configuration types"""
    DSP_RUNTIME = "DSP_Runtime"      # DSP and NPU are coupled/bound together
    ACPU_RUNTIME = "ACPU_Runtime"    # Default runtime, segments can be pipelined

class SegmentationStrategy(Enum):
    """Network segmentation strategies"""
    NO_SEGMENTATION = "NoSeg"        # Use original segment without cutting
    ADAPTIVE_SEGMENTATION = "AdaptiveSeg"  # Automatically choose optimal cuts
    FORCED_SEGMENTATION = "ForcedSeg"      # Force segmentation at all cut points
    CUSTOM_SEGMENTATION = "CustomSeg"      # Use custom cut point selection

class CutPointStatus(Enum):
    """Status of a cut point in a segment"""
    ENABLED = "enabled"      # Cut point is used for segmentation
    DISABLED = "disabled"    # Cut point is ignored
    AUTO = "auto"           # Let scheduler decide whether to use this cut point