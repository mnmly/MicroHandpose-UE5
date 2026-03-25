# MicroHandpose-UE5

Real-time hand pose estimation plugin for Unreal Engine 5.7. Detects hands and outputs 21 3D landmarks per hand, entirely on the GPU using compute shaders. No external dependencies (TensorFlow, ONNX, etc.) required.

Ported from [micro-handpose](https://github.com/svenflow/micro-handpose), a WebGPU hand tracking library by [svenflow](https://github.com/svenflow). The pipeline architecture, model weights, and compute shader logic are adapted from that project. The original model weights are derived from [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).

## Requirements

- Unreal Engine 5.7+
- Metal SM6 (macOS). Not tested for Windows
- A webcam or video input source

## Installation

1. Clone or copy this repository into your project's `Plugins/` directory:
   ```
   YourProject/
     Plugins/
       MicroHandpose/
   ```
2. Regenerate project files and build.

## Quick Start: Webcam Hand Tracking

### Step 1: Create Media Assets

In the Content Browser, create three assets:

1. **Media Player** - Right-click > Media > Media Player
   - Uncheck "Play on Open" in the asset (optional)

2. **Media Texture** - Created automatically when you create the Media Player (or create manually: Right-click > Media > Media Texture)
   - In the Media Texture details, set **Media Player** to your Media Player asset

3. **File Media Source** (for webcam) - Right-click > Media > File Media Source
   - Set the **File Path** to your camera URL:
     - **macOS FaceTime/webcam**: `avf://`
     - **macOS specific camera**: `avf://camera:0` (first camera), `avf://camera:1` (second), etc.
     - **Windows webcam**: use a DirectShow or Media Foundation URL, or leave empty and use the platform's default capture device

### Step 2: Blueprint Setup

In your Level Blueprint or any Actor Blueprint:

1. **Begin Play**:
   ```
   Get Handpose Subsystem
     -> Set Max Hands (2)
     -> Set Score Threshold (0.5)
     -> Set Palm Score Threshold (0.5)

   Open Source (Media Player, File Media Source)

   Get Handpose Subsystem -> Start Tracking (Media Texture)
   ```

2. **Read Results** (pick one approach):

   **Option A: Poll on Tick**
   ```
   Get Handpose Subsystem -> Get Latest Results -> For Each (Hand Result)
     -> Access: Score, Handedness, Landmarks array
   ```

   **Option B: Event-Driven**
   ```
   Get Handpose Subsystem -> Bind Event to On Handpose Updated
     -> Event receives TArray<HandposeResult>
   ```

3. **End Play**:
   ```
   Get Handpose Subsystem -> Stop Tracking
   Close (Media Player)
   ```

### Step 3: C++ Setup (Alternative)

```cpp
#include "HandposeSubsystem.h"
#include "MediaPlayer.h"
#include "MediaTexture.h"

// In your actor's BeginPlay:
void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    // Assuming MediaPlayer and MediaTexture are UPROPERTY references set in editor
    MediaPlayer->OpenUrl(TEXT("avf://"));

    UHandposeSubsystem* Handpose = UHandposeSubsystem::GetInstance();
    Handpose->SetMaxHands(2);
    Handpose->StartTracking(MediaTexture);
}

// In Tick:
void AMyActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    UHandposeSubsystem* Handpose = UHandposeSubsystem::GetInstance();
    TArray<FHandposeResult> Hands = Handpose->GetLatestResults();

    for (const FHandposeResult& Hand : Hands)
    {
        FVector IndexTip = Hand.GetIndexTip().Position;  // normalized [0,1]
        EHandedness Side = Hand.Handedness;               // Left or Right
        float Confidence = Hand.Score;                     // 0-1
    }
}
```

## API Reference

### UHandposeSubsystem

| Method | Description |
|--------|-------------|
| `StartTracking(UMediaTexture*)` | Begin hand tracking with the given media texture as input |
| `StopTracking()` | Stop tracking and release resources |
| `GetLatestResults()` | Returns `TArray<FHandposeResult>` with current detections |
| `SetMaxHands(int32)` | Max simultaneous hands to detect (1-3, default: 2) |
| `SetScoreThreshold(float)` | Minimum landmark confidence (0-1, default: 0.5) |
| `SetPalmScoreThreshold(float)` | Minimum palm detection confidence (0-1, default: 0.5) |
| `IsTracking()` | Whether tracking is active |
| `OnHandposeUpdated` | Delegate fired each frame with results |

### FHandposeResult

| Field | Type | Description |
|-------|------|-------------|
| `Score` | `float` | Detection confidence (0-1) |
| `Handedness` | `EHandedness` | `Left` or `Right` |
| `Landmarks` | `TArray<FHandLandmark>` | 21 landmark points |

Convenience accessors: `GetWrist()`, `GetThumbTip()`, `GetIndexTip()`, `GetMiddleTip()`, `GetRingTip()`, `GetPinkyTip()`.

### 21 Hand Landmarks

```
        4-ThumbTip
       /
      3-ThumbIP        8-IndexTip   12-MiddleTip  16-RingTip   20-PinkyTip
     /                /             /             /             /
    2-ThumbMCP       7-IndexDIP   11-MiddleDIP  15-RingDIP   19-PinkyDIP
   /                /             /             /             /
  1-ThumbCMC       6-IndexPIP   10-MiddlePIP  14-RingPIP   18-PinkyPIP
   \              /             /             /             /
    \            5-IndexMCP    9-MiddleMCP  13-RingMCP   17-PinkyMCP
     \          /             /             /             /
      --------0-Wrist---------------------------------
```

## Architecture

The pipeline runs entirely on the GPU via compute shaders:

1. **Letterbox Resize** - Input image scaled to 192x192, maintaining aspect ratio
2. **Palm Detection** (BlazeNet SSD) - 24-layer backbone + FPN + SSD heads
3. **CPU Post-Process** - Anchor decode, weighted NMS, ROI extraction
4. **Affine Crop** - Per-hand 224x224 rotated crop based on palm ROI
5. **Landmark Regression** (EfficientNet-B0) - 16 MBConv blocks + 3 FC heads
6. **CPU Post-Process** - Inverse affine transform, denormalize to image coordinates

Async multi-frame pipeline: palm detection and landmark inference use GPU readback across frames to avoid stalling.

## Troubleshooting

- **No hands detected**: Lower `PalmScoreThreshold` (try 0.3). Ensure your hand is well-lit and clearly visible.
- **Shader compilation errors**: Delete `Saved/ShaderDebugInfo` and `Intermediate/Shaders` directories, then relaunch.
- **Camera not opening**: Verify the camera URL. On macOS, try `avf://` first. Check system camera permissions for Unreal Editor.
- **Low FPS**: The pipeline uses async readback, so detection runs 1-2 frames behind. If still slow, reduce `MaxHands` to 1.

## Credits

- Pipeline architecture, compute shader logic, and model weights ported from [micro-handpose](https://github.com/svenflow/micro-handpose) by [svenflow](https://github.com/svenflow) (MIT license).
- Hand landmark model architecture and weights originally from [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe) by Google (Apache 2.0 license).

## License

MIT -- see [LICENSE](./LICENSE) for details.
