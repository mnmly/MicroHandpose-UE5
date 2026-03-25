#include "LandmarkPostProcess.h"

float FLandmarkPostProcess::NormalizeRadians(float Angle)
{
	return Angle - 2.0f * UE_PI * FMath::FloorToFloat((Angle + UE_PI) / (2.0f * UE_PI));
}

FHandposeResult FLandmarkPostProcess::DenormalizeLandmarks(
	const TArray<float>& RawLandmarks,
	float HandFlag,
	float Handedness,
	const FPixelROI& ROI,
	int32 SrcWidth,
	int32 SrcHeight)
{
	FHandposeResult Result;
	Result.Score = HandFlag;
	Result.Handedness = (Handedness > 0.5f) ? EHandedness::Right : EHandedness::Left;
	Result.Landmarks.SetNum(NUM_HAND_LANDMARKS);

	float CosR = FMath::Cos(ROI.Rotation);
	float SinR = FMath::Sin(ROI.Rotation);

	for (int32 i = 0; i < NUM_HAND_LANDMARKS; i++)
	{
		float LX = RawLandmarks[i * 3];     // [0,1] in crop space
		float LY = RawLandmarks[i * 3 + 1]; // [0,1] in crop space
		float LZ = RawLandmarks[i * 3 + 2]; // Relative depth

		// Transform from crop [0,1] to pixel offset from center
		float DX = (LX - 0.5f) * ROI.SizePx;
		float DY = (LY - 0.5f) * ROI.SizePx;

		// Rotate back to original image space
		float OrigXPx = CosR * DX - SinR * DY + ROI.CenterXPx;
		float OrigYPx = SinR * DX + CosR * DY + ROI.CenterYPx;

		Result.Landmarks[i].Position = FVector(
			OrigXPx / static_cast<float>(SrcWidth),
			OrigYPx / static_cast<float>(SrcHeight),
			LZ
		);
	}

	return Result;
}

FLandmarkPostProcess::FPixelROI FLandmarkPostProcess::DetectionToPixelROI(
	const FPalmDetection& Detection,
	int32 ImgW,
	int32 ImgH)
{
	// Compute rotation from wrist(kp0) to middle finger MCP(kp2) in pixel space
	float DXPx = (Detection.Keypoints[2][0] - Detection.Keypoints[0][0]) * ImgW;
	float DYPx = (Detection.Keypoints[2][1] - Detection.Keypoints[0][1]) * ImgH;
	float Angle = FMath::Atan2(-DYPx, DXPx);
	float TargetAngle = UE_PI / 2.0f;
	float Rotation = NormalizeRadians(TargetAngle - Angle);

	float CosR = FMath::Cos(Rotation);
	float SinR = FMath::Sin(Rotation);

	// Shift: shift_x=0, shift_y=-0.5 (MediaPipe RectTransformationCalculator)
	float BoxHPx = Detection.Height * ImgH;
	float NewCX = Detection.CenterX + (0.5f * BoxHPx * SinR) / ImgW;
	float NewCY = Detection.CenterY + (-0.5f * Detection.Height * CosR);

	// Square: long side in pixels
	float LongSidePx = FMath::Max(Detection.Width * ImgW, Detection.Height * ImgH);
	float Scale = 2.6f;
	float SizePx = LongSidePx * Scale;

	FPixelROI ROI;
	ROI.CenterXPx = NewCX * ImgW;
	ROI.CenterYPx = NewCY * ImgH;
	ROI.SizePx = SizePx;
	ROI.Rotation = Rotation;
	return ROI;
}

FLandmarkPostProcess::FPixelROI FLandmarkPostProcess::LandmarksToROI(
	const TArray<FHandLandmark>& Landmarks,
	int32 ImgW,
	int32 ImgH)
{
	// Compute rotation from wrist + averaged finger MCPs
	FVector Wrist = Landmarks[0].Position;
	FVector IndexMCP = Landmarks[5].Position;
	FVector MiddleMCP = Landmarks[9].Position;
	FVector RingMCP = Landmarks[13].Position;

	float X0 = Wrist.X * ImgW;
	float Y0 = Wrist.Y * ImgH;
	float X1 = ((IndexMCP.X + RingMCP.X) / 2.0f + MiddleMCP.X) / 2.0f * ImgW;
	float Y1 = ((IndexMCP.Y + RingMCP.Y) / 2.0f + MiddleMCP.Y) / 2.0f * ImgH;

	float RawRotation = UE_PI / 2.0f - FMath::Atan2(-(Y1 - Y0), X1 - X0);
	float Rotation = NormalizeRadians(RawRotation);

	// Extract 12 partial landmarks for bounding box
	static const int32 PartialIndices[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
	float CosR = FMath::Cos(Rotation);
	float SinR = FMath::Sin(Rotation);

	float MinRX = TNumericLimits<float>::Max(), MaxRX = TNumericLimits<float>::Lowest();
	float MinRY = TNumericLimits<float>::Max(), MaxRY = TNumericLimits<float>::Lowest();

	for (int32 Idx : PartialIndices)
	{
		float PX = Landmarks[Idx].Position.X * ImgW;
		float PY = Landmarks[Idx].Position.Y * ImgH;
		float RX = CosR * PX + SinR * PY;
		float RY = -SinR * PX + CosR * PY;
		MinRX = FMath::Min(MinRX, RX);
		MaxRX = FMath::Max(MaxRX, RX);
		MinRY = FMath::Min(MinRY, RY);
		MaxRY = FMath::Max(MaxRY, RY);
	}

	// Center in rotated space, then rotate back
	float RCX = (MinRX + MaxRX) / 2.0f;
	float RCY = (MinRY + MaxRY) / 2.0f;
	float CX = (CosR * RCX - SinR * RCY);
	float CY = (SinR * RCX + CosR * RCY);

	float BoxW = MaxRX - MinRX;
	float BoxH = MaxRY - MinRY;

	// Shift_y = -0.1 toward fingers
	float ShiftY = -0.1f;
	float BoxHNorm = BoxH / ImgH;
	float CXNorm = CX / ImgW + (-ImgH * BoxHNorm * ShiftY * SinR) / ImgW;
	float CYNorm = CY / ImgH + BoxHNorm * ShiftY * CosR;

	// Square: long side in pixels
	float LongSidePx = FMath::Max(BoxW, BoxH);
	float Scale = 2.0f; // Tracking path uses 2.0 (palm path uses 2.6)
	float SizePx = LongSidePx * Scale;

	FPixelROI ROI;
	ROI.CenterXPx = CXNorm * ImgW;
	ROI.CenterYPx = CYNorm * ImgH;
	ROI.SizePx = SizePx;
	ROI.Rotation = Rotation;
	return ROI;
}

FMatrix44f FLandmarkPostProcess::ComputeAffineMatrix(
	const FPixelROI& ROI,
	int32 SrcW,
	int32 SrcH,
	int32 CropSize)
{
	// Compute affine: maps crop dst normalized [0..1] → source UV [0..1]
	// The shader normalizes dst pixel (dx,dy) to [0,1] before applying this matrix.

	float CosR = FMath::Cos(ROI.Rotation);
	float SinR = FMath::Sin(ROI.Rotation);

	// Scale: ROI.SizePx covers the full [0..1] range of the crop
	float SX = ROI.SizePx / static_cast<float>(SrcW);
	float SY = ROI.SizePx / static_cast<float>(SrcH);

	// Rotation + scale
	float A =  CosR * SX;
	float B = -SinR * SX;
	float C =  SinR * SY;
	float D =  CosR * SY;

	// Translation: center of crop [0.5, 0.5] maps to ROI center in UV space
	float CenterU = ROI.CenterXPx / static_cast<float>(SrcW);
	float CenterV = ROI.CenterYPx / static_cast<float>(SrcH);
	float TX = CenterU - 0.5f * (A + B);
	float TY = CenterV - 0.5f * (C + D);

	FMatrix44f Mat = FMatrix44f::Identity;
	Mat.M[0][0] = A;  Mat.M[0][1] = B;  Mat.M[0][2] = TX;
	Mat.M[1][0] = C;  Mat.M[1][1] = D;  Mat.M[1][2] = TY;
	return Mat;
}
