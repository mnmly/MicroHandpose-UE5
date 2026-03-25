#include "PalmDetectionPostProcess.h"

FPalmDetectionPostProcess::FPalmDetectionPostProcess()
{
	GenerateAnchors();
}

void FPalmDetectionPostProcess::GenerateAnchors()
{
	Anchors.Reserve(2016);

	// Layer 0: 12x12 grid, 6 anchors per cell
	for (int32 Y = 0; Y < 12; Y++)
	{
		for (int32 X = 0; X < 12; X++)
		{
			float CX = (X + 0.5f) / 12.0f;
			float CY = (Y + 0.5f) / 12.0f;
			for (int32 A = 0; A < 6; A++)
			{
				Anchors.Add({ CX, CY });
			}
		}
	}

	// Layer 1: 24x24 grid, 2 anchors per cell
	for (int32 Y = 0; Y < 24; Y++)
	{
		for (int32 X = 0; X < 24; X++)
		{
			float CX = (X + 0.5f) / 24.0f;
			float CY = (Y + 0.5f) / 24.0f;
			for (int32 A = 0; A < 2; A++)
			{
				Anchors.Add({ CX, CY });
			}
		}
	}

	check(Anchors.Num() == 2016);
}

TArray<FPalmDetection> FPalmDetectionPostProcess::DecodeDetections(
	const TArray<float>& Scores,
	const TArray<float>& Regressors,
	float ScoreThreshold) const
{
	TArray<FPalmDetection> Detections;
	constexpr float InputSize = 192.0f;

	for (int32 i = 0; i < Anchors.Num(); i++)
	{
		float Score = Sigmoid(Scores[i]);
		if (Score < ScoreThreshold) continue;

		const FAnchor& Anchor = Anchors[i];
		int32 RegBase = i * 18;

		FPalmDetection Det;
		Det.Score = Score;
		Det.CenterX = Anchor.X + Regressors[RegBase + 0] / InputSize;
		Det.CenterY = Anchor.Y + Regressors[RegBase + 1] / InputSize;
		Det.Width = Regressors[RegBase + 2] / InputSize;
		Det.Height = Regressors[RegBase + 3] / InputSize;

		// 7 keypoints
		for (int32 K = 0; K < 7; K++)
		{
			Det.Keypoints[K][0] = Anchor.X + Regressors[RegBase + 4 + K * 2] / InputSize;
			Det.Keypoints[K][1] = Anchor.Y + Regressors[RegBase + 4 + K * 2 + 1] / InputSize;
		}

		Detections.Add(Det);
	}

	return Detections;
}

float FPalmDetectionPostProcess::ComputeIoU(const FPalmDetection& A, const FPalmDetection& B)
{
	float AX1 = A.CenterX - A.Width / 2.0f;
	float AY1 = A.CenterY - A.Height / 2.0f;
	float AX2 = A.CenterX + A.Width / 2.0f;
	float AY2 = A.CenterY + A.Height / 2.0f;

	float BX1 = B.CenterX - B.Width / 2.0f;
	float BY1 = B.CenterY - B.Height / 2.0f;
	float BX2 = B.CenterX + B.Width / 2.0f;
	float BY2 = B.CenterY + B.Height / 2.0f;

	float IX1 = FMath::Max(AX1, BX1);
	float IY1 = FMath::Max(AY1, BY1);
	float IX2 = FMath::Min(AX2, BX2);
	float IY2 = FMath::Min(AY2, BY2);

	float IW = FMath::Max(0.0f, IX2 - IX1);
	float IH = FMath::Max(0.0f, IY2 - IY1);
	float Intersection = IW * IH;

	float AArea = (AX2 - AX1) * (AY2 - AY1);
	float BArea = (BX2 - BX1) * (BY2 - BY1);
	float Union = AArea + BArea - Intersection;

	return Union > 0.0f ? Intersection / Union : 0.0f;
}

TArray<FPalmDetection> FPalmDetectionPostProcess::WeightedNMS(
	const TArray<FPalmDetection>& Detections,
	float IoUThreshold)
{
	if (Detections.Num() == 0) return {};

	// Sort by score descending
	TArray<FPalmDetection> Sorted = Detections;
	Sorted.Sort([](const FPalmDetection& A, const FPalmDetection& B) { return A.Score > B.Score; });

	TArray<FPalmDetection> Kept;
	TSet<int32> Suppressed;

	for (int32 i = 0; i < Sorted.Num(); i++)
	{
		if (Suppressed.Contains(i)) continue;

		// Collect overlapping detections (cluster)
		TArray<int32> Cluster;
		Cluster.Add(i);

		for (int32 j = i + 1; j < Sorted.Num(); j++)
		{
			if (Suppressed.Contains(j)) continue;
			if (ComputeIoU(Sorted[i], Sorted[j]) > IoUThreshold)
			{
				Cluster.Add(j);
				Suppressed.Add(j);
			}
		}

		// Score-weighted average of box and keypoints
		float TotalWeight = 0.0f;
		float AvgCX = 0, AvgCY = 0, AvgW = 0, AvgH = 0;
		float AvgKps[7][2] = {};

		for (int32 Idx : Cluster)
		{
			const FPalmDetection& Det = Sorted[Idx];
			float W = Det.Score;
			TotalWeight += W;
			AvgCX += Det.CenterX * W;
			AvgCY += Det.CenterY * W;
			AvgW += Det.Width * W;
			AvgH += Det.Height * W;
			for (int32 K = 0; K < 7; K++)
			{
				AvgKps[K][0] += Det.Keypoints[K][0] * W;
				AvgKps[K][1] += Det.Keypoints[K][1] * W;
			}
		}

		float InvW = 1.0f / TotalWeight;
		FPalmDetection Result;
		Result.Score = Sorted[i].Score; // Keep top score
		Result.CenterX = AvgCX * InvW;
		Result.CenterY = AvgCY * InvW;
		Result.Width = AvgW * InvW;
		Result.Height = AvgH * InvW;
		for (int32 K = 0; K < 7; K++)
		{
			Result.Keypoints[K][0] = AvgKps[K][0] * InvW;
			Result.Keypoints[K][1] = AvgKps[K][1] * InvW;
		}

		Kept.Add(Result);
	}

	return Kept;
}

FHandROI FPalmDetectionPostProcess::DetectionToROI(const FPalmDetection& Detection)
{
	// Compute rotation from wrist (kp0) to middle finger MCP (kp2)
	float DX = Detection.Keypoints[2][0] - Detection.Keypoints[0][0];
	float DY = Detection.Keypoints[2][1] - Detection.Keypoints[0][1];
	float Angle = FMath::Atan2(DY, DX);

	// Target angle: hand pointing up = -PI/2
	float TargetAngle = -UE_PI / 2.0f;
	float Rotation = TargetAngle - Angle;

	// MediaPipe RectTransformationCalculator
	float LongSide = FMath::Max(Detection.Width, Detection.Height);
	float Scale = 2.6f;
	float Size = LongSide * Scale;

	// Shift in rotated frame: shift_y = -0.5 toward fingers
	float ShiftAmount = -0.5f * LongSide;
	float CosR = FMath::Cos(Rotation);
	float SinR = FMath::Sin(Rotation);

	FHandROI ROI;
	ROI.CenterX = Detection.CenterX + ShiftAmount * SinR;
	ROI.CenterY = Detection.CenterY + ShiftAmount * CosR;
	ROI.Width = Size;
	ROI.Height = Size;
	ROI.Rotation = Rotation;

	return ROI;
}

void FPalmDetectionPostProcess::RemoveLetterboxPadding(TArray<FPalmDetection>& Detections, float PadX, float PadY)
{
	constexpr float InputSize = 192.0f;
	float ScaleX = InputSize / (InputSize - 2.0f * PadX);
	float ScaleY = InputSize / (InputSize - 2.0f * PadY);
	float OffX = PadX / InputSize;
	float OffY = PadY / InputSize;

	for (FPalmDetection& Det : Detections)
	{
		Det.CenterX = (Det.CenterX - OffX) * ScaleX;
		Det.CenterY = (Det.CenterY - OffY) * ScaleY;
		Det.Width *= ScaleX;
		Det.Height *= ScaleY;

		for (int32 K = 0; K < 7; K++)
		{
			Det.Keypoints[K][0] = (Det.Keypoints[K][0] - OffX) * ScaleX;
			Det.Keypoints[K][1] = (Det.Keypoints[K][1] - OffY) * ScaleY;
		}
	}
}
