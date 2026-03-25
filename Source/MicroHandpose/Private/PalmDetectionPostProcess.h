#pragma once

#include "CoreMinimal.h"

/** A single SSD anchor */
struct FAnchor
{
	float X; // Center x in [0,1]
	float Y; // Center y in [0,1]
};

/** A decoded palm detection */
struct FPalmDetection
{
	float Score;
	float CenterX, CenterY, Width, Height; // Bounding box in normalized [0,1]
	float Keypoints[7][2]; // 7 keypoints: [x,y] in normalized [0,1]
};

/** Region of interest for hand landmark crop */
struct FHandROI
{
	float CenterX; // Center in original image coords [0,1]
	float CenterY;
	float Width;
	float Height;
	float Rotation; // Radians
};

/**
 * CPU-side post-processing for palm detection model output.
 * Handles anchor generation, detection decoding, NMS, and ROI computation.
 */
class FPalmDetectionPostProcess
{
public:
	FPalmDetectionPostProcess();

	/**
	 * Decode raw SSD output into palm detections.
	 * @param Scores Raw classifier logits [2016]
	 * @param Regressors Raw regression values [2016 * 18]
	 * @param ScoreThreshold Minimum sigmoid score to keep
	 * @return Decoded detections
	 */
	TArray<FPalmDetection> DecodeDetections(const TArray<float>& Scores, const TArray<float>& Regressors, float ScoreThreshold) const;

	/**
	 * Weighted non-maximum suppression (MediaPipe WEIGHTED algorithm).
	 * @param Detections Input detections (sorted by score internally)
	 * @param IoUThreshold Overlap threshold for suppression
	 * @return Filtered detections with score-weighted averaging
	 */
	static TArray<FPalmDetection> WeightedNMS(const TArray<FPalmDetection>& Detections, float IoUThreshold);

	/**
	 * Convert a palm detection to a hand crop ROI.
	 * Uses wrist (kp0) and middle finger MCP (kp2) for orientation.
	 */
	static FHandROI DetectionToROI(const FPalmDetection& Detection);

	/**
	 * Remove letterbox padding from detections.
	 * @param Detections In/out detections to correct
	 * @param PadX Horizontal padding in pixels (relative to 192)
	 * @param PadY Vertical padding in pixels (relative to 192)
	 */
	static void RemoveLetterboxPadding(TArray<FPalmDetection>& Detections, float PadX, float PadY);

private:
	TArray<FAnchor> Anchors; // Pre-generated, 2016 total

	void GenerateAnchors();
	static float ComputeIoU(const FPalmDetection& A, const FPalmDetection& B);
	static float Sigmoid(float X) { return 1.0f / (1.0f + FMath::Exp(-X)); }
};
