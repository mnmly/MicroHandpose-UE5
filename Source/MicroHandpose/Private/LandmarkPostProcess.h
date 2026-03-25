#pragma once

#include "CoreMinimal.h"
#include "HandposeTypes.h"
#include "PalmDetectionPostProcess.h"

/**
 * CPU-side post-processing for hand landmark model output.
 * Handles landmark denormalization, ROI computation from previous landmarks,
 * and affine matrix computation for GPU crop.
 */
class FLandmarkPostProcess
{
public:
	struct FPixelROI
	{
		float CenterXPx;
		float CenterYPx;
		float SizePx;
		float Rotation;
	};

	/**
	 * Denormalize landmark model output from crop-space [0,1] to image-space [0,1].
	 * @param RawLandmarks 63 float values (21 landmarks × 3: x, y, z) in [0,1] crop coords
	 * @param HandFlag Sigmoid confidence (0-1)
	 * @param Handedness Sigmoid value (>0.5 = right)
	 * @param ROI The pixel-space crop ROI used for inference
	 * @param SrcWidth Source image width in pixels
	 * @param SrcHeight Source image height in pixels
	 * @return Denormalized hand result
	 */
	static FHandposeResult DenormalizeLandmarks(
		const TArray<float>& RawLandmarks,
		float HandFlag,
		float Handedness,
		const FPixelROI& ROI,
		int32 SrcWidth,
		int32 SrcHeight);

	/**
	 * Compute a pixel-space ROI from palm detection.
	 * @param Detection Palm detection in normalized [0,1] coords
	 * @param ImgW Image width in pixels
	 * @param ImgH Image height in pixels
	 * @return Pixel-space ROI {centerX_px, centerY_px, size_px, rotation}
	 */
	static FPixelROI DetectionToPixelROI(const FPalmDetection& Detection, int32 ImgW, int32 ImgH);

	/**
	 * Compute ROI from previous frame's landmarks (tracking path).
	 * Uses landmarks 0(wrist), 5(indexMCP), 9(middleMCP), 13(ringMCP) for rotation,
	 * and partial landmarks for bounding box.
	 * @param Landmarks Previous frame's 21 landmarks in normalized [0,1] image coords
	 * @param ImgW Image width
	 * @param ImgH Image height
	 * @return Pixel-space ROI for next frame's crop
	 */
	static FPixelROI LandmarksToROI(const TArray<FHandLandmark>& Landmarks, int32 ImgW, int32 ImgH);

	/**
	 * Compute the 2x3 affine matrix that maps crop destination coords [0..cropSize] to
	 * source texture UV [0..1]. Used by the AffineCrop compute shader.
	 * @param ROI Pixel-space ROI
	 * @param SrcW Source width
	 * @param SrcH Source height
	 * @param CropSize Crop output size (e.g. 224)
	 * @return 4x4 matrix (only first 2 rows meaningful for 2D affine)
	 */
	static FMatrix44f ComputeAffineMatrix(const FPixelROI& ROI, int32 SrcW, int32 SrcH, int32 CropSize);

private:
	/** Normalize angle to [-PI, PI] */
	static float NormalizeRadians(float Angle);
};
