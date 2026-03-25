#pragma once

#include "CoreMinimal.h"
#include "HandposeTypes.h"
#include "LandmarkPostProcess.h"
#include "RenderGraphResources.h"
#include "RHIGPUReadback.h"

struct FWeightCollection;
class FPalmDetectionPostProcess;

/**
 * GPU weight buffer: a single tensor uploaded to GPU as a StructuredBuffer<float>.
 */
struct FGPUWeightBuffer
{
	TRefCountPtr<FRDGPooledBuffer> PooledBuffer;
	int32 NumElements = 0;
};

/**
 * Specification for one backbone block in the palm detection model.
 */
struct FPalmBackboneBlock
{
	int32 DwIdx;     // depthwise_conv2d index
	int32 PwIdx;     // conv2d index for pointwise
	int32 InCh;
	int32 OutCh;
	int32 Stride;
	int32 InH;       // Input spatial height (square)
};

/**
 * Specification for one MBConv block in the EfficientNet landmark model.
 */
struct FMBConvBlock
{
	int32 InCh;
	int32 ExpandCh;
	int32 DwKernel;
	int32 Stride;
	int32 OutCh;
	bool bHasResidual;
	bool bHasProject;
	int32 SpatialOut; // Output spatial size (square)
};

/**
 * Pipeline execution phase for async multi-frame inference.
 */
enum class EPipelinePhase : uint8
{
	Idle,
	PalmDispatched,       // Palm detection dispatched, waiting for GPU readback
	LandmarkDispatched,   // Landmark inference dispatched, waiting for GPU readback
};

/**
 * Orchestrates the full hand pose estimation pipeline on GPU.
 *
 * Owns:
 * - Weight buffers for both palm detection and landmark models
 * - Intermediate tensor buffers (allocated on first use)
 * - The dispatch sequence for each model
 * - Async readback state for multi-frame pipeline
 *
 * Usage:
 * 1. Call Initialize() once to load weights
 * 2. Call RunPipeline() each frame from the render thread
 */
class MICROHANDPOSE_API FHandposeDetector
{
public:
	FHandposeDetector();
	~FHandposeDetector();

	/** Load model weights from plugin Resources. Must be called before RunPipeline. */
	bool Initialize(const FString& PluginBaseDir);

	/** Whether weights have been successfully loaded */
	bool IsInitialized() const { return bInitialized; }

	/**
	 * Run the full pipeline: palm detection -> crop -> landmark for each hand.
	 * Must be called from the render thread. Uses async readback across frames.
	 *
	 * @param GraphBuilder RDG graph builder
	 * @param InputTexture Camera/input texture to process
	 * @param MaxHands Maximum hands to detect
	 * @param ScoreThreshold Minimum landmark confidence
	 * @param PalmScoreThreshold Minimum palm detection confidence
	 * @return Detected hands with landmarks (from previous completed inference)
	 */
	TArray<FHandposeResult> RunPipeline(
		FRDGBuilder& GraphBuilder,
		FRDGTexture* InputTexture,
		int32 MaxHands,
		float ScoreThreshold,
		float PalmScoreThreshold);

private:
	bool bInitialized = false;

	// Palm detection model weights
	TMap<FString, FGPUWeightBuffer> PalmWeightBuffers;

	// Landmark model weights
	TMap<FString, FGPUWeightBuffer> LandmarkWeightBuffers;

	// Palm model block specifications
	TArray<FPalmBackboneBlock> PalmBlocks;

	// Landmark model block specifications
	TArray<FMBConvBlock> LandmarkBlocks;

	// Zero-bias buffer for palm DW convolutions (bias folded into weights)
	FGPUWeightBuffer ZeroBias256; // Large enough for any channel count

	// Weight loading helpers
	FGPUWeightBuffer UploadWeightToGPU(const TArray<float>& Data, const FString& DebugName);
	bool LoadPalmWeights(const FString& PluginBaseDir);
	bool LoadLandmarkWeights(const FString& PluginBaseDir);

	// Pipeline helpers
	void InitPalmBlockSpecs();
	void InitLandmarkBlockSpecs();

	// Weight SRV lookup helpers
	FRDGBufferSRVRef GetWeightSRV(FRDGBuilder& GraphBuilder, const TMap<FString, FGPUWeightBuffer>& WeightMap, const FString& ExactKey) const;
	FRDGBufferSRVRef FindWeightSRV(FRDGBuilder& GraphBuilder, const TMap<FString, FGPUWeightBuffer>& WeightMap, const TArray<FString>& Substrings) const;
	FRDGBufferSRVRef GetZeroBiasSRV(FRDGBuilder& GraphBuilder) const;

	// ---- Palm Detection Dispatch ----
	void DispatchPalmDetection(FRDGBuilder& GraphBuilder, FRDGTexture* InputTexture);

	// ---- Landmark Dispatch ----
	void DispatchLandmarkInference(FRDGBuilder& GraphBuilder, FRDGTexture* InputTexture, const FLandmarkPostProcess::FPixelROI& PixelROI, int32 HandIndex);

	// SSD output reorganization (CHW -> HWC anchor order)
	void ReorganizeSSDOutput(
		const TArray<float>& Cls16, const TArray<float>& Reg16,
		const TArray<float>& Cls8, const TArray<float>& Reg8,
		TArray<float>& OutScores, TArray<float>& OutRegressors) const;

	// ---- Pipeline State ----
	EPipelinePhase Phase = EPipelinePhase::Idle;

	// Post-processing
	TUniquePtr<FPalmDetectionPostProcess> PalmPostProcess;

	// Cached results from last completed inference
	TArray<FHandposeResult> CachedResults;

	// Source image dimensions (from last input texture)
	int32 SrcWidth = 0;
	int32 SrcHeight = 0;

	// Letterbox padding (needed for removing padding from detections)
	float LetterboxPadX = 0.0f;
	float LetterboxPadY = 0.0f;

	// Pending pixel ROIs for landmark inference (from decoded palm detections)
	TArray<FLandmarkPostProcess::FPixelROI> PendingROIs;

	// ---- Readback Objects ----
	// Palm detection readback (4 SSD output buffers)
	FRHIGPUBufferReadback* PalmCls16Readback = nullptr;
	FRHIGPUBufferReadback* PalmReg16Readback = nullptr;
	FRHIGPUBufferReadback* PalmCls8Readback = nullptr;
	FRHIGPUBufferReadback* PalmReg8Readback = nullptr;

	// Landmark readback (one per hand, each holds 65 floats: handflag + handedness + 63 landmarks)
	TArray<FRHIGPUBufferReadback*> LandmarkReadbacks;
	TArray<FLandmarkPostProcess::FPixelROI> LandmarkROIs; // Pixel ROIs corresponding to each readback
};
