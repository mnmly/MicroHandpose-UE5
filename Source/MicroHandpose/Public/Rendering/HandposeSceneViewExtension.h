#pragma once

#include "CoreMinimal.h"
#include "SceneViewExtension.h"
#include "RenderGraphResources.h"
#include "HandposeDetector.h"
#include "HandposeTypes.h"

class MICROHANDPOSE_API FHandposeSceneViewExtension : public FSceneViewExtensionBase
{
public:
	FHandposeSceneViewExtension(const FAutoRegister& AutoRegister);
	virtual ~FHandposeSceneViewExtension() override;

	// Configuration (set from game thread)
	void SetMaxHands(int32 InMaxHands) { MaxHands = FMath::Clamp(InMaxHands, 1, 3); }
	void SetScoreThreshold(float InThreshold) { ScoreThreshold = FMath::Clamp(InThreshold, 0.0f, 1.0f); }
	void SetPalmScoreThreshold(float InThreshold) { PalmScoreThreshold = FMath::Clamp(InThreshold, 0.0f, 1.0f); }
	void SetEnabled(bool bInEnabled) { bEnabled = bInEnabled; }
	bool IsEnabled() const { return bEnabled; }

	/** Set an external RHI texture as input (e.g. from MediaTexture). Thread-safe. */
	void SetInputTextureRHI(FRHITexture* InTexture) { ExternalTextureRHI = InTexture; }

	/** Set the texture resource to resolve RHI texture from on the render thread. */
	void SetInputTextureResource(FTextureResource* InResource) { InputTextureResource = InResource; }

	/** Initialize the detector (loads weights). Call from render thread before enabling. */
	bool InitializeDetector(const FString& PluginBaseDir);

	// Thread-safe result retrieval (called from game thread)
	TArray<FHandposeResult> GetLatestResults();

	// Scene View Extension overrides
	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {}
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}
	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override {}
	virtual void PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override {}
	virtual void PreRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView) override {}
	virtual void PostRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView) override {}
	virtual void PostRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override;

	virtual void PrePostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessingInputs& Inputs) override {}

private:
	bool bEnabled = false;
	int32 MaxHands = 2;
	float ScoreThreshold = 0.5f;
	float PalmScoreThreshold = 0.5f;
	bool bLoggedFirstRun = false;

	// External input texture (e.g. webcam via MediaTexture)
	FRHITexture* ExternalTextureRHI = nullptr;
	FTextureResource* InputTextureResource = nullptr;

	// The detector (owns weights, pipeline state, readback objects)
	FHandposeDetector Detector;

	// Thread-safe results
	TArray<FHandposeResult> PendingResults;
	FCriticalSection ResultGuard;
};
