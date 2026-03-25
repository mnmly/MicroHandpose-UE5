#include "Rendering/HandposeSceneViewExtension.h"
#include "MicroHandposeModule.h"
#include "RenderGraphBuilder.h"
#include "RendererInterface.h"

FHandposeSceneViewExtension::FHandposeSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
}

FHandposeSceneViewExtension::~FHandposeSceneViewExtension()
{
}

bool FHandposeSceneViewExtension::InitializeDetector(const FString& PluginBaseDir)
{
	if (Detector.IsInitialized())
	{
		return true;
	}
	return Detector.Initialize(PluginBaseDir);
}

TArray<FHandposeResult> FHandposeSceneViewExtension::GetLatestResults()
{
	FScopeLock Lock(&ResultGuard);
	return PendingResults;
}

void FHandposeSceneViewExtension::PostRenderViewFamily_RenderThread(
	FRDGBuilder& GraphBuilder,
	FSceneViewFamily& InViewFamily)
{
	if (!bEnabled || !Detector.IsInitialized())
	{
		return;
	}

	// Resolve the RHI texture on the render thread (MediaTexture updates TextureRHI here)
	FRHITexture* ResolvedTexture = ExternalTextureRHI;
	if (!ResolvedTexture && InputTextureResource)
	{
		ResolvedTexture = InputTextureResource->TextureRHI.GetReference();
	}

	if (!ResolvedTexture)
	{
		return;
	}

	// Skip UE's 2x2 placeholder texture
	FIntVector TexSize = ResolvedTexture->GetSizeXYZ();
	if (TexSize.X <= 2 || TexSize.Y <= 2)
	{
		return;
	}

	if (!bLoggedFirstRun)
	{
		UE_LOG(LogMicroHandpose, Log, TEXT("SVE: first pipeline run — input texture %dx%d, format=%d"),
			TexSize.X, TexSize.Y, (int32)ResolvedTexture->GetFormat());
		bLoggedFirstRun = true;
	}

	// Wrap the external RHI texture as a pooled render target so RDG can use it
	TRefCountPtr<IPooledRenderTarget> PooledRT = CreateRenderTarget(ResolvedTexture, TEXT("HandposeInput"));
	FRDGTextureRef InputTexture = GraphBuilder.RegisterExternalTexture(PooledRT, TEXT("HandposeInputTex"));

	// Run the pipeline
	TArray<FHandposeResult> Results = Detector.RunPipeline(
		GraphBuilder,
		InputTexture,
		MaxHands,
		ScoreThreshold,
		PalmScoreThreshold);

	// Store results thread-safely
	{
		FScopeLock Lock(&ResultGuard);
		PendingResults = MoveTemp(Results);
	}
}
