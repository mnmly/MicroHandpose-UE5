#include "HandposeSubsystem.h"
#include "MicroHandposeModule.h"
#include "Rendering/HandposeSceneViewExtension.h"
#include "MediaTexture.h"
#include "MediaPlayer.h"
#include "Interfaces/IPluginManager.h"
#include "RenderingThread.h"

void UHandposeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	SceneViewExtension = FSceneViewExtensions::NewExtension<FHandposeSceneViewExtension>();

	BeginFrameHandle = FCoreDelegates::OnBeginFrame.AddUObject(this, &UHandposeSubsystem::OnBeginFrame);
	WorldCleanupHandle = FWorldDelegates::OnWorldCleanup.AddUObject(this, &UHandposeSubsystem::OnWorldCleanup);
}

void UHandposeSubsystem::Deinitialize()
{
	StopTracking();

	FCoreDelegates::OnBeginFrame.Remove(BeginFrameHandle);
	FWorldDelegates::OnWorldCleanup.Remove(WorldCleanupHandle);

	SceneViewExtension.Reset();

	Super::Deinitialize();
}

UHandposeSubsystem* UHandposeSubsystem::GetInstance()
{
	if (GEngine)
	{
		return GEngine->GetEngineSubsystem<UHandposeSubsystem>();
	}
	return nullptr;
}

void UHandposeSubsystem::OnBeginFrame()
{
	if (!bIsTracking || !SceneViewExtension.IsValid())
	{
		return;
	}

	// Pass the media texture's resource to the SVE — it resolves the RHI texture on the render thread
	if (InputMediaTexture)
	{
		FTextureResource* Resource = InputMediaTexture->GetResource();
		if (Resource)
		{
			SceneViewExtension->SetInputTextureResource(Resource);
		}
		else
		{
			TextureWaitFrames++;
			if (TextureWaitFrames % 60 == 1)
			{
				UE_LOG(LogMicroHandpose, Warning, TEXT("Subsystem: MediaTexture has no resource yet (frame %d)"), TextureWaitFrames);
			}
		}
	}

	// Pull latest results from render thread
	TArray<FHandposeResult> NewResults = SceneViewExtension->GetLatestResults();

	{
		FScopeLock Lock(&ResultGuard);
		LatestResults = MoveTemp(NewResults);
	}

	if (LatestResults.Num() > 0)
	{
		OnHandposeUpdated.Broadcast(LatestResults);
	}
}

void UHandposeSubsystem::OnWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources)
{
	StopTracking();
}

void UHandposeSubsystem::StartTracking(UMediaTexture* InMediaTexture)
{
	if (!SceneViewExtension.IsValid() || !InMediaTexture)
	{
		UE_LOG(LogMicroHandpose, Warning, TEXT("[MicroHandpose] StartTracking requires a valid MediaTexture"));
		return;
	}

	InputMediaTexture = InMediaTexture;
	TextureWaitFrames = 0;

	SceneViewExtension->SetMaxHands(MaxHands);
	SceneViewExtension->SetScoreThreshold(ScoreThreshold);
	SceneViewExtension->SetPalmScoreThreshold(PalmScoreThreshold);

	// Initialize detector on the render thread (loads weight buffers), then enable tracking
	FString PluginBaseDir = IPluginManager::Get().FindPlugin(TEXT("MicroHandpose"))->GetBaseDir();
	TSharedPtr<FHandposeSceneViewExtension, ESPMode::ThreadSafe> SVE = SceneViewExtension;
	ENQUEUE_RENDER_COMMAND(HandposeInit)(
		[SVE, PluginBaseDir](FRHICommandListImmediate& RHICmdList)
		{
			if (SVE.IsValid())
			{
				if (SVE->InitializeDetector(PluginBaseDir))
				{
					SVE->SetEnabled(true);
					UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Detector initialized, tracking enabled"));
				}
				else
				{
					UE_LOG(LogMicroHandpose, Error, TEXT("[MicroHandpose] Failed to initialize detector"));
				}
			}
		});

	bIsTracking = true;

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Tracking started (MaxHands=%d, Threshold=%.2f)"), MaxHands, ScoreThreshold);
}

void UHandposeSubsystem::StopTracking()
{
	if (SceneViewExtension.IsValid())
	{
		SceneViewExtension->SetEnabled(false);
		SceneViewExtension->SetInputTextureRHI(nullptr);
		SceneViewExtension->SetInputTextureResource(nullptr);
	}

	InputMediaTexture = nullptr;
	bIsTracking = false;

	FScopeLock Lock(&ResultGuard);
	LatestResults.Empty();

	UE_LOG(LogMicroHandpose, Log, TEXT("[MicroHandpose] Tracking stopped"));
}

TArray<FHandposeResult> UHandposeSubsystem::GetLatestResults()
{
	FScopeLock Lock(&ResultGuard);
	return LatestResults;
}

void UHandposeSubsystem::SetMaxHands(int32 InMaxHands)
{
	MaxHands = FMath::Clamp(InMaxHands, 1, 3);
	if (SceneViewExtension.IsValid())
	{
		SceneViewExtension->SetMaxHands(MaxHands);
	}
}

void UHandposeSubsystem::SetScoreThreshold(float Threshold)
{
	ScoreThreshold = FMath::Clamp(Threshold, 0.0f, 1.0f);
	if (SceneViewExtension.IsValid())
	{
		SceneViewExtension->SetScoreThreshold(ScoreThreshold);
	}
}

void UHandposeSubsystem::SetPalmScoreThreshold(float Threshold)
{
	PalmScoreThreshold = FMath::Clamp(Threshold, 0.0f, 1.0f);
	if (SceneViewExtension.IsValid())
	{
		SceneViewExtension->SetPalmScoreThreshold(PalmScoreThreshold);
	}
}
