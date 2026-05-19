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
		// EMA smoothing — hide jitter between detector cycles. Match incoming
		// hands to prior-frame hands by handedness so a left/right swap doesn't
		// blend two different hands together. New hands enter at their raw
		// position (no warm-up jump).
		if (SmoothingAlpha < 1.0f)
		{
			const float A = SmoothingAlpha;
			const float OneMinusA = 1.0f - A;
			TArray<FHandposeResult> Next;
			Next.Reserve(LatestResults.Num());

			for (const FHandposeResult& In : LatestResults)
			{
				const FHandposeResult* Prev = SmoothedResults.FindByPredicate(
					[&In](const FHandposeResult& R) { return R.Handedness == In.Handedness; });

				FHandposeResult Out = In;
				if (Prev && Prev->Landmarks.Num() == Out.Landmarks.Num())
				{
					for (int32 i = 0; i < Out.Landmarks.Num(); ++i)
					{
						Out.Landmarks[i].Position = A * In.Landmarks[i].Position
						                          + OneMinusA * Prev->Landmarks[i].Position;
					}
				}
				Next.Add(MoveTemp(Out));
			}

			SmoothedResults = Next;
			LatestResults = SmoothedResults;
		}
		else
		{
			SmoothedResults = LatestResults;
		}

		// Instrumentation: log update rate + how often the wrist position is
		// actually changing (i.e. whether downstream is getting new data or
		// the same cached frame repeatedly).
		{
			static double LastSummaryTime = FPlatformTime::Seconds();
			static int32 BroadcastsThisSecond = 0;
			static int32 ChangedThisSecond = 0;
			static FVector LastWrist(0.f);

			BroadcastsThisSecond++;
			const FVector NewWrist = LatestResults[0].Landmarks.Num() > 0 ? LatestResults[0].Landmarks[0].Position : FVector::ZeroVector;
			if (!NewWrist.Equals(LastWrist, KINDA_SMALL_NUMBER))
			{
				ChangedThisSecond++;
				LastWrist = NewWrist;
			}

			const double Now = FPlatformTime::Seconds();
			if (Now - LastSummaryTime >= 1.0)
			{
				UE_LOG(LogMicroHandpose, Log,
					TEXT("[Subsystem] %d broadcasts/s, %d wrist-changed/s, lastWrist=(%.3f,%.3f,%.3f)"),
					BroadcastsThisSecond, ChangedThisSecond, LastWrist.X, LastWrist.Y, LastWrist.Z);
				BroadcastsThisSecond = 0;
				ChangedThisSecond = 0;
				LastSummaryTime = Now;
			}
		}

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
	SmoothedResults.Empty();

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
