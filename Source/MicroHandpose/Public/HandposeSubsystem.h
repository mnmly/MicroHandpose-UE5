#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "HandposeTypes.h"
#include "HandposeSubsystem.generated.h"

class FHandposeSceneViewExtension;
class UMediaPlayer;
class UMediaTexture;

UCLASS()
class MICROHANDPOSE_API UHandposeSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()

private:
	TSharedPtr<FHandposeSceneViewExtension, ESPMode::ThreadSafe> SceneViewExtension;

	/** Latest results from the render thread */
	TArray<FHandposeResult> LatestResults;
	FCriticalSection ResultGuard;

	bool bIsTracking = false;
	int32 MaxHands = 2;
	float ScoreThreshold = 0.5f;
	float PalmScoreThreshold = 0.5f;

	FDelegateHandle BeginFrameHandle;
	FDelegateHandle WorldCleanupHandle;

	/** The media texture used as input (kept alive by UPROPERTY) */
	UPROPERTY()
	TObjectPtr<UMediaTexture> InputMediaTexture = nullptr;

	/** The media player for camera capture */
	UPROPERTY()
	TObjectPtr<UMediaPlayer> CameraPlayer = nullptr;

	int32 TextureWaitFrames = 0;

	void OnBeginFrame();
	void OnWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources);

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	UFUNCTION(BlueprintPure, Category = "Handpose", meta = (DisplayName = "Get Handpose Subsystem"))
	static UHandposeSubsystem* GetInstance();

	/** Start hand tracking with a MediaTexture as input (e.g. from a webcam MediaPlayer). */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	void StartTracking(UMediaTexture* InMediaTexture);

	/** Stop hand tracking and release resources. */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	void StopTracking();

	/** Get the latest hand pose results. */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	TArray<FHandposeResult> GetLatestResults();

	/** Set maximum number of hands to detect (1-3). */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	void SetMaxHands(int32 InMaxHands);

	/** Set minimum confidence threshold for hand detection (0-1). */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	void SetScoreThreshold(float Threshold);

	/** Set minimum confidence threshold for palm detection (0-1). */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	void SetPalmScoreThreshold(float Threshold);

	/** Whether hand tracking is currently active. */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	bool IsTracking() const { return bIsTracking; }

	/** Fired each frame when hand results are updated. */
	UPROPERTY(BlueprintAssignable, Category = "Handpose")
	FOnHandposeUpdated OnHandposeUpdated;
};
