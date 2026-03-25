#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HandposeTypes.h"
#include "HandposeComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHandDetected, const FHandposeResult&, Result);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnHandLost);

UCLASS(ClassGroup=(Input), meta=(BlueprintSpawnableComponent))
class MICROHANDPOSE_API UHandposeComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UHandposeComponent();

	/** Current hand results for this component */
	UPROPERTY(BlueprintReadOnly, Category = "Handpose")
	TArray<FHandposeResult> HandResults;

	/** Which hand index this component tracks (0 = first detected hand) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Handpose")
	int32 TrackedHandIndex = 0;

	/** Fired when a hand is detected */
	UPROPERTY(BlueprintAssignable, Category = "Handpose")
	FOnHandDetected OnHandDetected;

	/** Fired when a tracked hand is lost */
	UPROPERTY(BlueprintAssignable, Category = "Handpose")
	FOnHandLost OnHandLost;

	/** Get the landmark position for a specific joint. Returns ZeroVector if not tracked. */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	FVector GetLandmarkPosition(EHandLandmark Landmark) const;

	/** Whether this component currently has a valid tracked hand */
	UFUNCTION(BlueprintCallable, Category = "Handpose")
	bool HasTrackedHand() const;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	bool bWasTracking = false;
};
