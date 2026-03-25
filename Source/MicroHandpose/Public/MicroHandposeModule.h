#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

MICROHANDPOSE_API DECLARE_LOG_CATEGORY_EXTERN(LogMicroHandpose, Log, All);

class FMicroHandposeModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
