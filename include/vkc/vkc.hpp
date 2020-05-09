#pragma once

#include "vkc/libcommon.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vkc/vkc_types.hpp"
#include "vkc/vkc_internal_types.hpp"

// define to enable validation layers
#ifndef NDEBUG
#define ENABLE_VALIDATION_LAYERS
#endif

namespace vkc
{
    /**
     * Initialize Vulkan-based graphics.
     */
    VKC_API void init(const int initial_screen_width, const int initial_screen_height, const char* title);

    /**
     * Run the application.
     * This will never return unless user clicks on close button, or programmatically call a certain function to break the loop.
     */
    VKC_API void loopRun();    
}
