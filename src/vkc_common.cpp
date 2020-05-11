#include "vkc/vkc.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <chrono>
#include <map>
#include <queue>
#include <algorithm>
#include <set>
#include <cmath>

/** global variables (only for this implementation file. **/
/** FIXME: Make it configurable */
const std::string MODEL_PATH = "res/models/mythical-beast.obj";
const std::string TEXTURE_PATH = "res/textures/Lev-edinorog_complete_0.png";
const float FPS_GRANULARITY_SEC = 1.0f; // how often to update FPS
char title[50];

struct VkcInternal
{
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> imagesInFlight;
    vkc::QueueFamilyIndices queueFamilyIndices;
    size_t currentFrame = 0;
    size_t semaphoreIndex = 0;
    std::string windowTitle;
    bool framebufferResized = false;
    std::vector<vkc::Vertex> modelVertices;
    std::vector<uint32_t> modelIndices;
    std::vector<vkc::Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    std::vector<VkBuffer> uniformBuffers;       // uniform buffer for each swapchain's image
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    bool isNeedStagingBuffer = true;       // APU doesn't need staging buffer for better performance
    uint32_t mipLevels;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;   // equivalent to no multisampling
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    uint32_t numRenderedFrames = 0;
    float fps = 0.0f;
    double prevTime = 0.0f;
};

/** for now just assume there is only once instance of Vulkan system **/
VkcInternal intrn;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

/** --- util functions --- **/

/** TODO: Might need to refactor out **/
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// get access to extension functions
static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

/** --- internal functions --- **/
/** function signatures **/
VKC_INTRN void initWindow(const int width, const int height, const char* title);
VKC_INTRN void initVulkan();
VKC_INTRN void mainLoop();
VKC_INTRN void drawFrame();
VKC_INTRN void cleanup();
VKC_INTRN void createInstance();
VKC_INTRN std::vector<const char*> getRequiredExtensions();
VKC_INTRN VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
VKC_INTRN void framebufferResizeCallback(GLFWwindow* window, int width, int height);
VKC_INTRN void setupDebugMessenger();
VKC_INTRN void createSyncObjects();
VKC_INTRN void createCommandPool();
VKC_INTRN void createTextureImage();
VKC_INTRN void createTextureImageView();
VKC_INTRN void createTextureSampler();
VKC_INTRN void createVertexBuffer();
VKC_INTRN void createCommandBuffers();
VKC_INTRN void createFramebuffers();
VKC_INTRN void createRenderPass();
VKC_INTRN VkShaderModule createShaderModule(const std::vector<char>& code);
VKC_INTRN void createGraphicsPipeline();
VKC_INTRN void createImageViews();
VKC_INTRN void createSwapChain();
VKC_INTRN void createSurface();
VKC_INTRN void createLogicalDevice();
VKC_INTRN void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
VKC_INTRN vkc::QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
VKC_INTRN vkc::SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
VKC_INTRN void pickPhysicalDevice();
VKC_INTRN int rateDevice(VkPhysicalDevice device, bool& isDiscrete);
VKC_INTRN void printDeviceInfo(VkPhysicalDevice& device);
VKC_INTRN std::string getDriverVersionString(uint32_t vendorID, uint32_t driverVersion);
VKC_INTRN std::string getVendorString(uint32_t vendorID);
VKC_INTRN std::string getDeviceTypeString(uint32_t deviceType);
VKC_INTRN bool isDeviceSuitable(VkPhysicalDevice device, const VkPhysicalDeviceFeatures* supportedFeatures);
VKC_INTRN bool checkDeviceExtensionSupport(VkPhysicalDevice device);
VKC_INTRN bool checkAllRequiredExtensionsSupported();
VKC_INTRN VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
VKC_INTRN VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
VKC_INTRN VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
VKC_INTRN bool checkValidationLayerSupport();
VKC_INTRN void recreateSwapChain();
VKC_INTRN void cleanupSwapChain();
VKC_INTRN uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
VKC_INTRN void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
VKC_INTRN void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
VKC_INTRN void createIndexBuffer();
VKC_INTRN void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
VKC_INTRN void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
VKC_INTRN void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
VKC_INTRN void createDescriptorSetLayout();
VKC_INTRN void createUniformBuffers();
VKC_INTRN void updateUniformBuffer(uint32_t currentImage);
VKC_INTRN void createDescriptorPool();
VKC_INTRN void createDescriptorSets();
VKC_INTRN VkCommandBuffer beginSingleTimeCommands();
VKC_INTRN void endSingleTimeCommands(VkCommandBuffer commandBuffer);
VKC_INTRN VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
VKC_INTRN void createDepthResources();
VKC_INTRN VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
VKC_INTRN VkFormat findDepthFormat();
VKC_INTRN bool hasStencilComponent(VkFormat format);
VKC_INTRN void loadModel();
VKC_INTRN void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
VKC_INTRN VkSampleCountFlagBits getMaxUsableSampleCount();
VKC_INTRN void createColorResources();

/** function definition **/
VKC_INTRN void initWindow(const int width, const int height, const char* title)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    intrn.windowTitle = title;    // save window's title for later use
    intrn.window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    glfwSetWindowUserPointer(intrn.window, &intrn);
    glfwSetFramebufferSizeCallback(intrn.window, framebufferResizeCallback);
}

VKC_INTRN void initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    loadModel();
    /* warning: we could better off create a single large buffer holding all sub-buffers and use offset to locate each type of buffer for better efficiency. */
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

VKC_INTRN void mainLoop()
{
    while (!glfwWindowShouldClose(intrn.window)) {
        glfwPollEvents();
        drawFrame();

        ++intrn.numRenderedFrames;
        const double currTime = glfwGetTime();
        const double diffTime = currTime - intrn.prevTime;
        if (diffTime >= FPS_GRANULARITY_SEC) {
            intrn.prevTime = currTime;
            intrn.fps = intrn.numRenderedFrames / diffTime;
            std::snprintf(title, sizeof(title), "%s: %.2f FPS", intrn.windowTitle.c_str(), intrn.fps);
            glfwSetWindowTitle(intrn.window, title);
            intrn.numRenderedFrames = 0;
        }
    }

    vkDeviceWaitIdle(intrn.device);
}

VKC_INTRN void drawFrame()
{
    intrn.semaphoreIndex = (intrn.semaphoreIndex + 1) % intrn.imageAvailableSemaphores.size();
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(intrn.device, intrn.swapChain, UINT64_MAX, intrn.imageAvailableSemaphores[intrn.semaphoreIndex], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("failed to acquire swap chain image!");

    vkWaitForFences(intrn.device, 1, &intrn.imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    vkResetFences(intrn.device, 1, &intrn.imagesInFlight[imageIndex]);

    updateUniformBuffer(imageIndex);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // - setup semaphore to wait before submitting the command buffer
    VkSemaphore waitSemaphores[] = { intrn.imageAvailableSemaphores[intrn.semaphoreIndex] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &intrn.commandBuffers[imageIndex];   // hook it up with already recoreded command buffer

    // - setup semaphore to signal when the command buffer done their job
    VkSemaphore signalSemaphores[] = { intrn.renderFinishedSemaphores[intrn.semaphoreIndex] };
    
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(intrn.graphicsQueue, 1, &submitInfo, intrn.imagesInFlight[imageIndex]) != VK_SUCCESS)
        throw std::runtime_error("failed to submit draw command buffer!");

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores; // wait until render finished

    VkSwapchainKHR swapChains[] = { intrn.swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    result = vkQueuePresentKHR(intrn.presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || intrn.framebufferResized) {
        recreateSwapChain();
        intrn.framebufferResized = false;
    }
    else if (result != VK_SUCCESS)
        throw std::runtime_error("failed to present swap chain image!");
}

VKC_INTRN void cleanup()
{
    cleanupSwapChain();

    vkDestroySampler(intrn.device, intrn.textureSampler, nullptr);
    vkDestroyImageView(intrn.device, intrn.textureImageView, nullptr);
    vkDestroyImage(intrn.device, intrn.textureImage, nullptr);
    vkFreeMemory(intrn.device, intrn.textureImageMemory, nullptr);

    vkDestroyDescriptorSetLayout(intrn.device, intrn.descriptorSetLayout, nullptr);

    vkDestroyBuffer(intrn.device, intrn.indexBuffer, nullptr);
    vkFreeMemory(intrn.device, intrn.indexBufferMemory, nullptr);

    vkDestroyBuffer(intrn.device, intrn.vertexBuffer, nullptr);
    vkFreeMemory(intrn.device, intrn.vertexBufferMemory, nullptr);

    for (size_t i=0; i<intrn.swapChainImages.size(); ++i) {
        vkDestroySemaphore(intrn.device, intrn.imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(intrn.device, intrn.renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(intrn.device, intrn.imagesInFlight[i], nullptr);
    }
    vkDestroyCommandPool(intrn.device, intrn.commandPool, nullptr);
    vkDestroyDevice(intrn.device, nullptr);
#ifdef ENABLE_VALIDATION_LAYERS
    DestroyDebugUtilsMessengerEXT(intrn.instance, intrn.debugMessenger, nullptr);
#endif

    vkDestroySurfaceKHR(intrn.instance, intrn.surface, nullptr);
    vkDestroyInstance(intrn.instance, nullptr);

    glfwDestroyWindow(intrn.window);
    glfwTerminate();
}

VKC_INTRN void createInstance()
{
#ifdef ENABLE_VALIDATION_LAYERS
    if (!checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested, but not available!");
#endif

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

#ifdef ENABLE_VALIDATION_LAYERS
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    populateDebugMessengerCreateInfo(debugCreateInfo);

    // setup debug callback as prior to creating a vulkan instance
    createInfo.pNext = static_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&debugCreateInfo);
#else
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;
#endif

    if (vkCreateInstance(&createInfo, nullptr, &intrn.instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance!");
}

VKC_INTRN std::vector<const char*> getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions;
    for (uint32_t i=0; i<glfwExtensionCount; ++i) {
        extensions.push_back(glfwExtensions[i]);
    }
#ifdef ENABLE_VALIDATION_LAYERS
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return extensions;
}

VKC_INTRN VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

VKC_INTRN void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto instance = reinterpret_cast<VkcInternal*>(glfwGetWindowUserPointer(window));
    instance->framebufferResized = true;
}

VKC_INTRN void setupDebugMessenger()
{
#ifdef ENABLE_VALIDATION_LAYERS
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(intrn.instance, &createInfo, nullptr, &intrn.debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger!");
#endif
}

VKC_INTRN void createSyncObjects()
{
    intrn.imageAvailableSemaphores.resize(intrn.swapChainImages.size());
    intrn.renderFinishedSemaphores.resize(intrn.swapChainImages.size());
    intrn.imagesInFlight.resize(intrn.swapChainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i=0; i<intrn.swapChainImages.size(); ++i) {
        if (vkCreateSemaphore(intrn.device, &semaphoreInfo, nullptr, &intrn.imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(intrn.device, &semaphoreInfo, nullptr, &intrn.renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(intrn.device, &fenceInfo, nullptr, &intrn.imagesInFlight[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects from a frame!");
        }
    }
}

VKC_INTRN void createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = intrn.queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0;

    if (vkCreateCommandPool(intrn.device, &poolInfo, nullptr, &intrn.commandPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool!");
}

VKC_INTRN void createTextureImage()
{
    int texWidth;
    int texHeight;
    int texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    if (!pixels)
        throw std::runtime_error("failed to load texture image!");

    intrn.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    if (intrn.isNeedStagingBuffer) {
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(intrn.device, stagingBufferMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(intrn.device, stagingBufferMemory);
    }
    else {
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(intrn.device, stagingBufferMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(intrn.device, stagingBufferMemory);
    }

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, intrn.mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, intrn.textureImage, intrn.textureImageMemory);
    transitionImageLayout(intrn.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, intrn.mipLevels);
    copyBufferToImage(stagingBuffer, intrn.textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    // if enable generating mipmap, comment transitionImageLayout() line, otherwise
    // comment another line
    //transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);
    generateMipmaps(intrn.textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, intrn.mipLevels);

    vkDestroyBuffer(intrn.device, stagingBuffer, nullptr);
    vkFreeMemory(intrn.device, stagingBufferMemory, nullptr);
}

VKC_INTRN void createTextureImageView()
{
    intrn.textureImageView = createImageView(intrn.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, intrn.mipLevels);
}

VKC_INTRN void createTextureSampler()
{
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(intrn.mipLevels);

    if (vkCreateSampler(intrn.device, &samplerInfo, nullptr, &intrn.textureSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture sampler!");
}

VKC_INTRN void createVertexBuffer()
{
    VkDeviceSize bufferSize = sizeof(intrn.vertices[0]) * intrn.vertices.size();

    if (intrn.isNeedStagingBuffer) {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(intrn.device, stagingBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, intrn.vertices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(intrn.device, stagingBufferMemory);

        // non-mappable buffer now
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, intrn.vertexBuffer, intrn.vertexBufferMemory);
        copyBuffer(stagingBuffer, intrn.vertexBuffer, bufferSize);
        vkDestroyBuffer(intrn.device, stagingBuffer, nullptr);
        vkFreeMemory(intrn.device, stagingBufferMemory, nullptr);
    }
    else {
        createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, intrn.vertexBuffer, intrn.vertexBufferMemory);

        void * data;
        vkMapMemory(intrn.device, intrn.vertexBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, intrn.vertices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(intrn.device, intrn.vertexBufferMemory);
    }
}

VKC_INTRN void createCommandBuffers()
{
    intrn.commandBuffers.resize(intrn.swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = intrn.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(intrn.commandBuffers.size());

    if (vkAllocateCommandBuffers(intrn.device, &allocInfo, intrn.commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers!");

    for (size_t i=0; i<intrn.commandBuffers.size(); ++i) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(intrn.commandBuffers[i], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("failed to begin recording command buffer!");

        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = intrn.renderPass;
        renderPassInfo.framebuffer = intrn.swapChainFramebuffers[i];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = intrn.swapChainExtent;

        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(intrn.commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(intrn.commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, intrn.graphicsPipeline);

        VkBuffer vertexBuffers[] = {intrn.vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(intrn.commandBuffers[i], 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(intrn.commandBuffers[i], intrn.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(intrn.commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, intrn.pipelineLayout, 0, 1, &intrn.descriptorSets[i], 0, nullptr);
        vkCmdDrawIndexed(intrn.commandBuffers[i], static_cast<uint32_t>(intrn.indices.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(intrn.commandBuffers[i]);

        if (vkEndCommandBuffer(intrn.commandBuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to record command buffer!");
    }
}

VKC_INTRN void createFramebuffers()
{
    intrn.swapChainFramebuffers.resize(intrn.swapChainImageViews.size());

    for (size_t i=0; i<intrn.swapChainImageViews.size(); ++i) {
        // order is important here, specify first attachment that pixels will be written to
        std::array<VkImageView, 3> attachments = {intrn.colorImageView, intrn.depthImageView, intrn.swapChainImageViews[i]};
        
        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = intrn.renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = intrn.swapChainExtent.width;
        framebufferInfo.height = intrn.swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(intrn.device, &framebufferInfo, nullptr, &intrn.swapChainFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create framebuffers!");
    }
}

VKC_INTRN void createRenderPass()
{
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = intrn.swapChainImageFormat;
    colorAttachment.samples = intrn.msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = intrn.msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool cachedHasStencil = hasStencilComponent(depthAttachment.format);
    if (cachedHasStencil)
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    else
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve = {};
    colorAttachmentResolve.format = intrn.swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // specify the layout that attachment uses during subpass
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    if (cachedHasStencil)
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    else
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef = {};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // create all subpass (just one in this case)
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // create renderpass
    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(intrn.device, &renderPassInfo, nullptr, &intrn.renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create render pass!");
}

VKC_INTRN VkShaderModule createShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(intrn.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module!");

    return shaderModule;
}

VKC_INTRN void createGraphicsPipeline()
{
    /** FIXME: make it configurable */
    auto vertShaderCode = readFile("res/shaders/vert.spv");
    auto fragShaderCode = readFile("res/shaders/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    // create fixed-function pipeline
    // - Vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    auto vertexBindingDescription = vkc::Vertex::getBindingDescription();
    auto vertexAttributeDescriptions = vkc::Vertex::getAttributeDescriptions();

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &vertexBindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();

    // - Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // - Viewport and Scissors
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(intrn.swapChainExtent.width);
    viewport.height = static_cast<float>(intrn.swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = intrn.swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // - Rasterizers
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // - Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.rasterizationSamples = intrn.msaaSamples;
    multisampling.minSampleShading = 0.2f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // - Depth testing
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};
    depthStencil.back = {};

    // - Color blending (disabled for now)
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    /** FIXME: Remove these two blocks of code ? */
    // configure dynamic states to avoid re-creation of pipeline but we need to specify runtime data at drawing time
    /**VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };*/

    /** isn't used anymore */
    /*VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 1;
    dynamicState.pDynamicStates = dynamicStates;*/

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &intrn.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(intrn.device, &pipelineLayoutInfo, nullptr, &intrn.pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = intrn.pipelineLayout;
    pipelineInfo.renderPass = intrn.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(intrn.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &intrn.graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    vkDestroyShaderModule(intrn.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(intrn.device, vertShaderModule, nullptr);
}

VKC_INTRN void createImageViews()
{
    intrn.swapChainImageViews.resize(intrn.swapChainImages.size());

    for (size_t i=0; i < intrn.swapChainImages.size(); ++i) {
        intrn.swapChainImageViews[i] = createImageView(intrn.swapChainImages[i], intrn.swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
}

VKC_INTRN void createSwapChain()
{
    vkc::SwapChainSupportDetails swapChainSupport = querySwapChainSupport(intrn.physicalDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = 3;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = intrn.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t indices[] = { intrn.queueFamilyIndices.graphicsFamily.value(), intrn.queueFamilyIndices.presentFamily.value() };

    if (intrn.queueFamilyIndices.graphicsFamily != intrn.queueFamilyIndices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = indices;
    }
    else {
        // see imageSharingMode man-page as .queueFamilyIndexCount should be 0
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(intrn.device, &createInfo, nullptr, &intrn.swapChain) != VK_SUCCESS)
        throw std::runtime_error("failed to create swap chain!");

    // populate swapchain's images
    vkGetSwapchainImagesKHR(intrn.device, intrn.swapChain, &imageCount, nullptr);
    intrn.swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(intrn.device, intrn.swapChain, &imageCount, intrn.swapChainImages.data());

    // cache essential values for later use
    intrn.swapChainImageFormat = surfaceFormat.format;
    intrn.swapChainExtent = extent;
}

VKC_INTRN void createSurface()
{
    if (glfwCreateWindowSurface(intrn.instance, intrn.window, nullptr, &intrn.surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");
}

VKC_INTRN void createLogicalDevice()
{
    // find queue families now, and cache it for later use across the codebase
    intrn.queueFamilyIndices = findQueueFamilies(intrn.physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {intrn.queueFamilyIndices.graphicsFamily.value(), intrn.queueFamilyIndices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // enable device's feature we need
    VkPhysicalDeviceFeatures deviceFeatures = {}; 
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;

    // check if device supports separate depth/stencil layouts
    VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures queriedSeparateDepthStencilFeature = {};
    queriedSeparateDepthStencilFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES;

    VkPhysicalDeviceFeatures2 queriedDeviceFeatures2 = {};
    queriedDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    queriedDeviceFeatures2.pNext = &queriedSeparateDepthStencilFeature;
    vkGetPhysicalDeviceFeatures2(intrn.physicalDevice, &queriedDeviceFeatures2);

    std::cout << "Separate Depth/Stencil buffer support: " << (queriedSeparateDepthStencilFeature.separateDepthStencilLayouts?"true":"false") << '\n';

    // structure to enable separate depth/stencil layouts
    // usually use depth/stencil as one layout via VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    queriedSeparateDepthStencilFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES;
    queriedSeparateDepthStencilFeature.separateDepthStencilLayouts = VK_TRUE;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // enable separate depth/stencil layouts especially for this program as we only use depth buffer.
    createInfo.pNext = &queriedSeparateDepthStencilFeature;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

#ifdef ENABLE_VALIDATION_LAYERS
    // TODO: fix deprecations, might need to combine with .ppEnabledExtensionNames
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
#else
    createInfo.enabledLayerCount = 0;
#endif

    if (vkCreateDevice(intrn.physicalDevice, &createInfo, nullptr, &intrn.device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device!");

    vkGetDeviceQueue(intrn.device, intrn.queueFamilyIndices.graphicsFamily.value(), 0, &intrn.graphicsQueue);
    vkGetDeviceQueue(intrn.device, intrn.queueFamilyIndices.presentFamily.value(), 0, &intrn.presentQueue);
}

VKC_INTRN void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    //createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

VKC_INTRN vkc::QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
{
    vkc::QueueFamilyIndices indices = {};

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, intrn.surface, &presentSupport);
        if (presentSupport)
            indices.presentFamily = i;
        
        if (indices.isComplete())
            break;

        ++i;
    }

    return indices;
}

VKC_INTRN vkc::SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
{
    vkc::SwapChainSupportDetails details = {};
    // query for basic capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, intrn.surface, &details.capabilities);

    // query for formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, intrn.surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, intrn.surface, &formatCount, details.formats.data());
    }

    // query for presentation mode
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, intrn.surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, intrn.surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VKC_INTRN void pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(intrn.instance, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(intrn.instance, &deviceCount, devices.data());

    // device scores
    typedef int DeviceScore;
    typedef bool DeviceIsDiscreteGPU;
    std::priority_queue<std::tuple<VkPhysicalDevice, DeviceScore, DeviceIsDiscreteGPU>> pq;

    for (uint32_t i=0; i<devices.size(); ++i) {
        bool isDiscrete;
        int score = rateDevice(devices[i], isDiscrete);
        if (score > 0)
            pq.push(std::make_tuple(devices[i], score, isDiscrete));
    }

    if (pq.size() > 0) 
        // custom selection of GPU can be added here
        intrn.physicalDevice = std::get<0>(pq.top());

    if (intrn.physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU with vulkan support!");

    // set msaa samples
    intrn.msaaSamples = getMaxUsableSampleCount();
#ifndef NDEBUG
    std::cout << "MSAA Samples: " << intrn.msaaSamples << "\n";
#endif

    // check if staging buffer is necessary
    // in case of integrated GPU, this is not necessary
    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(intrn.physicalDevice, &deviceProps);
    if (deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        intrn.isNeedStagingBuffer = false;

    printDeviceInfo(intrn.physicalDevice);
}

VKC_INTRN int rateDevice(VkPhysicalDevice device, bool& isDiscrete)
{
    VkPhysicalDeviceProperties deviceProperties = {};
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures = {};
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    // return early if not suitable to be used
    if (!isDeviceSuitable(device, static_cast<const VkPhysicalDeviceFeatures*>(&deviceFeatures)))
        return 0;

    int score = 0;
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 100;
        isDiscrete = true;
    }
    else {
        score += 30;
        isDiscrete = false;
    }

    return score;
}

VKC_INTRN void printDeviceInfo(VkPhysicalDevice& device)
{
    VkPhysicalDeviceDriverProperties driverProps = {};
    driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driverProps.pNext = nullptr;

    VkPhysicalDeviceProperties2 deviceProps2 = {};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &driverProps;
    vkGetPhysicalDeviceProperties2(device, &deviceProps2);

    VkPhysicalDeviceProperties deviceProps = deviceProps2.properties;

    std::cout << "Device Information\n";
    std::cout << "  API version: " << VK_VERSION_MAJOR(deviceProps.apiVersion) << '.' << VK_VERSION_MINOR(deviceProps.apiVersion) << '.' << VK_VERSION_PATCH(deviceProps.apiVersion) << '\n';
    
    // driverVersion is vendor-specific
    std::cout << "  Driver version: " << getDriverVersionString(deviceProps.vendorID, deviceProps.driverVersion) << '\n';
    std::cout << "  Vendor ID: " << std::hex << deviceProps.vendorID << " [" << getVendorString(deviceProps.vendorID) << "]\n";
    std::cout << "  Device type: " << getDeviceTypeString(deviceProps.deviceType) << '\n';
    std::cout << "  Device name: " << deviceProps.deviceName << '\n';
    std::cout << "  Driver version (string): " << driverProps.driverInfo << '\n';
    std::cout << "  Driver name: " << driverProps.driverName << '\n';
    std::cout << "  Driver info: " << driverProps.driverInfo << '\n';
    std::cout << "  Driver ID: " << std::hex << driverProps.driverID << '\n';
}

VKC_INTRN std::string getDriverVersionString(uint32_t vendorID, uint32_t driverVersion)
{
    switch (vendorID) {
        // AMD
        case 0x1002:
            return std::to_string(VK_VERSION_MAJOR(driverVersion)) + "." + std::to_string(VK_VERSION_MINOR(driverVersion)) + "." + std::to_string(VK_VERSION_PATCH(driverVersion));

        // Nvidia
        case 0x10DE:
            return std::to_string((driverVersion >> 22) & 0x3ff) + "." + std::to_string((driverVersion >> 14) & 0x0ff) + "." + std::to_string((driverVersion >> 6) & 0x0ff) + "." + std::to_string(driverVersion & 0x003f);

        // Intel
        case 0x8086:
            return std::to_string((driverVersion >> 14)) + "." + std::to_string(driverVersion & 0x3fff);

        // Use vulkan convention for the less
        default:
            return std::to_string(VK_VERSION_MAJOR(driverVersion)) + "." + std::to_string(VK_VERSION_MINOR(driverVersion)) + "." + std::to_string(VK_VERSION_PATCH(driverVersion));
    }
}

VKC_INTRN std::string getVendorString(uint32_t vendorID)
{
    switch (vendorID) {
        case 0x1002: return "AMD";
        case 0x1010: return "ImgTec";
        case 0x10DE: return "Nvidia";
        case 0x13B5: return "ARM";
        case 0x5143: return "Qualcomm";
        case 0x8086: return "Intel";
        default: return "Unknown";
    }
}

VKC_INTRN std::string getDeviceTypeString(uint32_t deviceType)
{
    switch (deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            return "Other";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "iGPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "CPU";
        default:
            return "Unknown";
    }
}

VKC_INTRN bool isDeviceSuitable(VkPhysicalDevice device, const VkPhysicalDeviceFeatures* supportedFeatures)
{
    vkc::QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        vkc::SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures;
}

VKC_INTRN bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    // remove to avoid O(N^2) but the size is very small, so should be very similar
    // if at last requiredExtensions is empty, then all is oka as it is all supported
    for (const auto& extension: availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

VKC_INTRN bool checkAllRequiredExtensionsSupported()
{
    // get all extensions required by glfw
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    // get all extensions available from vulkan
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    // O(N^2) checking
    for (uint32_t i=0; i<glfwExtensionCount; ++i) {
        bool thisExtFound = false;
        for (uint32_t j=0; j<extensionCount; ++j) {
            if (std::strcmp(glfwExtensions[i], extensions[j].extensionName) == 0) {
                thisExtFound = true;
                break;
            }
        }
        if (!thisExtFound) {
            std::cerr << glfwExtensions[i] << " is not supported\n";
            return false;
        }
    }

    return true;
}

VKC_INTRN VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VKC_INTRN VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VKC_INTRN VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {
        // sanity check and choose in range of support, not assume reasonable value of
        // window's resolution
        int width;
        int height;
        glfwGetFramebufferSize(intrn.window, &width, &height);

        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

VKC_INTRN bool checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    // O(N^2) checking
    for (const char* layerName : validationLayers) {
        bool thisLayerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (std::strcmp(layerName, layerProperties.layerName) == 0) {
                thisLayerFound = true;
                break;
            }
        }
        if (!thisLayerFound) {
            std::cerr << "layer " << layerName << " is not supported!\n";
            return false;
        }
    }

    return true;
}

VKC_INTRN void recreateSwapChain()
{
    vkDeviceWaitIdle(intrn.device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

VKC_INTRN void cleanupSwapChain()
{
    vkDestroyImageView(intrn.device, intrn.colorImageView, nullptr);
    vkDestroyImage(intrn.device, intrn.colorImage, nullptr);
    vkFreeMemory(intrn.device, intrn.colorImageMemory, nullptr);

    vkDestroyImageView(intrn.device, intrn.depthImageView, nullptr);
    vkDestroyImage(intrn.device, intrn.depthImage, nullptr);
    vkFreeMemory(intrn.device, intrn.depthImageMemory, nullptr);

    for (size_t i=0; i<intrn.swapChainFramebuffers.size(); ++i)
        vkDestroyFramebuffer(intrn.device, intrn.swapChainFramebuffers[i], nullptr);

    vkFreeCommandBuffers(intrn.device, intrn.commandPool, static_cast<uint32_t>(intrn.commandBuffers.size()), intrn.commandBuffers.data());

    vkDestroyPipeline(intrn.device, intrn.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(intrn.device, intrn.pipelineLayout, nullptr);
    vkDestroyRenderPass(intrn.device, intrn.renderPass, nullptr);

    for (size_t i=0; i<intrn.swapChainImageViews.size(); ++i)
        vkDestroyImageView(intrn.device, intrn.swapChainImageViews[i], nullptr);

    for (size_t i=0; i<intrn.swapChainImages.size(); ++i) {
        vkDestroyBuffer(intrn.device, intrn.uniformBuffers[i], nullptr);
        vkFreeMemory(intrn.device, intrn.uniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(intrn.device, intrn.descriptorPool, nullptr);
    vkDestroySwapchainKHR(intrn.device, intrn.swapChain, nullptr);
}

VKC_INTRN uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(intrn.physicalDevice, &memProperties);

    for (uint32_t i=0; i<memProperties.memoryTypeCount; ++i) {
        // bitshift curiosity see man page of VkPhysicalDeviceMemoryProperties
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

VKC_INTRN void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(intrn.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create vertex buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(intrn.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT for visible to host and able to map
    // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT for no need to manually invalidate ranges to make it visible for device->host or host->device
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(intrn.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate vertex buffer memory!");

    vkBindBufferMemory(intrn.device, buffer, bufferMemory, 0);   
}

VKC_INTRN void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(intrn.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("failed to create image!");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(intrn.device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(intrn.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate image memory!");

    vkBindImageMemory(intrn.device, image, imageMemory, 0);
}

VKC_INTRN void createIndexBuffer()
{
    VkDeviceSize bufferSize = sizeof(intrn.indices[0]) * intrn.indices.size();

    if (intrn.isNeedStagingBuffer) {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(intrn.device, stagingBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, intrn.indices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(intrn.device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, intrn.indexBuffer, intrn.indexBufferMemory);
        copyBuffer(stagingBuffer, intrn.indexBuffer, bufferSize);
        vkDestroyBuffer(intrn.device, stagingBuffer, nullptr);
        vkFreeMemory(intrn.device, stagingBufferMemory, nullptr);
    }
    else {
        createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, intrn.indexBuffer, intrn.indexBufferMemory);

        void* data;
        vkMapMemory(intrn.device, intrn.indexBufferMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, intrn.indices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(intrn.device, intrn.indexBufferMemory);
    }
}

VKC_INTRN void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

VKC_INTRN void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

VKC_INTRN void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // normal image, or depth + (optional) stencil
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (hasStencilComponent(format))
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    else
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    // set src/dst stage masks
    VkPipelineStageFlags srcPipelineStage;
    VkPipelineStageFlags dstPipelineStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        // before transfer
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        srcPipelineStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstPipelineStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        // after transfer
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        srcPipelineStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstPipelineStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
              newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        srcPipelineStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstPipelineStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else
        throw std::runtime_error("unsupported layout transition!");

    vkCmdPipelineBarrier(commandBuffer, srcPipelineStage, dstPipelineStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
}

VKC_INTRN void createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(intrn.device, &layoutInfo, nullptr, &intrn.descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout!");
}

VKC_INTRN void createUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    intrn.uniformBuffers.resize(intrn.swapChainImages.size());
    intrn.uniformBuffersMemory.resize(intrn.swapChainImages.size());

    for (size_t i=0; i<intrn.swapChainImages.size(); ++i) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, intrn.uniformBuffers[i], intrn.uniformBuffersMemory[i]);

        /*
         * initially set view, and projection matrix
         * both are infrequent update.
         */
        UniformBufferObject ubo = {};
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), intrn.swapChainExtent.width / static_cast<float>(intrn.swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        void* data;
        vkMapMemory(intrn.device, intrn.uniformBuffersMemory[i], 0, sizeof(ubo), 0, &data);
        std::memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(intrn.device, intrn.uniformBuffersMemory[i]);
    }
}

VKC_INTRN void updateUniformBuffer(uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // update only what necessary
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    void* data;
    vkMapMemory(intrn.device, intrn.uniformBuffersMemory[currentImage], 0, sizeof(model), 0, &data);
    std::memcpy(data, &model, sizeof(model));
    vkUnmapMemory(intrn.device, intrn.uniformBuffersMemory[currentImage]);
    // tech note: better perf is to use push constants instead
}

VKC_INTRN void createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(intrn.swapChainImages.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(intrn.swapChainImages.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(intrn.swapChainImages.size());

    if (vkCreateDescriptorPool(intrn.device, &poolInfo, nullptr, &intrn.descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}

VKC_INTRN void createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(intrn.swapChainImages.size(), intrn.descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = intrn.descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(intrn.swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    intrn.descriptorSets.resize(intrn.swapChainImages.size());
    if (vkAllocateDescriptorSets(intrn.device, &allocInfo, intrn.descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    for (size_t i=0; i<intrn.swapChainImages.size(); ++i) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = intrn.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = intrn.textureImageView;
        imageInfo.sampler = intrn.textureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = intrn.descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        descriptorWrites[0].pImageInfo = nullptr;
        descriptorWrites[0].pTexelBufferView = nullptr;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = intrn.descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = nullptr;
        descriptorWrites[1].pImageInfo = &imageInfo;
        descriptorWrites[1].pTexelBufferView = nullptr;

        vkUpdateDescriptorSets(intrn.device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

VKC_INTRN VkCommandBuffer beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = intrn.commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(intrn.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;

}

VKC_INTRN void endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(intrn.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(intrn.graphicsQueue);     // TODO: review this as barrier might be possible?
    vkFreeCommandBuffers(intrn.device, intrn.commandPool, 1, &commandBuffer);
}

VKC_INTRN VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
{
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    VkImageView imageView;

    if (vkCreateImageView(intrn.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture image view!");

    return imageView;

}

VKC_INTRN void createDepthResources()
{
    VkFormat depthFormat = findDepthFormat(); 

    createImage(intrn.swapChainExtent.width, intrn.swapChainExtent.height, 1, intrn.msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, intrn.depthImage, intrn.depthImageMemory);
    intrn.depthImageView = createImageView(intrn.depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    transitionImageLayout(intrn.depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

VKC_INTRN VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(intrn.physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VKC_INTRN VkFormat findDepthFormat()
{
    return findSupportedFormat(
            {
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT
            },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VKC_INTRN bool hasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VKC_INTRN void loadModel()
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
        throw std::runtime_error(warn + err);

    std::unordered_map<vkc::Vertex, uint32_t> uniqueVertices;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            vkc::Vertex vertex = {};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = {1.0f, 1.0f, 1.0f};

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(intrn.vertices.size());    // value for setting at indices later
                intrn.vertices.push_back(vertex);
            }

            intrn.indices.push_back(uniqueVertices[vertex]);
        }
    }

#ifndef NDEBUG
    std::cout << "size of processed vertices: " << intrn.vertices.size() << " from " << std::dec << attrib.vertices.size() << "\n";
#endif
}

VKC_INTRN void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
{
    // check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(intrn.physicalDevice, imageFormat, &formatProperties);
    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        throw std::runtime_error("texture image format does not support linear blitting!");

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

        VkImageBlit blit = {};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth >> 1:1, mipHeight > 1 ? mipHeight >> 1:1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        // blit into itself but different mip level
        vkCmdBlitImage(commandBuffer,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

        if (mipWidth > 1) mipWidth >>= 1;
        if (mipHeight > 1) mipHeight >>= 1;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

VKC_INTRN VkSampleCountFlagBits getMaxUsableSampleCount()
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(intrn.physicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
    if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
    if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
    if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
    if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
    if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;

    return VK_SAMPLE_COUNT_1_BIT;
}

VKC_INTRN void createColorResources()
{
    VkFormat colorFormat = intrn.swapChainImageFormat;

    createImage(intrn.swapChainExtent.width, intrn.swapChainExtent.height, 1, intrn.msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, intrn.colorImage, intrn.colorImageMemory);
    intrn.colorImageView = createImageView(intrn.colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}

/** --- end of internal functions --- **/

void vkc::init(const int initial_screen_width, const int initial_screen_height, const char* title)
{
    initWindow(initial_screen_width, initial_screen_height, title);
    initVulkan();
}

void vkc::loopRun()
{
    mainLoop();
    cleanup();
}
