#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include "Image.hpp"

class Renderer {
public:
    Renderer(uint32_t width, uint32_t height);
    ~Renderer();

    Image render();

private:
    uint32_t m_width, m_height;
    uint32_t m_maxRecursion;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice         m_device         = VK_NULL_HANDLE;
    uint32_t         m_queueFamily    = 0;
    VkQueue          m_queue          = VK_NULL_HANDLE;
    VkCommandPool    m_commandPool    = VK_NULL_HANDLE;

    // Geometry
    VkBuffer       m_vertexBuffer  = VK_NULL_HANDLE;
    VkDeviceMemory m_vertexMemory  = VK_NULL_HANDLE;
    VkBuffer       m_indexBuffer   = VK_NULL_HANDLE;
    VkDeviceMemory m_indexMemory   = VK_NULL_HANDLE;

    // Acceleration structures
    VkAccelerationStructureKHR m_blas = VK_NULL_HANDLE;
    VkBuffer                   m_blasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory             m_blasMemory = VK_NULL_HANDLE;

    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
    VkBuffer                   m_tlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory             m_tlasMemory = VK_NULL_HANDLE;

    VkBuffer       m_instanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_instanceMemory = VK_NULL_HANDLE;

    // Storage image (ray tracing output)
    VkImage        m_storageImage       = VK_NULL_HANDLE;
    VkDeviceMemory m_storageImageMemory = VK_NULL_HANDLE;
    VkImageView    m_storageImageView   = VK_NULL_HANDLE;

    // Descriptors
    VkDescriptorSetLayout m_descLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_descSet    = VK_NULL_HANDLE;

    // Pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_rtPipeline     = VK_NULL_HANDLE;

    // Shader binding table
    VkBuffer       m_sbtBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_sbtMemory = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR m_rgenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callRegion{};

    // Readback buffer
    VkBuffer       m_outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_outputMemory = VK_NULL_HANDLE;

    // KHR function pointers
    PFN_vkCreateAccelerationStructureKHR          pfn_vkCreateAccelerationStructureKHR          = nullptr;
    PFN_vkDestroyAccelerationStructureKHR         pfn_vkDestroyAccelerationStructureKHR         = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR   pfn_vkGetAccelerationStructureBuildSizesKHR   = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR       pfn_vkCmdBuildAccelerationStructuresKHR       = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR            pfn_vkCreateRayTracingPipelinesKHR            = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR      pfn_vkGetRayTracingShaderGroupHandlesKHR      = nullptr;
    PFN_vkCmdTraceRaysKHR                         pfn_vkCmdTraceRaysKHR                         = nullptr;

    void createInstance();
    void pickPhysicalDevice();
    void createDevice();
    void loadFunctionPointers();
    void createCommandPool();
    void createGeometry();
    void buildAccelerationStructures();
    void createStorageImage();
    void createDescriptors();
    void createRTPipeline();
    void createSBT();
    void traceRays();
    Image readback();

    uint32_t       findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);
    void           createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer& buf, VkDeviceMemory& mem,
                                bool deviceAddress = false);
    VkDeviceAddress getBufferAddress(VkBuffer buf);
    VkDeviceAddress getASAddress(VkAccelerationStructureKHR as);
    VkCommandBuffer beginCmd();
    void            endCmd(VkCommandBuffer cmd);
    VkShaderModule  loadSPV(const std::string& path);
};