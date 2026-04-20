#include "Renderer.hpp"
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <vector>
#include <iostream>

#define VK_CHECK(x)                                                         \
    do { VkResult _r = (x);                                                 \
         if (_r != VK_SUCCESS)                                              \
             throw std::runtime_error("Vulkan error in " #x); } while (0)

static inline uint32_t alignUp(uint32_t v, uint32_t a) {
    return (v + a - 1) & ~(a - 1);
}

// ─────────────────────────────────────────────────────────────────────────────

Renderer::Renderer(uint32_t width, uint32_t height)
    : m_width(width), m_height(height)
{
    createInstance();
    pickPhysicalDevice();
    createDevice();
    loadFunctionPointers();
    createCommandPool();
    createGeometry();
    buildAccelerationStructures();
    createStorageImage();
    createDescriptors();
    createRTPipeline();
    createSBT();
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(m_device);

    vkDestroyBuffer(m_device, m_outputBuffer, nullptr);
    vkFreeMemory   (m_device, m_outputMemory, nullptr);

    vkDestroyBuffer(m_device, m_sbtBuffer, nullptr);
    vkFreeMemory   (m_device, m_sbtMemory, nullptr);

    vkDestroyPipeline      (m_device, m_rtPipeline,     nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

    vkDestroyDescriptorPool     (m_device, m_descPool,   nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descLayout, nullptr);

    vkDestroyImageView(m_device, m_storageImageView,   nullptr);
    vkDestroyImage    (m_device, m_storageImage,       nullptr);
    vkFreeMemory      (m_device, m_storageImageMemory, nullptr);

    pfn_vkDestroyAccelerationStructureKHR(m_device, m_tlas, nullptr);
    vkDestroyBuffer(m_device, m_tlasBuffer, nullptr);
    vkFreeMemory   (m_device, m_tlasMemory, nullptr);

    pfn_vkDestroyAccelerationStructureKHR(m_device, m_blas, nullptr);
    vkDestroyBuffer(m_device, m_blasBuffer, nullptr);
    vkFreeMemory   (m_device, m_blasMemory, nullptr);

    vkDestroyBuffer(m_device, m_instanceBuffer, nullptr);
    vkFreeMemory   (m_device, m_instanceMemory, nullptr);

    vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
    vkFreeMemory   (m_device, m_vertexMemory, nullptr);
    vkDestroyBuffer(m_device, m_indexBuffer,  nullptr);
    vkFreeMemory   (m_device, m_indexMemory,  nullptr);

    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    vkDestroyDevice     (m_device, nullptr);
    vkDestroyInstance   (m_instance, nullptr);
}

Image Renderer::render() {
    traceRays();
    return readback();
}

// ─────────────────────────────────────────────────────────────────────────────

void Renderer::createInstance() {
    VkApplicationInfo app{};
    app.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4; // version

    VkInstanceCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));
}

void Renderer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devs.data());

    for (auto& d : devs) {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());

        bool hasRT = false, hasAS = false, hasDHO = false;
        for (auto& e : exts) {
            if (!strcmp(e.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))  hasRT  = true;
            if (!strcmp(e.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) hasAS  = true;
            if (!strcmp(e.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) hasDHO = true;
        }

        if (hasRT && hasAS && hasDHO) {
            m_physicalDevice = d;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
            rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            props2.pNext = &rtProps;
            vkGetPhysicalDeviceProperties2(d, &props2);
            std::cout << "GPU: " << props2.properties.deviceName << "\n";
            std::cout << "Max Ray Recursion Depth: " << rtProps.maxRayRecursionDepth << "\n";
            m_maxRecursion = rtProps.maxRayRecursionDepth;
            return;
        }
    }
    throw std::runtime_error("No GPU with KHR ray tracing support found.");
}

void Renderer::createDevice() {

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, qfs.data());

    m_queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_queueFamily = i;
            break;
        }
    }
    if (m_queueFamily == UINT32_MAX)
        throw std::runtime_error("No compute queue family found.");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    const char* exts[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, // this may be redundant in newer versions of vulkan, it is used to ensure proper floating point operations
    };

    // Feature chain (pNext linked list)
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
    bdaFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeatures.rayTracingPipeline = VK_TRUE;
    rtFeatures.pNext              = &bdaFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;
    asFeatures.pNext                 = &rtFeatures;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &asFeatures;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &features2;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = (uint32_t)std::size(exts);
    dci.ppEnabledExtensionNames = exts;

    VK_CHECK(vkCreateDevice(m_physicalDevice, &dci, nullptr, &m_device));
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);
}

void Renderer::loadFunctionPointers() {
#define LOAD(fn) \
    pfn_##fn = (PFN_##fn)vkGetDeviceProcAddr(m_device, #fn); \
    if (!pfn_##fn) throw std::runtime_error("Failed to load " #fn);
    LOAD(vkCreateAccelerationStructureKHR)
    LOAD(vkDestroyAccelerationStructureKHR)
    LOAD(vkGetAccelerationStructureBuildSizesKHR)
    LOAD(vkCmdBuildAccelerationStructuresKHR)
    LOAD(vkGetAccelerationStructureDeviceAddressKHR)
    LOAD(vkCreateRayTracingPipelinesKHR)
    LOAD(vkGetRayTracingShaderGroupHandlesKHR)
    LOAD(vkCmdTraceRaysKHR)
#undef LOAD
}

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = m_queueFamily;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool));
}

// ─────────────────────────────────────────────────────────────────────────────

// encontrar el tipo de memoria que necesita la imagen, no lo entiengdo muy bien tbh
uint32_t Renderer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable memory type.");
}

// esto simple y llanamente hace allocate de memoria en la GPU para multiprpose: imagen, sbtc, etc..
// concepto: el buffer es local en GPU pero visible en CPU
void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props,
                            VkBuffer& buf, VkDeviceMemory& mem, bool deviceAddress) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = usage;
    VK_CHECK(vkCreateBuffer(m_device, &bci, nullptr, &buf));

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(m_device, buf, &reqs);

    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = deviceAddress ? &flagsInfo : nullptr;
    ai.allocationSize  = reqs.size;
    ai.memoryTypeIndex = findMemoryType(reqs.memoryTypeBits, props);

    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(m_device, buf, mem, 0));
}

// pido la direccoion de memoria en la GPU de un buff
VkDeviceAddress Renderer::getBufferAddress(VkBuffer buf) {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(m_device, &info);
}

// returns the GPU address of an acceleration structur
VkDeviceAddress Renderer::getASAddress(VkAccelerationStructureKHR as) {
    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = as;
    return pfn_vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);
}

// esto se usa para meter en la cosa cosas rapidas, como copiar datos y tal, interrumpe cola y luego se sale rapidamente.
VkCommandBuffer Renderer::beginCmd() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

// mismo q arriba
void Renderer::endCmd(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_queue));
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

// spirv son los binarios compilados de los shaders, esto los carga.
VkShaderModule Renderer::loadSPV(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open shader: " + path);
    auto sz = f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = buf.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());

    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(m_device, &ci, nullptr, &mod));
    return mod;
}

// ─────────────────────────────────────────────────────────────────────────────

void Renderer::createGeometry() {
    const float verts[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    const uint32_t idxs[] = { 0, 1, 2 };

    // estas dos variables determinan el tipo de buffer y memoria, ojo que esto es configurable
    constexpr VkBufferUsageFlags geoUsage =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr VkMemoryPropertyFlags hostProps =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    createBuffer(sizeof(verts), geoUsage, hostProps, m_vertexBuffer, m_vertexMemory, true);
    void* p;
    vkMapMemory(m_device, m_vertexMemory, 0, sizeof(verts), 0, &p);
    memcpy(p, verts, sizeof(verts));
    vkUnmapMemory(m_device, m_vertexMemory);

    createBuffer(sizeof(idxs), geoUsage, hostProps, m_indexBuffer, m_indexMemory, true);
    vkMapMemory(m_device, m_indexMemory, 0, sizeof(idxs), 0, &p);
    memcpy(p, idxs, sizeof(idxs));
    vkUnmapMemory(m_device, m_indexMemory);
}

void Renderer::buildAccelerationStructures() {
    // ── BLAS ──────────────────────────────────────────────────────────────────

    /// STEP 1 - CREAR TRIDATA
    // es bueno trabajar con tan sólo triangulos! es mas, es obligatorio para usar todo el potencial de gpu
    // ojo aqui, esto para mmayor numero de mallas y muchos triangulos creo q hat que cambiar el maxVertex,
    VkAccelerationStructureGeometryTrianglesDataKHR triData{};
    triData.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triData.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triData.vertexData.deviceAddress = getBufferAddress(m_vertexBuffer);
    triData.vertexStride  = 3 * sizeof(float);
    triData.maxVertex     = 2;
    triData.indexType     = VK_INDEX_TYPE_UINT32;
    triData.indexData.deviceAddress  = getBufferAddress(m_indexBuffer);

    /// STEP 2 - el TRIDATA, LO METEMOS EN UN MISMO BLAS
    VkAccelerationStructureGeometryKHR blasGeo{};
    blasGeo.sType                        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    blasGeo.geometryType                 = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    blasGeo.geometry.triangles           = triData;
    blasGeo.flags                        = VK_GEOMETRY_OPAQUE_BIT_KHR; // IMPORTANT!AKI LE DECIMOS QUE TODOS LOS TRIANGULOS SON OPACOS

    /// STEP 3 -  CONSTRUIR EL BLAS
    VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
    blasBuild.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    blasBuild.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; // optimize this for ray tracing speed, not build speed
    blasBuild.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blasBuild.geometryCount = 1;
    blasBuild.pGeometries   = &blasGeo;

    // Asks Vulkan how much memory is needed for BLas and buff
    uint32_t primCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
    blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuild, &primCount, &blasSizes);

    // crea el buffer donde vamos a meter el blas
    createBuffer(blasSizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_blasBuffer, m_blasMemory, true);

    // a partir de aqui interoducimos el blas en el buffer de antes
    VkAccelerationStructureCreateInfoKHR blasCI{};
    blasCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blasCI.buffer = m_blasBuffer;
    blasCI.size   = blasSizes.accelerationStructureSize;
    blasCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    pfn_vkCreateAccelerationStructureKHR(m_device, &blasCI, nullptr, &m_blas);

    // Temporary working memory used only while building the BLAS. ni idea la vardad
    VkBuffer blasScratch; VkDeviceMemory blasScratchMem;
    createBuffer(blasSizes.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blasScratch, blasScratchMem, true);

    blasBuild.dstAccelerationStructure  = m_blas;
    blasBuild.scratchData.deviceAddress = getBufferAddress(blasScratch);

    // ── TLAS ──────────────────────────────────────────────────────────────────

    // STEP 1- CREATE ONE INSTANC EOF TLAS LINKED TO EACH BLAS, IN THIS CASE ONLY 1
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0]                    = 1.0f;
    instance.transform.matrix[1][1]                    = 1.0f;
    instance.transform.matrix[2][2]                    = 1.0f;
    instance.instanceCustomIndex                       = 0;
    instance.mask                                      = 0xFF; //This means the instance is visible to rays using matching masks.
    instance.instanceShaderBindingTableRecordOffset = 0;       // Chooses the hit-group offset in the SBT. With one object / one hit group, zero is correct.
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; // el face cull se usa para TAN SOLO randerizar una de las caras del triangulo
    instance.accelerationStructureReference            = getASAddress(m_blas); // linkamos una instancia de tlas con un blas

    // store tlas in memory, same as with blas
    createBuffer(sizeof(instance),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_instanceBuffer, m_instanceMemory, true);

    void* ip;
    vkMapMemory(m_device, m_instanceMemory, 0, sizeof(instance), 0, &ip);
    memcpy(ip, &instance, sizeof(instance));
    vkUnmapMemory(m_device, m_instanceMemory);

    VkAccelerationStructureGeometryInstancesDataKHR instData{};
    instData.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instData.data.deviceAddress = getBufferAddress(m_instanceBuffer);

    VkAccelerationStructureGeometryKHR tlasGeo{};
    tlasGeo.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeo.geometryType             = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeo.geometry.instances       = instData;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
    tlasBuild.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasBuild.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuild.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuild.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuild.geometryCount = 1;
    tlasBuild.pGeometries   = &tlasGeo;

    uint32_t instCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
    tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild, &instCount, &tlasSizes);

    createBuffer(tlasSizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_tlasBuffer, m_tlasMemory, true);

    VkAccelerationStructureCreateInfoKHR tlasCI{};
    tlasCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCI.buffer = m_tlasBuffer;
    tlasCI.size   = tlasSizes.accelerationStructureSize;
    tlasCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    pfn_vkCreateAccelerationStructureKHR(m_device, &tlasCI, nullptr, &m_tlas);

    VkBuffer tlasScratch; VkDeviceMemory tlasScratchMem;
    createBuffer(tlasSizes.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlasScratch, tlasScratchMem, true);

    tlasBuild.dstAccelerationStructure  = m_tlas;
    tlasBuild.scratchData.deviceAddress = getBufferAddress(tlasScratch);

    // ── Build both in one command buffer ──────────────────────────────────────

    // ejecutamos los comando para buildear y inyectar en la memoria de la gpu nuestras estructuras de acc
    // bueno se construyen dentro de la gpu

    VkCommandBuffer cmd = beginCmd();

    // BUILD BLAS
    VkAccelerationStructureBuildRangeInfoKHR blasRange{ 1, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pBlasRange = &blasRange;
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blasBuild, &pBlasRange);

    // SUPER CRUCIAL, ESTO SE USA PARA ASEGURAR QUE EL BLAS SE HA TERMINADO DE CONSTRUIR CUANDO EL TLAS LO LEE!
    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);


    //BUILD TLAS
    VkAccelerationStructureBuildRangeInfoKHR tlasRange{ 1, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasBuild, &pTlasRange);

    endCmd(cmd);

    vkDestroyBuffer(m_device, blasScratch, nullptr);
    vkFreeMemory   (m_device, blasScratchMem, nullptr);
    vkDestroyBuffer(m_device, tlasScratch, nullptr);
    vkFreeMemory   (m_device, tlasScratchMem, nullptr);
}

void Renderer::createStorageImage() {
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R32_SFLOAT;
    ici.extent        = { m_width, m_height, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &m_storageImage));

    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(m_device, m_storageImage, &reqs);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = reqs.size;
    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT means keep it in fast GPU memory memans
    ai.memoryTypeIndex = findMemoryType(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_storageImageMemory));
    VK_CHECK(vkBindImageMemory(m_device, m_storageImage, m_storageImageMemory, 0));

    VkImageViewCreateInfo vci{};
    vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image            = m_storageImage;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = VK_FORMAT_R32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_storageImageView));

    // Transition UNDEFINED → GENERAL
    VkCommandBuffer cmd = beginCmd();

    VkImageMemoryBarrier imgBarrier{};
    imgBarrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBarrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    imgBarrier.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.srcAccessMask    = 0;
    imgBarrier.dstAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.image            = m_storageImage;
    imgBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    endCmd(cmd);
}

// el purpose de esta funcion es que el shader de raygen pueda ver tanto
// la escena (tlas) como la imagen donde hay q guardar
void Renderer::createDescriptors() {
    VkDescriptorSetLayoutBinding bindings[2] = {};

    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR; // That means only raygen can access these descriptors. bueno para ir mas rapido

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2;
    lci.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &lci, nullptr, &m_descLayout));

    VkDescriptorPoolSize poolSizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               1 },
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(m_device, &pci, nullptr, &m_descPool));

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = m_descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &m_descLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_descSet));

    // Write TLAS
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures    = &m_tlas;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext           = &asWrite;
    writes[0].dstSet          = m_descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = m_storageImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = m_descSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &imgInfo;

    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
}

void Renderer::createRTPipeline() {
    const std::string dir = SHADER_DIR;
    VkShaderModule rgenMod = loadSPV(dir + "/raygen.rgen.spv");
    VkShaderModule missMod = loadSPV(dir + "/miss.rmiss.spv");
    VkShaderModule chitMod = loadSPV(dir + "/closesthit.rchit.spv");

    VkPipelineShaderStageCreateInfo stages[3] = {};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR,       rgenMod, "main", nullptr };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_MISS_BIT_KHR,         missMod, "main", nullptr };
    stages[2] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  chitMod, "main", nullptr };

    // Group 0 = raygen (general), Group 1 = miss (general), Group 2 = hit (triangles)
    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {};
    for (auto& g : groups) {
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = VK_SHADER_UNUSED_KHR;
        g.anyHitShader       = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0; // raygen
    groups[1].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1; // miss
    groups[2].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].closestHitShader = 2; // triangle hit

    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &m_descLayout;
    VK_CHECK(vkCreatePipelineLayout(m_device, &plci, nullptr, &m_pipelineLayout));

    VkRayTracingPipelineCreateInfoKHR rtci{};
    rtci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtci.stageCount                   = 3;
    rtci.pStages                      = stages;
    rtci.groupCount                   = 3;
    rtci.pGroups                      = groups;
    rtci.maxPipelineRayRecursionDepth = m_maxRecursion;
    rtci.layout                       = m_pipelineLayout;

    VK_CHECK(pfn_vkCreateRayTracingPipelinesKHR(m_device,
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtci, nullptr, &m_rtPipeline));

    vkDestroyShaderModule(m_device, rgenMod, nullptr);
    vkDestroyShaderModule(m_device, missMod, nullptr);
    vkDestroyShaderModule(m_device, chitMod, nullptr);
}

void Renderer::createSBT() {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 devProps{};
    devProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    devProps.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &devProps);

    const uint32_t handleSize  = rtProps.shaderGroupHandleSize;
    const uint32_t stride      = alignUp(handleSize, rtProps.shaderGroupHandleAlignment);
    const uint32_t baseAlign   = rtProps.shaderGroupBaseAlignment;

    const uint32_t rgenOff = 0;
    const uint32_t missOff = alignUp(stride,         baseAlign);
    const uint32_t hitOff  = alignUp(missOff + stride, baseAlign);
    const uint32_t sbtSize = hitOff + stride;

    createBuffer(sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_sbtBuffer, m_sbtMemory, true);

    std::vector<uint8_t> handles(3 * handleSize);
    VK_CHECK(pfn_vkGetRayTracingShaderGroupHandlesKHR(
        m_device, m_rtPipeline, 0, 3, handles.size(), handles.data()));

    void* mapped;
    vkMapMemory(m_device, m_sbtMemory, 0, sbtSize, 0, &mapped);
    auto* sbt = static_cast<uint8_t*>(mapped);
    memcpy(sbt + rgenOff, handles.data() + 0 * handleSize, handleSize);
    memcpy(sbt + missOff, handles.data() + 1 * handleSize, handleSize);
    memcpy(sbt + hitOff,  handles.data() + 2 * handleSize, handleSize);
    vkUnmapMemory(m_device, m_sbtMemory);

    const VkDeviceAddress base = getBufferAddress(m_sbtBuffer);
    m_rgenRegion = { base + rgenOff, stride, stride };
    m_missRegion = { base + missOff, stride, stride };
    m_hitRegion  = { base + hitOff,  stride, stride };
    m_callRegion = {};
}

void Renderer::traceRays() {
    VkCommandBuffer cmd = beginCmd();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    pfn_vkCmdTraceRaysKHR(cmd,
        &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion,
        m_width, m_height, 1);

    endCmd(cmd);
}

Image Renderer::readback() {
    const VkDeviceSize bufSize = (VkDeviceSize)m_width * m_height * sizeof(float);

    createBuffer(bufSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_outputBuffer, m_outputMemory);

    VkCommandBuffer cmd = beginCmd();

    VkImageMemoryBarrier toSrc{};
    toSrc.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toSrc.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
    toSrc.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
    toSrc.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
    toSrc.image            = m_storageImage;
    toSrc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toSrc);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { m_width, m_height, 1 };

    vkCmdCopyImageToBuffer(cmd, m_storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_outputBuffer, 1, &region);

    endCmd(cmd);

    void* mapped;
    vkMapMemory(m_device, m_outputMemory, 0, bufSize, 0, &mapped);
    Image img(m_width, m_height);
    img.setFromFloat(static_cast<const float*>(mapped));
    vkUnmapMemory(m_device, m_outputMemory);

    return img;
}
