#include <cstdio>
#include <cstdint>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vk_enum_string_helper.h>

#include <glm/glm.hpp>

#include <sys/param.h>

#include "vma.hpp"

const unsigned work_size = 128;
const unsigned out_count = 8192;

static std::string AppName    = "Buddhabrot";
static std::string EngineName = "Buddhabrot";

static std::string ComputeShaderCode = "";

static VkBool32 debugCallback2(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
							   VkDebugUtilsMessageTypeFlagsEXT messageType,
							   const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
							   void* pUserData)
{
	std::string debugLevel;
	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		debugLevel = "\e[1;31m[Vulkan ERROR";
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		debugLevel = "\e[33m[Vulkan WARNING";
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
		debugLevel = "[Vulkan INFO";
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
		debugLevel = "\e[2m[Vulkan DEBUG";
	else
		debugLevel = "\e[1;37;41m[Vulkan FLAG=" + std::to_string(static_cast<uint32_t>(messageSeverity));

	std::cerr << debugLevel << "] " << pCallbackData->pMessage << "\e[0m" << std::endl;
	return VK_FALSE;
}

std::string readFileIntoString(const std::string& filename)
{
	std::stringstream buffer;

	{
		std::ifstream file(filename, std::ios::binary);
		if (!file)
			throw std::runtime_error("Failed to open file: " + filename);

		buffer << file.rdbuf();
	}

	return buffer.str();
}

// 1k = 1000, 1M = 1000000
void siprefix10(double value, double* display, char* prefix) {
	double gr = MIN(MAX((int) floor(log10(value) / 3), -10), 10);
	*display = value / powf(10, 3 * gr);
	*prefix = "qryzafpnum_kMGTPEZYRQ"[((int) gr) + 10];
}

// 1k = 1024, 1M = 1048576
void siprefix2(double value, double* display, char* prefix) {
	double gr = MIN(MAX((int) floor(log2(value) / 10), -10), 10);
	*display = value / powf(2, 10 * gr);
	*prefix = "qryzafpnum_kMGTPEZYRQ"[((int) gr) + 10];
}

int main() {
	ComputeShaderCode = readFileIntoString("shaders/budd.comp.spv");

	// Initialize Vulkan instance
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = AppName.c_str();
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = EngineName.c_str();
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_3;

	VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.pNext = nullptr,
		.flags = 0,
		.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		// | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
		.pfnUserCallback = debugCallback2,
		.pUserData = nullptr
	};

	VkInstanceCreateInfo createInfo {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = &debugMessengerInfo,
		.flags = 0,
		.pApplicationInfo = &appInfo
	};

	const char* debugLayers[] = {"VK_LAYER_KHRONOS_validation"};
	createInfo.enabledLayerCount = 1; // Set this value to enable debug layers
	createInfo.ppEnabledLayerNames = debugLayers;

	const char* extensions[] = {"VK_EXT_debug_utils"};
	createInfo.enabledExtensionCount = 1;
	createInfo.ppEnabledExtensionNames = extensions;

	VkInstance instance;
	VKASSERT(vkCreateInstance(&createInfo, nullptr, &instance));

	// Create physical device and logical device
	uint32_t gpuCount = 0;
	VKASSERT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));

	gpuCount = 1;

	VkPhysicalDevice gpu = nullptr;
	VKASSERT(vkEnumeratePhysicalDevices(instance, &gpuCount, &gpu));

	if (gpuCount == 0) {
		fprintf(stderr, "No GPUs?");
		exit(1);
	}

	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
	{
		vkGetPhysicalDeviceMemoryProperties(gpu, &physicalDeviceMemoryProperties);

		for (unsigned i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
			VkMemoryType& type = physicalDeviceMemoryProperties.memoryTypes[i];
			std::string flagstr = string_VkMemoryPropertyFlags(type.propertyFlags);
			printf("\tMemory Type %u uses heap %u and has flags 0x%04x (%s)\n", i, type.heapIndex, type.propertyFlags, flagstr.c_str());
		}
		for (unsigned i = 0; i < physicalDeviceMemoryProperties.memoryHeapCount; i++) {
			VkMemoryHeap& heap = physicalDeviceMemoryProperties.memoryHeaps[i];
			std::string heapstr = string_VkMemoryHeapFlags(heap.flags);

			double sizeNum;
			char sizeSuf;
			siprefix2(heap.size, &sizeNum, &sizeSuf);

			printf("\tMemory Heap %u has %g%cB and flags 0x%04x (%s)\n", i, sizeNum, sizeSuf, heap.flags, heapstr.c_str());
		}
	}

	VkQueue queue;

	// Initialize queue creation info
	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = 0;
	queueCreateInfo.queueCount = 1; // Number of queues in this family
	float queuePriority = 1.0f;     // Priority of each queue (0-1)
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkPhysicalDeviceFeatures feat {
		.shaderFloat64 = VK_TRUE
	};
	VkDeviceCreateInfo deviceInfo {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queueCreateInfo,
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = nullptr,
		.enabledExtensionCount = 0,
		.ppEnabledExtensionNames = nullptr,
		.pEnabledFeatures = &feat,
	};

	VkDevice device;
	VKASSERT(vkCreateDevice(gpu, &deviceInfo, nullptr, &device));

	// Retrieve the queue from the logical device
	uint32_t queueFamilyIndex = 0; // Replace with the actual queue family index
	vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

	// Create compute pipeline and shader module
	VkShaderModuleCreateInfo shaderInfo {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = ComputeShaderCode.length(),
		.pCode = reinterpret_cast<const uint32_t*>(ComputeShaderCode.c_str())
	};

	VkShaderModule computeShaderModule;
	VKASSERT(vkCreateShaderModule(device, &shaderInfo, nullptr, &computeShaderModule));

	VkPipelineShaderStageCreateInfo stageInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = computeShaderModule,
		.pName = "main"
	};

	VkPipelineCache pipelineCache;
	// Initialize pipeline cache creation info
	VkPipelineCacheCreateInfo pipelineCacheInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.initialDataSize = 0
	};

	VKASSERT(vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache));

	VkDescriptorSetLayoutBinding bindings[2] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.pImmutableSamplers = nullptr
		}
	};

	VkDescriptorSetLayoutCreateInfo descriptor_createInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.bindingCount = 2,
		.pBindings = bindings
	};

	VkDescriptorSetLayout descriptorSetLayout;
	VKASSERT(vkCreateDescriptorSetLayout(device, &descriptor_createInfo, nullptr, &descriptorSetLayout));

	VkDescriptorPoolSize descriptorPoolSize {
		.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1024
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.maxSets = 1024,
		.poolSizeCount = 1,
		.pPoolSizes = &descriptorPoolSize
	};
	VkDescriptorPool descriptorPool;
	VKASSERT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = nullptr,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptorSetLayout
	};
	VkDescriptorSet descriptorSet;
	VKASSERT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

	// Create compute pipelines with the specified layout
	VkPipelineLayout pipelineLayout;

	// Initialize pipeline layout creation info
	VkPipelineLayoutCreateInfo pipelineLayoutInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 1, // Specify number of layouts
		.pSetLayouts = &descriptorSetLayout, // Specify layouts
		.pushConstantRangeCount = 0, // Specify push constant range count
		.pPushConstantRanges = nullptr // Specify push constant ranges
	};

	VKASSERT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

	VkComputePipelineCreateInfo pipelineInfo = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = stageInfo,
		.layout = pipelineLayout,
	};

	VkPipeline pipeline;
	VKASSERT(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline));

	// Create command pool and command buffer
	VkCommandPoolCreateInfo poolInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = 0
	};

	VkCommandPool commandPool;
	VKASSERT(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

	VkCommandBufferAllocateInfo allocInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer commandBuffer;
	VKASSERT(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

	// Record commands into the command buffer
	VkCommandBufferBeginInfo beginInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = 0,
		.pInheritanceInfo = nullptr
	};

	VKASSERT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	vkCmdBindDescriptorSets(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		pipelineLayout,
		0,
		1,
		&descriptorSet,
		0,
		nullptr
	);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

	VkBuffer inBuffer, outBuffer;
	unsigned long inBufferBytes, outBufferBytes, bufferBytes;
	inBufferBytes = work_size * work_size * sizeof(glm::dvec2);
	outBufferBytes = work_size * work_size * sizeof(glm::dvec2) * out_count;
	bufferBytes = inBufferBytes + outBufferBytes;

	{
		double inBufNum, outBufNum, totBufNum;
		char inBufSuf, outBufSuf, totBufSuf;

		siprefix2(inBufferBytes, &inBufNum, &inBufSuf);
		siprefix2(outBufferBytes, &outBufNum, &outBufSuf);
		siprefix2(bufferBytes, &totBufNum, &totBufSuf);

		printf("\t--- in Buffer: %g%cB\n\t--- out Buffer: %g%cB\n\t--- total Buffer: %g%cB\n",
			inBufNum, inBufSuf,
			outBufNum, outBufSuf,
			totBufNum, totBufSuf);
	}

	VkBufferCreateInfo inBufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size  = inBufferBytes,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &queueFamilyIndex
	};
	VKASSERT(vkCreateBuffer(device, &inBufferCreateInfo, nullptr, &inBuffer));

	VkBufferCreateInfo outBufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size  = outBufferBytes,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &queueFamilyIndex
	};
	VKASSERT(vkCreateBuffer(device, &outBufferCreateInfo, nullptr, &outBuffer));

	VkDeviceMemory inBufferMemory;
	VkMemoryAllocateInfo memoryAllocateInfo {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = nullptr,
		.allocationSize = bufferBytes,
		.memoryTypeIndex = 1
	};
	VKASSERT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &inBufferMemory));

	VKASSERT(vkBindBufferMemory(device, inBuffer, inBufferMemory, 0));
	VKASSERT(vkBindBufferMemory(device, outBuffer, inBufferMemory, inBufferBytes));

	VkDescriptorBufferInfo descriptorBufferInfo[] = {
		{
			.buffer = inBuffer,
			.offset = 0,
			.range  = inBufferBytes,
		},
		{
			.buffer = outBuffer,
			.offset = 0,
			.range  = outBufferBytes,
		}
	};
	VkWriteDescriptorSet writeDescriptorSet {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = descriptorSet,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 2,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pImageInfo = nullptr,
		.pBufferInfo = descriptorBufferInfo,
		.pTexelBufferView = nullptr
	};
	vkUpdateDescriptorSets(
		device,
		1,
		&writeDescriptorSet,
		0,
		nullptr
	);

	// Dispatch compute shader
	vkCmdDispatch(commandBuffer, 1, 1, 1);

	// End recording and submit command buffer
	VKASSERT(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr
	};


	struct timespec startTime, endTime;
	fprintf(stderr, "Starting Compute Queue\n");
	clock_gettime(CLOCK_MONOTONIC, &startTime);

	for (int i = 16; i; --i) {
		VKASSERT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Wait for queue completion and cleanup resources
		VKASSERT(vkQueueWaitIdle(queue));
	}

	clock_gettime(CLOCK_MONOTONIC, &endTime);

	fprintf(stderr, "Compute Queue Finished after %gs\n", (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	vkDestroyCommandPool(device, commandPool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	vkDestroyPipelineCache(device, pipelineCache, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	// vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
	vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	vkDestroyShaderModule(device, computeShaderModule, nullptr);
	vkDestroyBuffer(device, inBuffer, nullptr);
	vkDestroyBuffer(device, outBuffer, nullptr);
	vkFreeMemory(device, inBufferMemory, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);

	return 0;
}
