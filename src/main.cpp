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
#include <fcntl.h>

#include "vma.hpp"
#include "stb.hpp"

uint64_t seed;
uint32_t PRNG(uint32_t max) {
	seed += (seed * seed) | 5u;
	if (max)
		return ((seed >> 32u) * max) >> 32u;
	else
		return (seed >> 32u);
}

float frand01() {
	return PRNG(0) / 4294967296.0f;
}

const unsigned work_size = 126;
const unsigned out_count = 16384;

const unsigned res = 8192;

static std::string AppName    = "Buddhabrot";
static std::string EngineName = "Buddhabrot";

static std::string ComputeShaderCode = "";

typedef uint32_t imgfl_t;

unsigned iterations = 64;

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

	{
		int r = open("/dev/urandom", O_RDONLY);
		if (r >= 0) {
			int i = read(r, &seed, sizeof(seed));
			close(r);
		}
		else {
			exit(1);
		}
	}

	imgfl_t* imgfl = (imgfl_t*)malloc(res * res * sizeof(imgfl_t));
	if (!imgfl) {
		double d;
		char s;
		siprefix2(res*res*sizeof(imgfl_t), &d, &s);
		fprintf(stderr, "Failed to allocate %g%cB for image map\n", d, s);
		exit(1);
	}
	{
		char fn[256];
		snprintf(fn, 256, "budd_%u.flt", res);
		int r = open(fn, O_RDONLY);
		if (r >= 0) {
			int i = read(r, imgfl, res*res*sizeof(imgfl_t));
			close(r);
		}
		else {
			fprintf(stderr, "Couldn't open %s, zeroing map\n", fn);
			bzero(imgfl, res*res*sizeof(imgfl_t));
		}
	}

	if (iterations > 0) {
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
		uint32_t queueFamilyIndex = -1U;
		uint32_t memorytype_vram = -1U, memorytype_copyram = -1U, memorytype_hostram = -1U;
		{
			/* Memory heaps and types */
			vkGetPhysicalDeviceMemoryProperties(gpu, &physicalDeviceMemoryProperties);

			for (unsigned i = 0; i < physicalDeviceMemoryProperties.memoryHeapCount; i++) {
				VkMemoryHeap& heap = physicalDeviceMemoryProperties.memoryHeaps[i];
				std::string heapstr = string_VkMemoryHeapFlags(heap.flags);

				double sizeNum;
				char sizeSuf;
				siprefix2(heap.size, &sizeNum, &sizeSuf);

				printf("\tMemory Heap %u has %g%cB and flags 0x%04x (%s)\n", i, sizeNum, sizeSuf, heap.flags, heapstr.c_str());
			}
			for (unsigned i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
				VkMemoryType& type = physicalDeviceMemoryProperties.memoryTypes[i];
				std::string flagstr = string_VkMemoryPropertyFlags(type.propertyFlags);

				VkMemoryHeap& heap = physicalDeviceMemoryProperties.memoryHeaps[type.heapIndex];
				std::string heapstr = string_VkMemoryHeapFlags(heap.flags);

				double sizeNum;
				char sizeSuf;
				siprefix2(heap.size, &sizeNum, &sizeSuf);

				printf("\tMemory Type %u uses heap %u (%g%cB, %s) and has flags 0x%04x (%s)\n", i, type.heapIndex, sizeNum, sizeSuf, heapstr.c_str(), type.propertyFlags, flagstr.c_str());

				if ((memorytype_vram == -1U) && (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
					memorytype_vram = i;
					printf("\t\tLooks like VRAM\n");
				}
				if ((memorytype_copyram == -1U) && ((type.propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) == (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) {
					memorytype_copyram = i;
					printf("\t\tLooks like Transfer RAM Window\n");
				}
				if ((memorytype_hostram == -1U) && ((type.propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) {
					memorytype_hostram = i;
					printf("\t\tLooks like CPU Host RAM\n");
				}
			}

			{
				bool error = false;
				if (memorytype_vram == -1U) {
					fprintf(stderr, "\tDidn't find VRAM!\n");
					error = true;
				}
				if (memorytype_copyram == -1U) {
					fprintf(stderr, "\tDidn't find Transfer RAM Window!\n");
					error = true;
				}
				if (memorytype_hostram == -1U) {
					fprintf(stderr, "\tDidn't find CPU Host RAM!\n");
					error = true;
				}
				if (error)
					exit(1);
			}
		}
		{
			/* Queue types */
			uint32_t queue_count = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_count, nullptr);

			if (queue_count == 0) {
				fprintf(stderr, "No queues?");
				exit(1);
			}

			VkQueueFamilyProperties* queueFamilyProperties = new VkQueueFamilyProperties[queue_count];
			if (!queueFamilyProperties) {
				fprintf(stderr, "new VkQueueFamilyProperties[] failed\n");
				exit(1);
			}
			vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_count, queueFamilyProperties);

			for (unsigned i = 0; i < queue_count; i++) {
				std::string flagstr = string_VkQueueFlags(queueFamilyProperties[i].queueFlags);
				printf("\tQueue %u [%u]: %s\n", i, queueFamilyProperties[i].queueCount, flagstr.c_str());

				if (queueFamilyIndex == -1U) {
					if ((queueFamilyProperties[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
						== (VK_QUEUE_COMPUTE_BIT)) {
						printf("\tLooks like Compute Queue\n");
						queueFamilyIndex = i;
					}
				}
			}
			if (queueFamilyIndex == -1U) {
				for (unsigned i = 0; i < queue_count; i++) {
					if (queueFamilyIndex == -1U) {
						if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
							printf("\tLooks like Compute Queue on second glance\n");
							queueFamilyIndex = i;
						}
					}
				}
			}
		}

		VkQueue queue;

		// Initialize queue creation info
		float queuePriority = 1.0f;     // Priority of each queue (0-1)
		VkDeviceQueueCreateInfo queueCreateInfo {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = queueFamilyIndex,
			.queueCount = 1, // Number of queues in this family
			.pQueuePriorities = &queuePriority,
		};

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
			.queueFamilyIndex = queueFamilyIndex
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

		VkBuffer inBuffer_vram, outBuffer_vram, inBuffer_host, outBuffer_host;
		unsigned long inBufferBytes = work_size * work_size * sizeof(glm::dvec2);
		unsigned long outBufferBytes = work_size * work_size * sizeof(glm::dvec2) * out_count;
		unsigned long bufferBytes = inBufferBytes + outBufferBytes;

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

		/*
		****** VRAM buffers ******
		*/
		VkBufferCreateInfo inBuffer_vramCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size  = inBufferBytes,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = &queueFamilyIndex
		};
		VKASSERT(vkCreateBuffer(device, &inBuffer_vramCreateInfo, nullptr, &inBuffer_vram));

		VkBufferCreateInfo outBuffer_vramCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size  = outBufferBytes,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = &queueFamilyIndex
		};
		VKASSERT(vkCreateBuffer(device, &outBuffer_vramCreateInfo, nullptr, &outBuffer_vram));

		VkDeviceMemory deviceMemory_vram;
		VkMemoryAllocateInfo memoryAllocateInfo_vram {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.pNext = nullptr,
			.allocationSize = bufferBytes,
			.memoryTypeIndex = memorytype_vram
		};
		VKASSERT(vkAllocateMemory(device, &memoryAllocateInfo_vram, nullptr, &deviceMemory_vram));

		VKASSERT(vkBindBufferMemory(device, inBuffer_vram, deviceMemory_vram, 0));
		VKASSERT(vkBindBufferMemory(device, outBuffer_vram, deviceMemory_vram, 0 + inBufferBytes));

		VkDescriptorBufferInfo descriptorBufferInfo[] = {
			{
				.buffer = inBuffer_vram,
				.offset = 0,
				.range  = inBufferBytes,
			},
			{
				.buffer = outBuffer_vram,
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
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

		/*
		*************** end VRAM buffers **************
		*
		*************** CPU/Host buffers **************
		*/

		VkBufferCreateInfo inBuffer_hostCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size  = inBufferBytes,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = &queueFamilyIndex
		};
		VKASSERT(vkCreateBuffer(device, &inBuffer_hostCreateInfo, nullptr, &inBuffer_host));

		VkBufferCreateInfo outBuffer_hostCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size  = outBufferBytes,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = &queueFamilyIndex
		};
		VKASSERT(vkCreateBuffer(device, &outBuffer_hostCreateInfo, nullptr, &outBuffer_host));

		VkDeviceMemory deviceMemory_host;
		VkMemoryAllocateInfo memoryAllocateInfo_host {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.pNext = nullptr,
			.allocationSize = bufferBytes,
			.memoryTypeIndex = memorytype_hostram
		};
		VKASSERT(vkAllocateMemory(device, &memoryAllocateInfo_host, nullptr, &deviceMemory_host));

		VKASSERT(vkBindBufferMemory(device, inBuffer_host, deviceMemory_host, 0));
		VKASSERT(vkBindBufferMemory(device, outBuffer_host, deviceMemory_host, 0 + inBufferBytes));

		/*
		*************** end CPU/Host buffers **********
		*/

		/*
		************** ready to go
		*/

		// Record commands into the command buffer
		VkCommandBufferBeginInfo beginInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = 0,
			.pInheritanceInfo = nullptr
		};
		VKASSERT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

		VkBufferCopy bufferCopy1 {
			.srcOffset = 0,
			.dstOffset = 0,
			.size      = inBufferBytes
		};
		vkCmdCopyBuffer(commandBuffer, inBuffer_host, inBuffer_vram, 1, &bufferCopy1);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		// Dispatch compute shader
		vkCmdDispatch(commandBuffer, work_size, work_size, 1);

		VkBufferCopy bufferCopy2 {
			.srcOffset = 0,
			.dstOffset = 0,
			.size      = outBufferBytes
		};
		vkCmdCopyBuffer(commandBuffer, outBuffer_vram, outBuffer_host, 1, &bufferCopy2);

		// End recording and submit command buffer
		VKASSERT(vkEndCommandBuffer(commandBuffer));

		for (unsigned runc = 0; runc < iterations; runc++) {

			/*
			* ************** fill starting array ***********
			*/


			glm::dvec2* inVals;
			vkMapMemory(device, deviceMemory_host, 0, inBufferBytes, 0, (void**) &inVals);

			for (unsigned i = 0; i < (work_size * work_size); i++) {
				double worksize = 4.0/work_size - 2.0;
				// inVals[i].x = (i / work_size) * 4.0/work_size - 2.0 + (4.0 / work_size / 2.0);
				// inVals[i].y = (i % work_size) * 4.0/work_size - 2.0 + (4.0 / work_size / 2.0);
				// inVals[i].x = (i / work_size) * 4.0/work_size - 2.0 + frand01() * worksize;
				// inVals[i].y = (i % work_size) * 4.0/work_size - 2.0 + frand01() * worksize;
				inVals[i].x = frand01() * 4.0 - 2.0;
				inVals[i].y = frand01() * 4.0 - 2.0;
				// printf("%5u: %g,%g\n", i, inVals[i].x, inVals[i].y);
			}

			vkUnmapMemory(device, deviceMemory_host);

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
			fprintf(stderr, "Starting Compute Queue %u\n", runc);
			clock_gettime(CLOCK_MONOTONIC, &startTime);

			// for (int i = 16; i; --i) {
				VKASSERT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

				// Wait for queue completion and cleanup resources
				VKASSERT(vkQueueWaitIdle(queue));
			// }

			clock_gettime(CLOCK_MONOTONIC, &endTime);

			fprintf(stderr, "Compute Queue %u Finished after %gs\n", runc, (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0);

			/*
			*****************************************
			*/

			glm::dvec2* outVals;
			vkMapMemory(device, deviceMemory_host, inBufferBytes, outBufferBytes, 0, (void**) &outVals);

			for (unsigned i = 0; i < (work_size * work_size); i++) {
				if ((outVals[i * out_count].x != 0) ||
					(outVals[i * out_count].y != 0)) {
					// printf("%u,%u: %g,%g\n", i / work_size, i % work_size, outVals[i * out_count].x, outVals[i * out_count].y);
					for (unsigned j = 0; j < out_count; j++) {
						// printf("       %g,%g\n", outVals[i * out_count + j].x, outVals[i * out_count + j].y);
						if (fabs(outVals[i * out_count + j].x) >= 2)
							break;
						if (fabs(outVals[i * out_count + j].y) >= 2)
							break;
						unsigned x = (outVals[i * out_count + j].x + 2.0) / 4.0 * res;
						unsigned y = (outVals[i * out_count + j].y + 2.0) / 4.0 * res;
						// printf("[%u,%u]", x, y);
						imgfl[x + (res * y)] += 1;
					}
				}
			}

			vkUnmapMemory(device, deviceMemory_host);

		}

		/*
		****************************************
		*/

		// fprintf(stderr, "Compute Queue Finished after %gs\n", (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipelineCache(device, pipelineCache, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		// vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyShaderModule(device, computeShaderModule, nullptr);
		vkDestroyBuffer(device, inBuffer_vram, nullptr);
		vkDestroyBuffer(device, outBuffer_vram, nullptr);
		vkFreeMemory(device, deviceMemory_vram, nullptr);

		vkDestroyBuffer(device, inBuffer_host, nullptr);
		vkDestroyBuffer(device, outBuffer_host, nullptr);
		vkFreeMemory(device, deviceMemory_host, nullptr);

		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

	{
		imgfl_t max = 0, max2 = 0;
		for (unsigned i = 0; i < (res * res); i++) {
			if (imgfl[i] > max)
				max = imgfl[i];
			else if (imgfl[i] > max2)
				max2 = imgfl[i];
		}
		printf("max is %u, max2 is %u\n", max, max2);

		// tall poppy syndrome
		for (unsigned i = 0; i < (res * res); i++) {
			if (imgfl[i] > max / 10)
				imgfl[i] = 0;
		}
	}

	{
		char fn[256];
		snprintf(fn, 256, "budd_%u.flt", res);
		int r = open(fn, O_WRONLY | O_CREAT, 0644);
		if (r >= 0) {
			int i = write(r, imgfl, res*res*sizeof(imgfl_t));
			close(r);
		}
		else {
			fprintf(stderr, "Couldn't save %s\n", fn);
			for (unsigned i = 0; i < (res * res); i++) {
				printf("%u,", imgfl[i]);
			}
		}
	}

	{
		imgfl_t max = 0, max2 = 0;

		for (unsigned i = 0; i < (res * res); i++) {
			if (imgfl[i] > max)
				max = imgfl[i];
			else if (imgfl[i] > max2)
				max2 = imgfl[i];
		}
		printf("max is %u, max2 is %u\n", max, max2);

		typedef struct __attribute__ ((packed)) {
			uint8_t r;
			uint8_t g;
			uint8_t b;
		} rgb_t;

		rgb_t palette[256];
		for (unsigned i = 0; i < 256; i++)
			// palette[i] = {(unsigned char) i, (unsigned char) i, (unsigned char) i, (unsigned char)((i == 0)?0:255)};
			palette[i] = {(unsigned char) i, (unsigned char) i, (unsigned char) i};
		palette[1] = {0, 0, 0};
		palette[2] = {0, 0, 0};

		rgb_t* img = new rgb_t[res*res];
		for (unsigned i = 0; i < (res * res); i++) {
			double val = std::min(1.0, std::max(0.0, imgfl[i] * (1.0 / max2)));
			imgfl_t  v = pow(3.0 * pow(val, 2) - 2.0 * pow(val, 3), 0.2) * 255;
			// imgfl_t  v = (imgfl[i] * (1.0 / max2)) * 255;
			unsigned x = i / res;
			unsigned y = i % res;
			unsigned newi = y * res + x;
			// fprintf(stderr, "%u/%u → val %g → v %u\t", imgfl[i], max2, val, v);
			img[newi] = palette[v];
		}

		char fn[256];
		snprintf(fn, 256, "budd_%u.png", res);
		stbi_write_png(fn, res, res, sizeof(rgb_t), img, res * sizeof(rgb_t));
	}

	return 0;
}
