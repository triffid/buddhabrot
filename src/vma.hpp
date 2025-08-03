#pragma once

#define VMA_VULKAN_VERSION 1003000

#include <vk_mem_alloc.h>

#define VKASSERT(f) do { VkResult r; if ((r = f) != VK_SUCCESS) { std::stringstream msg; msg << "\e[1;41;37m" << __FUNCTION__ << " returned " << string_VkResult(r) << " at " << __FILE__ << "#L" << __LINE__ << "\e[0m"; throw std::runtime_error(msg.str()); } } while (0)
