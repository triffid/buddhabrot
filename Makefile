# blah
PROJECT = buddhabrot

O ?= build

TOOLCHAIN_PATH ?=
ARCH      ?=
PREFIX    := $(if $(ARCH),$(ARCH)-,)

CXXSRC    := $(shell find src/ -name '*.cpp')
INC       := $(shell find src/ -type d)
RM_BDIRS  := $(shell find src/ -type d | sort -r)
RM_BDIRS  += shaders

LIBRARIES := m
PACKAGES  := vulkan sdl2 stb glfw3 fmt imgui

# vk-bootstrap
CXXSRC    += extern/vk-bootstrap/src/VkBootstrap.cpp
INC       += extern/vk-bootstrap/src
RM_BDIRS  += extern/vk-bootstrap/src extern/vk-bootstrap

# Vulkan-Memory-Allocator
# CXXSRC    +=
# INC       += extern/VulkanMemoryAllocator/include

RM_BDIRS  += extern

OBJ       := $(patsubst %.cpp,$(O)/%.o,$(CXXSRC))

FLAGS     := -O2 -g -march=native

CXXFLAGS  := $(FLAGS)
CXXFLAGS  += -Wall -Werror -Wno-error=unused-but-set-variable -Wno-error=unused-variable -std=gnu++20 -pipe -fno-rtti
CXXFLAGS  += $(foreach PACK,$(PACKAGES),$(shell pkgconf --cflags $(PACK)))
CXXFLAGS  += $(patsubst %,-I%,$(INC))

LDFLAGS   := $(FLAGS) -pie -Wl,-Map=$(O)/$(PROJECT).map

LIBS      := $(patsubst %,-l%,$(LIBRARIES))
LIBS      += $(foreach PACK,$(PACKAGES),$(shell pkgconf --libs $(PACK)))

CC        := $(TOOLCHAIN_PATH)$(PREFIX)gcc
CXX       := $(TOOLCHAIN_PATH)$(PREFIX)g++
LD        := $(TOOLCHAIN_PATH)$(PREFIX)g++
OBJCOPY   := $(TOOLCHAIN_PATH)$(PREFIX)objcopy
OBJDUMP   := $(TOOLCHAIN_PATH)$(PREFIX)objdump
AR        := $(TOOLCHAIN_PATH)$(PREFIX)ar
SIZE      := $(TOOLCHAIN_PATH)$(PREFIX)size
READELF   := $(TOOLCHAIN_PATH)$(PREFIX)readelf
NM        := $(TOOLCHAIN_PATH)$(PREFIX)nm
GDB       := $(TOOLCHAIN_PATH)$(PREFIX)gdb -ex run
RM        := rm -f
RMDIR     := rmdir
MKDIR     := mkdir -p
GLSLC     := glslc

SHADERS   := $(shell find shaders/ -type f)
SPVLIST   := $(patsubst %,$(O)/%.spv,$(SHADERS))

DEP       := $(patsubst %.o,%.d,$(OBJ)) $(patsubst %.spv,%.spv.d,$(SPVLIST))

ifeq (,$(VERBOSE))
QUIET:=@
endif

.PHONY: all clean run debug

all: $(O)/$(PROJECT) $(SPVLIST)

clean:
	@echo "  RM    " $(O)/$(PROJECT) $(OBJ) $(DEP) $(SPVLIST) $(O)/$(PROJECT).map $(O)/textures $(O)/models
	$(QUIET)$(RM) $(O)/$(PROJECT) $(OBJ) $(DEP) $(SPVLIST) $(O)/$(PROJECT).map $(O)/textures $(O)/models
	@echo "  RMDIR " $(patsubst %,$(O)/%,$(RM_BDIRS)) $(O)
	$(QUIET)$(RMDIR) $(patsubst %,$(O)/%,$(RM_BDIRS)) $(O) || true

run: all #| $(O)/models $(O)/textures
	@cd $(O); ./$(PROJECT)

debug: all #| $(O)/models $(O)/textures
	@cd $(O); $(GDB) ./$(PROJECT)

$(O):
	@echo "  MKDIR " $@
	$(QUIET)$(MKDIR) $@

$(O)/models:
	@ln -s ../models $@

$(O)/textures:
	@ln -s ../textures $@

$(O)/$(PROJECT): $(OBJ) | $(O)
	@echo "  LINK  " $@
	$(QUIET)$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

$(O)/%.o: %.cpp | $(O)
	@echo "  CXX   " $@
	@$(MKDIR) $(dir $@)
	$(QUIET)$(CXX) $(CXXFLAGS) -c -MMD -MF $(patsubst %.o,%.d,$@) -o $@ $<

$(O)/%.spv: % | $(O)
	@echo "  SHAD  " $@
	@$(MKDIR) $(dir $@)
	$(QUIET)$(GLSLC) -MD -MF $(patsubst %.spv,%.spv.d,$@) -o $@ $<

-include $(DEP)
