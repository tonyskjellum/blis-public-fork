#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# Only include this block of code once
ifndef COMMON_MK_INCLUDED
COMMON_MK_INCLUDED := yes



#
# --- CFLAGS storage functions -------------------------------------------------
#

# Define a function that stores the value of a variable to a different
# variable containing a specified suffix (corresponding to a configuration).
define store-var-for
$(strip $(1)).$(strip $(2)) := $($(strip $(1)))
endef

# Define a function similar to store-var-for, except that appends instead
# of overwriting.
define append-var-for
$(strip $(1)).$(strip $(2)) += $($(strip $(1)))
endef

# Define a function that stores the value of all of the variables in a
# make_defs.mk file to other variables with the configuration (the
# argument $(1)) added as a suffix. This function is called once from
# each make_defs.mk. Also, add the configuration to CONFIGS_INCL.
define store-make-defs
$(eval $(call store-var-for,CC,         $(1)))
$(eval $(call store-var-for,CC_VENDOR,  $(1)))
$(eval $(call store-var-for,CPPROCFLAGS,$(1)))
$(eval $(call store-var-for,CLANGFLAGS, $(1)))
$(eval $(call store-var-for,CXXLANGFLAGS,$(1)))
$(eval $(call store-var-for,CMISCFLAGS, $(1)))
$(eval $(call store-var-for,CPICFLAGS,  $(1)))
$(eval $(call store-var-for,CWARNFLAGS, $(1)))
$(eval $(call store-var-for,CDBGFLAGS,  $(1)))
$(eval $(call store-var-for,COPTFLAGS,  $(1)))
$(eval $(call store-var-for,CKOPTFLAGS, $(1)))
$(eval $(call store-var-for,CKVECFLAGS, $(1)))
$(eval $(call store-var-for,CROPTFLAGS, $(1)))
$(eval $(call store-var-for,CRVECFLAGS, $(1)))
CONFIGS_INCL += $(1)
endef

# Define a function that retreives the value of a variable for a
# given configuration.
define load-var-for
$($(strip $(1)).$(strip $(2)))
endef



#
# --- CFLAGS query functions ---------------------------------------------------
#

# Define some functions that return the appropriate CFLAGS for a given
# configuration. This assumes that the make_defs.mk files have already been
# included, which results in those values having been stored to
# configuration-qualified variables.

get-noopt-cflags-for   = $(strip $(CFLAGS_PRESET) \
                                 $(call load-var-for,CDBGFLAGS,$(1)) \
                                 $(call load-var-for,CWARNFLAGS,$(1)) \
                                 $(call load-var-for,CPICFLAGS,$(1)) \
                                 $(call load-var-for,CMISCFLAGS,$(1)) \
                                 $(call load-var-for,CLANGFLAGS,$(1)) \
                                 $(call load-var-for,CPPROCFLAGS,$(1)) \
                                 $(INCLUDE_PATHS) $(VERS_DEF) \
                          )

get-noopt-cxxflags-for = $(strip $(CFLAGS_PRESET) \
                                 $(call load-var-for,CDBGFLAGS,$(1)) \
                                 $(call load-var-for,CWARNFLAGS,$(1)) \
                                 $(call load-var-for,CPICFLAGS,$(1)) \
                                 $(call load-var-for,CMISCFLAGS,$(1)) \
                                 $(call load-var-for,CXXLANGFLAGS,$(1)) \
                                 $(call load-var-for,CPPROCFLAGS,$(1)) \
                                 $(INCLUDE_PATHS) $(VERS_DEF) \
                          )

get-refinit-cflags-for = $(strip $(call load-var-for,COPTFLAGS,$(1)) \
                                 $(call get-noopt-cflags-for,$(1)) \
                                 -DBLIS_CNAME=$(1) \
                          )

get-refkern-cflags-for = $(strip $(call load-var-for,CROPTFLAGS,$(1)) \
                                 $(call load-var-for,CRVECFLAGS,$(1)) \
                                 $(call get-noopt-cflags-for,$(1)) \
                                 -DBLIS_CNAME=$(1) \
                          )

get-config-cflags-for  = $(strip $(call load-var-for,COPTFLAGS,$(1)) \
                                 $(call get-noopt-cflags-for,$(1)) \
                          )

get-frame-cflags-for   = $(strip $(call load-var-for,COPTFLAGS,$(1)) \
                                 $(call get-noopt-cflags-for,$(1)) \
                          )

get-kernel-cflags-for  = $(strip $(call load-var-for,CKOPTFLAGS,$(1)) \
                                 $(call load-var-for,CKVECFLAGS,$(1)) \
                                 $(call get-noopt-cflags-for,$(1)) \
                          )

# When compiling sandboxes, we use flags similar to those of general framework
# source. This ensures that the same code can be linked and run across various
# sub-configurations. (If we switch to using refkern/kernel flags, we should
# prevent enabling sandboxes for umbrella families by verifying that
# config_list == config_name if --enable-sandbox is given.)
get-sandbox-c99flags-for = $(call load-var-for,COPTFLAGS,$(1)) \
                           $(call get-noopt-cflags-for,$(1)) \
                           $(SANDBOX_INCLUDE_PATHS)
get-sandbox-cxxflags-for = $(call load-var-for,COPTFLAGS,$(1)) \
                           $(call get-noopt-cxxflags-for,$(1)) \
                           $(SANDBOX_INCLUDE_PATHS)

get-noopt-text          = "(CFLAGS for no optimization)"
get-refinit-text-for    = "('$(1)' CFLAGS for ref. kernel init)"
get-refkern-text-for    = "('$(1)' CFLAGS for ref. kernels)"
get-config-text-for     = "('$(1)' CFLAGS for config code)"
get-frame-text-for      = "('$(1)' CFLAGS for framework code)"
get-kernel-text-for     = "('$(1)' CFLAGS for kernels)"
get-sandbox-c99text-for = "('$(1)' CFLAGS for sandboxes)"
get-sandbox-cxxtext-for = "('$(1)' CXXFLAGS for sandboxes)"



#
# --- Miscellaneous helper functions -------------------------------------------
#

# Define functions that filters a list of filepaths $(1) that contain (or
# omit) an arbitrary substring $(2).
files-that-contain      = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),$(f),)))
files-that-dont-contain = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),,$(f))))



#
# --- Include makefile configuration file --------------------------------------
#

# The path to the directory in which BLIS was built.
ifeq ($(strip $(BUILD_PATH)),)
BUILD_PATH        := .
endif

# Define the name of the configuration file.
CONFIG_MK_FILE     := config.mk

# Include the configuration file.
-include $(BUILD_PATH)/$(CONFIG_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
IS_CONFIGURED     := yes
else
CONFIG_MK_PRESENT := no
IS_CONFIGURED     := no
endif

# If we didn't get config.mk, then we need to set some basic variables so
# that make will function without error for things like 'make clean'.
ifeq ($(IS_CONFIGURED),no)

# If this makefile fragment is being run and there is no config.mk present,
# then it's probably safe to assume that the user is currently located in the
# source distribution.
DIST_PATH := .

# Even though they won't be used explicitly, it appears that setting these
# INSTALL_* variables to something sane (that is, not allowing them default
# to the empty string) is necessary to prevent make from hanging, likely
# because the statements that define UNINSTALL_LIBS and UNINSTALL_HEADERS,
# when evaluated, result in running 'find' on the root directory--definitely
# something we would like to avoid.
INSTALL_LIBDIR := $(HOME)/blis/lib
INSTALL_INCDIR := $(HOME)/blis/include

endif


#
# --- Primary makefile variable definitions ------------------------------------
#

# Construct the architecture-version string, which will be used to name the
# library upon installation.
VERS_CONF          := $(VERSION)-$(CONFIG_NAME)

# All makefile fragments in the tree will have this name.
FRAGMENT_MK        := .fragment.mk

# Locations of important files.
BUILD_DIR          := build
CONFIG_DIR         := config
FRAME_DIR          := frame
REFKERN_DIR        := ref_kernels
KERNELS_DIR        := kernels
SANDBOX_DIR        := sandbox
OBJ_DIR            := obj
LIB_DIR            := lib
INCLUDE_DIR        := include
BLASTEST_DIR       := blastest
TESTSUITE_DIR      := testsuite

# The filename suffix for reference kernels.
REFNM              := ref

# Source suffixes.
CONFIG_SRC_SUFS    := c

KERNELS_SRC_SUFS   := c s S

FRAME_SRC_SUFS     := c

SANDBOX_C99_SUFS   := c
SANDBOX_CXX_SUFS   := cc cpp cxx
SANDBOX_SRC_SUFS   := $(SANDBOX_C99_SUFS) $(SANDBOX_CXX_SUFS)

# Header suffixes.
FRAME_HDR_SUFS     := h

SANDBOX_H99_SUFS   := h
SANDBOX_HXX_SUFS   := hh hpp hxx
SANDBOX_HDR_SUFS   := $(SANDBOX_H99_SUFS) $(SANDBOX_HXX_SUFS)

# Combine all header suffixes and remove duplicates via sort().
ALL_HDR_SUFS       := $(sort $(FRAME_HDR_SUFS) \
                             $(SANDBOX_HDR_SUFS) )

ALL_H99_SUFS       := $(sort $(FRAME_HDR_SUFS) \
                             $(SANDBOX_H99_SUFS) )

# The names of the testsuite input/configuration files.
TESTSUITE_CONF_GEN := input.general
TESTSUITE_CONF_OPS := input.operations
TESTSUITE_FAST_GEN := input.general.fast
TESTSUITE_FAST_OPS := input.operations.fast
TESTSUITE_OUT_FILE := output.testsuite

# CHANGELOG file.
CHANGELOG          := CHANGELOG

# Something for OS X so that echo -n works as expected.
SHELL              := bash

# Construct paths to the four primary directories of source code:
# the config directory, general framework code, reference kernel code,
# and optimized kernel code. NOTE: We declare these as recursively
# expanded variables since DIST_PATH may be overridden later.
CONFIG_PATH        := $(DIST_PATH)/$(CONFIG_DIR)
FRAME_PATH         := $(DIST_PATH)/$(FRAME_DIR)
REFKERN_PATH       := $(DIST_PATH)/$(REFKERN_DIR)
KERNELS_PATH       := $(DIST_PATH)/$(KERNELS_DIR)
SANDBOX_PATH       := $(DIST_PATH)/$(SANDBOX_DIR)



#
# --- Library name and local paths ---------------------------------------------
#

# The base name of the BLIS library that we will build.
LIBBLIS            := libblis

# Construct the base path for the library.
BASE_LIB_PATH      := ./$(LIB_DIR)/$(CONFIG_NAME)

# The shared (dynamic) library file suffix is different for Linux and OS X.
ifeq ($(OS_NAME),Darwin)
SHLIB_EXT          := dylib
else
SHLIB_EXT          := so
endif

# Note: These names will be modified later to include the configuration and
# version strings.
LIBBLIS_A          := $(LIBBLIS).a
LIBBLIS_SO         := $(LIBBLIS).$(SHLIB_EXT)

# Append the base library path to the library names.
LIBBLIS_A_PATH     := $(BASE_LIB_PATH)/$(LIBBLIS_A)
LIBBLIS_SO_PATH    := $(BASE_LIB_PATH)/$(LIBBLIS_SO)



#
# --- Utility program definitions ----------------------------------------------
#

SH         := /bin/sh
MV         := mv
MKDIR      := mkdir -p
RM_F       := rm -f
RM_RF      := rm -rf
SYMLINK    := ln -sf
FIND       := find
GREP       := grep
EGREP      := grep -E
XARGS      := xargs
INSTALL    := install -c

# Script for creating a monolithic header file.
#FLATTEN_H  := $(DIST_PATH)/build/flatten-headers.sh
FLATTEN_H  := $(DIST_PATH)/build/flatten-headers.py

# Default archiver flags.
AR         := ar
ARFLAGS    := cr

# Used to refresh CHANGELOG.
GIT        := git
GIT_LOG    := $(GIT) log --decorate



#
# --- Default linker definitions -----------------------------------------------
#

# NOTE: This section needs to reside before the inclusion of make_defs.mk
# files (just below), as most configurations' make_defs.mk don't tinker
# with things like LDFLAGS, but some do (or may), in which case they can
# manually override whatever they need.

# Define the external libraries we may potentially need at link-time.
LIBM       := -lm
LIBMEMKIND := -lmemkind
LIBPTHREAD := -lpthread

# Default linker flags.
# NOTE: -lpthread is needed unconditionally because BLIS uses pthread_once()
# to initialize itself in a thread-safe manner.
LDFLAGS    := $(LDFLAGS_PRESET) $(LIBM) $(LIBPTHREAD)

# Add libmemkind to the link-time flags, if it was enabled at configure-time.
ifeq ($(MK_ENABLE_MEMKIND),yes)
LDFLAGS    += $(LIBMEMKIND)
endif

# Never use libm with Intel compilers.
ifeq ($(CC_VENDOR),icc)
LDFLAGS    := $(filter-out $(LIBM),$(LDFLAGS))
endif

# Never use libmemkind with Intel SDE.
ifeq ($(DEBUG_TYPE),sde)
LDFLAGS    := $(filter-out $(LIBMEMKIND),$(LDFLAGS))
endif

# The default flag for creating shared objects is different for Linux and
# OS X.
ifeq ($(OS_NAME),Darwin)
SOFLAGS    := -dynamiclib
SOFLAGS    += -Wl,-install_name,$(LIBBLIS_SO).$(SO_MAJOR)
else
SOFLAGS    := -shared
SOFLAGS    += -Wl,-soname,$(LIBBLIS_SO).$(SO_MAJOR)
endif

# Specify the shared library's 'soname' field.

# Decide which library to link to for things like the testsuite. Default
# to the static library, unless only the shared library was enabled, in
# which case we use the shared library.
LIBBLIS_LINK   := $(LIBBLIS_A_PATH)
ifeq ($(MK_ENABLE_SHARED),yes)
ifeq ($(MK_ENABLE_STATIC),no)
LIBBLIS_LINK   := $(LIBBLIS_SO_PATH)
endif
endif


#
# --- Include makefile definitions file ----------------------------------------
#

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE     := make_defs.mk

# Assembly a list of all configuration family members, including the
# configuration family name itself. Note that sort() will remove duplicates
# for situations where CONFIG_NAME is present in CONFIG_LIST, such as would
# be the case for singleton families.
CONFIG_LIST_FAM    := $(sort $(strip $(CONFIG_LIST) $(CONFIG_NAME)))

# Construct the paths to the makefile definitions files, each of which
# resides in a separate configuration sub-directory. We use CONFIG_LIST_FAM
# since we might need the makefile definitions associated with the
# configuration family (if it is an umbrella family).
CONFIG_PATHS       := $(addprefix $(CONFIG_PATH)/, $(CONFIG_LIST_FAM))
MAKE_DEFS_MK_PATHS := $(addsuffix /$(MAKE_DEFS_FILE), $(CONFIG_PATHS))

# Initialize the list of included (found) configurations to empty.
CONFIGS_INCL       :=

# Include the makefile definitions files implied by the list of configurations.
-include $(MAKE_DEFS_MK_PATHS)

# Detect whether we actually got all of the make definitions files. If
# we didn't, then maybe a configuration is mislabeled or missing. The
# check-env-make-defs target checks ALL_MAKE_DEFS_MK_PRESENT and outputs
# an error message if it is set to 'no'.
# NOTE: We use CONFIG_LIST_FAM as the expected list of configurations.
# This combines CONFIG_NAME with CONFIG_LIST. The inclusion of CONFIG_NAME
# is needed for situations where the configuration family is an umbrella
# family (e.g. 'intel64'), since families have separate make_def.mk files.
CONFIGS_EXPECTED := $(CONFIG_LIST_FAM)
ifeq ($(sort $(strip $(CONFIGS_INCL))), \
      $(sort $(strip $(CONFIGS_EXPECTED))))
ALL_MAKE_DEFS_MK_PRESENT := yes
else
ALL_MAKE_DEFS_MK_PRESENT := no
endif



#
# --- Configuration-agnostic flags ---------------------------------------------
#

# --- Linker program ---

# Use whatever compiler was chosen.
LINKER     := $(CC)

# --- Warning flags ---

CWARNFLAGS :=

# Disable unused function warnings and stop compiling on first error for
# all compilers that accept such options: gcc, clang, and icc.
ifneq ($(CC_VENDOR),ibm)
CWARNFLAGS += -Wall -Wno-unused-function -Wfatal-errors
endif

# Disable tautological comparision warnings in clang.
ifeq ($(CC_VENDOR),clang)
CWARNFLAGS += -Wno-tautological-compare
endif

$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CWARNFLAGS,$(c))))

# --- Shared library (position-independent code) flags ---

# Emit position-independent code for dynamic linking.
CPICFLAGS := -fPIC
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CPICFLAGS,$(c))))

# --- Language flags ---

# Enable C99.
CLANGFLAGS := -std=c99
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CLANGFLAGS,$(c))))

# Enable C++11.
CXXLANGFLAGS := -std=c++11
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CXXLANGFLAGS,$(c))))

# --- C Preprocessor flags ---

# Enable clock_gettime() in time.h.
CPPROCFLAGS := -D_POSIX_C_SOURCE=200112L
$(foreach c, $(CONFIG_LIST_FAM), $(eval $(call append-var-for,CPPROCFLAGS,$(c))))



#
# --- Adjust verbosity level manually using make V=[0,1] -----------------------
#

ifeq ($(V),1)
ENABLE_VERBOSE := yes
BLIS_ENABLE_TEST_OUTPUT := yes
endif

ifeq ($(V),0)
ENABLE_VERBOSE := no
BLIS_ENABLE_TEST_OUTPUT := no
endif



#
# --- Append OS-specific libraries to LDFLAGS ----------------------------------
#

ifeq ($(OS_NAME),Linux)
LDFLAGS += -lrt
endif



#
# --- LDFLAGS cleanup ----------------------------------------------------------
#

# Remove duplicate flags/options in LDFLAGS (such as -lpthread) by sorting.
LDFLAGS := $(sort $(LDFLAGS))



#
# --- Include makefile fragments -----------------------------------------------
#

# Initialize our list of directory paths to makefile fragments with the empty
# list. This variable will accumulate all of the directory paths in which
# makefile fragments reside.
FRAGMENT_DIR_PATHS :=

# Initialize our makefile variables that source code files will be accumulated
# into by the makefile fragments. This initialization is very important! These
# variables will end up with weird contents if we don't initialize them to
# empty prior to recursively including the makefile fragments.
MK_CONFIG_SRC      :=
MK_KERNELS_SRC     :=
MK_REFKERN_SRC     :=
MK_FRAME_SRC       :=
MK_SANDBOX_SRC     :=

# -- config --

# Construct paths to each of the sub-configurations specified in the
# configuration list. Note that we use CONFIG_LIST_FAM, which already
# has CONFIG_NAME included (with duplicates removed).
CONFIG_PATHS       := $(addprefix $(CONFIG_PATH)/, $(CONFIG_LIST_FAM))

# This variable is used by the include statements as they recursively include
# one another. For the 'config' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(CONFIG_DIR)

# Recursively include the makefile fragments in each of the sub-configuration
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(CONFIG_PATHS))

# -- kernels --

# Construct paths to each of the kernel sets required by the sub-configurations
# in the configuration list.
KERNEL_PATHS       := $(addprefix $(KERNELS_PATH)/, $(KERNEL_LIST))

# This variable is used by the include statements as they recursively include
# one another. For the 'kernels' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(KERNELS_DIR)

# Recursively include the makefile fragments in each of the kernels sub-
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(KERNEL_PATHS))

# -- ref_kernels --
# -- frame --

# This variable is used by the include statements as they recursively include
# one another. For the framework and reference kernel source trees (ie: the
# 'frame' and 'ref_kernels' directories), we initialize it to the top-level
# directory since that is its parent.
PARENT_PATH        := $(DIST_PATH)

# Recursively include all the makefile fragments in the directories for the
# reference kernels and portable framework.
-include $(addsuffix /$(FRAGMENT_MK), $(REFKERN_PATH))
-include $(addsuffix /$(FRAGMENT_MK), $(FRAME_PATH))

# -- sandbox --

# Construct paths to each sandbox. (At present, there can be only one.)
# NOTE: If $(SANDBOX) is empty (because no sandbox was enabled at configure-
# time) then $(SANDBOX_PATHS) will also be empty, which will cause no
# fragments to be included.
SANDBOX_PATHS      := $(addprefix $(SANDBOX_PATH)/, $(SANDBOX))

# This variable is used by the include statements as they recursively include
# one another. For the 'sandbox' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(SANDBOX_DIR)

# Recursively include the makefile fragments in the sandbox sub-directory.
-include $(addsuffix /$(FRAGMENT_MK), $(SANDBOX_PATHS))

# Create a list of the makefile fragments using the variable into which each
# of the above include statements accumulated their directory paths.
MAKEFILE_FRAGMENTS := $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_DIR_PATHS))

# Detect whether we actually got any makefile fragments. If we didn't, then it
# is likely that the user has not yet generated them (via configure).
ifeq ($(strip $(MAKEFILE_FRAGMENTS)),)
MAKEFILE_FRAGMENTS_PRESENT := no
else
MAKEFILE_FRAGMENTS_PRESENT := yes
endif


#
# --- Important sets of header files and paths ---------------------------------
#

# Define a function that will expand all of the directory paths given in $(1)
# to actual filepaths using the list of suffixes provided $(2).
get-filepaths = $(strip $(foreach path, $(1), \
                            $(foreach suf, $(2), \
                                $(wildcard $(path)/*.$(suf)) \
                 )       )   )

# Define a function that will expand all of the directory paths given in $(1)
# to actual filepaths using the list of suffixes provided $(2), taking only
# the first expansion from each directory with at least one file matching
# the current suffix. Finally, strip the filenames from all resulting files,
# returning only the directory paths.
get-dirpaths  = $(dir $(foreach path, $(1), \
                          $(firstword \
                              $(foreach suf, $(2), \
                                  $(wildcard $(path)/*.$(suf)) \
                 )     )   )   )

# We'll use two directory lists. The first is a list of all of the directories
# in which makefile fragments were generated (plus the current directory). The
# second is the subset of the first that begins with the sandbox root path.
ALLFRAG_DIR_PATHS := . $(FRAGMENT_DIR_PATHS)
SANDBOX_DIR_PATHS := $(filter $(SANDBOX_PATH)/%,$(ALLFRAG_DIR_PATHS))

ALL_H99_FILES     := $(call get-filepaths,$(ALLFRAG_DIR_PATHS),$(ALL_H99_SUFS))
FRAME_H99_FILES   := $(filter-out $(SANDBOX_PATH)/%,$(ALL_H99_FILES))

ALL_H99_DIRPATHS  := $(call get-dirpaths,$(ALLFRAG_DIR_PATHS),$(ALL_H99_SUFS))

SANDBOX_H99_FILES := $(call get-filepaths,$(SANDBOX_DIR_PATHS),$(SANDBOX_H99_SUFS))
SANDBOX_HXX_FILES := $(call get-filepaths,$(SANDBOX_DIR_PATHS),$(SANDBOX_HXX_SUFS))

SANDBOX_HDR_DIRPATHS := $(call get-dirpaths,$(SANDBOX_DIR_PATHS),$(ALL_HDR_SUFS))

# Add -I to each header path so we can specify our include search paths to the
# C compiler.
# NOTE: We no longer need every header path in the source tree since we
# now #include the monolithic/flattened blis.h instead, and thus this
# line is commented out.
#INCLUDE_PATHS   := $(strip $(patsubst %, -I%, $(ALL_HDR_DIRPATHS)))


#
# --- blis.h header definitions ------------------------------------------------
#

# Construct the base path for the intermediate include directory.
BASE_INC_PATH   := $(BUILD_PATH)/$(INCLUDE_DIR)/$(CONFIG_NAME)

# Isolate the path to blis.h by filtering the file from the list of framework
# header files.
BLIS_H          := blis.h
BLIS_H_SRC_PATH := $(filter %/$(BLIS_H), $(FRAME_H99_FILES))

# Construct the path to what will be the intermediate flattened/monolithic
# blis.h file.
BLIS_H_FLAT     := $(BASE_INC_PATH)/$(BLIS_H)


#
# --- cblas.h header definitions -----------------------------------------------
#

# Isolate the path to cblas.h by filtering the file from the list of framework
# header files.
CBLAS_H          := cblas.h
CBLAS_H_SRC_PATH := $(filter %/$(CBLAS_H), $(FRAME_H99_FILES))

# Construct the path to what will be the intermediate flattened/monolithic
# cblas.h file.
CBLAS_H_FLAT    := $(BASE_INC_PATH)/$(CBLAS_H)


#
# --- Compiler include path definitions ----------------------------------------
#

# Obtain a list of header files #included inside of the bli_cntx_ref.c file.
# Paths to these files will be needed when compiling with the monolithic
# header.
REF_KER_SRC     := $(DIST_PATH)/$(REFKERN_DIR)/bli_cntx_ref.c
REF_KER_HEADERS := $(shell $(GREP) "\#include" $(REF_KER_SRC) | sed -e "s/\#include [\"<]\([a-zA-Z0-9\_\.\/\-]*\)[\">].*/\1/g" | $(GREP) -v $(BLIS_H))

# Match each header found above with the path to that header, and then strip
# leading, trailing, and internal whitespace.
REF_KER_H_PATHS := $(strip $(foreach header, $(REF_KER_HEADERS), \
                               $(dir $(filter %/$(header), \
                                              $(FRAME_H99_FILES)))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler. Then add frame/include since it's needed for bli_oapi_w[o]_cntx.h.
REF_KER_I_PATHS := $(strip $(patsubst %, -I%, $(REF_KER_H_PATHS)))
REF_KER_I_PATHS += -I$(DIST_PATH)/frame/include

# Prefix the paths above with the base include path.
INCLUDE_PATHS   := -I$(BASE_INC_PATH) $(REF_KER_I_PATHS) -I$(DIST_PATH)/tci -I$(BUILD_PATH)/tci/tci

# Obtain a list of header paths in the configured sandbox. Then add -I to each
# header path.
SANDBOX_INCLUDE_PATHS := $(strip $(patsubst %, -I%, $(SANDBOX_HDR_DIRPATHS)))


#
# --- BLIS configuration header definitions ------------------------------------
#

# This file was created by configure, but we need to define it here so we can
# remove it as part of the clean targets.
BLIS_CONFIG_H   := ./bli_config.h


#
# --- Special preprocessor macro definitions -----------------------------------
#

# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
VERS_DEF       := -DBLIS_VERSION_STRING=\"$(VERSION)\"



# end of ifndef COMMON_MK_INCLUDED conditional block
endif

