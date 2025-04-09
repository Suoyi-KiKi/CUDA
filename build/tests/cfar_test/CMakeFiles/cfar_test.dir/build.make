# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kiki/MYCUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kiki/MYCUDA/build

# Include any dependencies generated for this target.
include tests/cfar_test/CMakeFiles/cfar_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/cfar_test/CMakeFiles/cfar_test.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/cfar_test/CMakeFiles/cfar_test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/cfar_test/CMakeFiles/cfar_test.dir/flags.make

tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o: tests/cfar_test/CMakeFiles/cfar_test.dir/flags.make
tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o: ../tests/cfar_test/test_cfar.cu
tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o: tests/cfar_test/CMakeFiles/cfar_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kiki/MYCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o"
	cd /home/kiki/MYCUDA/build/tests/cfar_test && /usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o -MF CMakeFiles/cfar_test.dir/test_cfar.cu.o.d -x cu -c /home/kiki/MYCUDA/tests/cfar_test/test_cfar.cu -o CMakeFiles/cfar_test.dir/test_cfar.cu.o

tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cfar_test.dir/test_cfar.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cfar_test.dir/test_cfar.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cfar_test
cfar_test_OBJECTS = \
"CMakeFiles/cfar_test.dir/test_cfar.cu.o"

# External object files for target cfar_test
cfar_test_EXTERNAL_OBJECTS =

tests/cfar_test/cfar_test: tests/cfar_test/CMakeFiles/cfar_test.dir/test_cfar.cu.o
tests/cfar_test/cfar_test: tests/cfar_test/CMakeFiles/cfar_test.dir/build.make
tests/cfar_test/cfar_test: src/libradar_algorithms.so.1.0.0
tests/cfar_test/cfar_test: /usr/local/cuda-12.6/lib64/libcufft.so
tests/cfar_test/cfar_test: /usr/local/cuda-12.6/lib64/libcusolver.so
tests/cfar_test/cfar_test: /usr/local/cuda-12.6/lib64/libcublas.so
tests/cfar_test/cfar_test: /usr/local/cuda-12.6/lib64/libcusparse.so
tests/cfar_test/cfar_test: /usr/local/cuda-12.6/lib64/libcurand.so
tests/cfar_test/cfar_test: tests/cfar_test/CMakeFiles/cfar_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kiki/MYCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable cfar_test"
	cd /home/kiki/MYCUDA/build/tests/cfar_test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cfar_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/cfar_test/CMakeFiles/cfar_test.dir/build: tests/cfar_test/cfar_test
.PHONY : tests/cfar_test/CMakeFiles/cfar_test.dir/build

tests/cfar_test/CMakeFiles/cfar_test.dir/clean:
	cd /home/kiki/MYCUDA/build/tests/cfar_test && $(CMAKE_COMMAND) -P CMakeFiles/cfar_test.dir/cmake_clean.cmake
.PHONY : tests/cfar_test/CMakeFiles/cfar_test.dir/clean

tests/cfar_test/CMakeFiles/cfar_test.dir/depend:
	cd /home/kiki/MYCUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kiki/MYCUDA /home/kiki/MYCUDA/tests/cfar_test /home/kiki/MYCUDA/build /home/kiki/MYCUDA/build/tests/cfar_test /home/kiki/MYCUDA/build/tests/cfar_test/CMakeFiles/cfar_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/cfar_test/CMakeFiles/cfar_test.dir/depend

