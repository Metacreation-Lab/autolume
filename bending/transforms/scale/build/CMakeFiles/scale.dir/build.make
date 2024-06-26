# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_SOURCE_DIR = /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build

# Include any dependencies generated for this target.
include CMakeFiles/scale.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/scale.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/scale.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/scale.dir/flags.make

CMakeFiles/scale.dir/op.cpp.o: CMakeFiles/scale.dir/flags.make
CMakeFiles/scale.dir/op.cpp.o: /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/op.cpp
CMakeFiles/scale.dir/op.cpp.o: CMakeFiles/scale.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/scale.dir/op.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/scale.dir/op.cpp.o -MF CMakeFiles/scale.dir/op.cpp.o.d -o CMakeFiles/scale.dir/op.cpp.o -c /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/op.cpp

CMakeFiles/scale.dir/op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/scale.dir/op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/op.cpp > CMakeFiles/scale.dir/op.cpp.i

CMakeFiles/scale.dir/op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/scale.dir/op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/op.cpp -o CMakeFiles/scale.dir/op.cpp.s

# Object files for target scale
scale_OBJECTS = \
"CMakeFiles/scale.dir/op.cpp.o"

# External object files for target scale
scale_EXTERNAL_OBJECTS =

libscale.so: CMakeFiles/scale.dir/op.cpp.o
libscale.so: CMakeFiles/scale.dir/build.make
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libtorch.so
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libc10.so
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libkineto.a
libscale.so: /opt/cuda/lib64/stubs/libcuda.so
libscale.so: /opt/cuda/lib64/libnvrtc.so
libscale.so: /opt/cuda/lib64/libnvToolsExt.so
libscale.so: /opt/cuda/lib64/libcudart.so
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libc10_cuda.so
libscale.so: /usr/lib/libopencv_imgproc.so.4.6.0
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libc10_cuda.so
libscale.so: /home/olaf/Documents/libtorch11.6/libtorch/lib/libc10.so
libscale.so: /opt/cuda/lib64/libcufft.so
libscale.so: /opt/cuda/lib64/libcurand.so
libscale.so: /opt/cuda/lib64/libcublas.so
libscale.so: /usr/lib/libcudnn.so
libscale.so: /opt/cuda/lib64/libnvToolsExt.so
libscale.so: /opt/cuda/lib64/libcudart.so
libscale.so: /usr/lib/libopencv_core.so.4.6.0
libscale.so: CMakeFiles/scale.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libscale.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/scale.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/scale.dir/build: libscale.so
.PHONY : CMakeFiles/scale.dir/build

CMakeFiles/scale.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/scale.dir/cmake_clean.cmake
.PHONY : CMakeFiles/scale.dir/clean

CMakeFiles/scale.dir/depend:
	cd /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build /home/olaf/PycharmProjects/Ray_Autolume/bending/transforms/scale/build/CMakeFiles/scale.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/scale.dir/depend

