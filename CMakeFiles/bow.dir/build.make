# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/mateusz/magisterska/BagOfWOrds (kopia)"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/mateusz/magisterska/BagOfWOrds (kopia)"

# Include any dependencies generated for this target.
include CMakeFiles/bow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bow.dir/flags.make

CMakeFiles/bow.dir/src/ImageGroup.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/ImageGroup.cpp.o: src/ImageGroup.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/ImageGroup.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/ImageGroup.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/ImageGroup.cpp"

CMakeFiles/bow.dir/src/ImageGroup.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/ImageGroup.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/ImageGroup.cpp" > CMakeFiles/bow.dir/src/ImageGroup.cpp.i

CMakeFiles/bow.dir/src/ImageGroup.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/ImageGroup.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/ImageGroup.cpp" -o CMakeFiles/bow.dir/src/ImageGroup.cpp.s

CMakeFiles/bow.dir/src/ImageGroup.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/ImageGroup.cpp.o.requires

CMakeFiles/bow.dir/src/ImageGroup.cpp.o.provides: CMakeFiles/bow.dir/src/ImageGroup.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/ImageGroup.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/ImageGroup.cpp.o.provides

CMakeFiles/bow.dir/src/ImageGroup.cpp.o.provides.build: CMakeFiles/bow.dir/src/ImageGroup.cpp.o

CMakeFiles/bow.dir/src/Group.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/Group.cpp.o: src/Group.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/Group.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/Group.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/Group.cpp"

CMakeFiles/bow.dir/src/Group.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/Group.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/Group.cpp" > CMakeFiles/bow.dir/src/Group.cpp.i

CMakeFiles/bow.dir/src/Group.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/Group.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/Group.cpp" -o CMakeFiles/bow.dir/src/Group.cpp.s

CMakeFiles/bow.dir/src/Group.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/Group.cpp.o.requires

CMakeFiles/bow.dir/src/Group.cpp.o.provides: CMakeFiles/bow.dir/src/Group.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/Group.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/Group.cpp.o.provides

CMakeFiles/bow.dir/src/Group.cpp.o.provides.build: CMakeFiles/bow.dir/src/Group.cpp.o

CMakeFiles/bow.dir/src/main.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/main.cpp.o: src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/main.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/main.cpp"

CMakeFiles/bow.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/main.cpp" > CMakeFiles/bow.dir/src/main.cpp.i

CMakeFiles/bow.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/main.cpp" -o CMakeFiles/bow.dir/src/main.cpp.s

CMakeFiles/bow.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/main.cpp.o.requires

CMakeFiles/bow.dir/src/main.cpp.o.provides: CMakeFiles/bow.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/main.cpp.o.provides

CMakeFiles/bow.dir/src/main.cpp.o.provides.build: CMakeFiles/bow.dir/src/main.cpp.o

CMakeFiles/bow.dir/src/image.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/image.cpp.o: src/image.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/image.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/image.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/image.cpp"

CMakeFiles/bow.dir/src/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/image.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/image.cpp" > CMakeFiles/bow.dir/src/image.cpp.i

CMakeFiles/bow.dir/src/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/image.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/image.cpp" -o CMakeFiles/bow.dir/src/image.cpp.s

CMakeFiles/bow.dir/src/image.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/image.cpp.o.requires

CMakeFiles/bow.dir/src/image.cpp.o.provides: CMakeFiles/bow.dir/src/image.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/image.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/image.cpp.o.provides

CMakeFiles/bow.dir/src/image.cpp.o.provides.build: CMakeFiles/bow.dir/src/image.cpp.o

CMakeFiles/bow.dir/src/BagOfWords.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/BagOfWords.cpp.o: src/BagOfWords.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/BagOfWords.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/BagOfWords.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/BagOfWords.cpp"

CMakeFiles/bow.dir/src/BagOfWords.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/BagOfWords.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/BagOfWords.cpp" > CMakeFiles/bow.dir/src/BagOfWords.cpp.i

CMakeFiles/bow.dir/src/BagOfWords.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/BagOfWords.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/BagOfWords.cpp" -o CMakeFiles/bow.dir/src/BagOfWords.cpp.s

CMakeFiles/bow.dir/src/BagOfWords.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/BagOfWords.cpp.o.requires

CMakeFiles/bow.dir/src/BagOfWords.cpp.o.provides: CMakeFiles/bow.dir/src/BagOfWords.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/BagOfWords.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/BagOfWords.cpp.o.provides

CMakeFiles/bow.dir/src/BagOfWords.cpp.o.provides.build: CMakeFiles/bow.dir/src/BagOfWords.cpp.o

CMakeFiles/bow.dir/src/File.cpp.o: CMakeFiles/bow.dir/flags.make
CMakeFiles/bow.dir/src/File.cpp.o: src/File.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles" $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bow.dir/src/File.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bow.dir/src/File.cpp.o -c "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/File.cpp"

CMakeFiles/bow.dir/src/File.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bow.dir/src/File.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/File.cpp" > CMakeFiles/bow.dir/src/File.cpp.i

CMakeFiles/bow.dir/src/File.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bow.dir/src/File.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/mateusz/magisterska/BagOfWOrds (kopia)/src/File.cpp" -o CMakeFiles/bow.dir/src/File.cpp.s

CMakeFiles/bow.dir/src/File.cpp.o.requires:
.PHONY : CMakeFiles/bow.dir/src/File.cpp.o.requires

CMakeFiles/bow.dir/src/File.cpp.o.provides: CMakeFiles/bow.dir/src/File.cpp.o.requires
	$(MAKE) -f CMakeFiles/bow.dir/build.make CMakeFiles/bow.dir/src/File.cpp.o.provides.build
.PHONY : CMakeFiles/bow.dir/src/File.cpp.o.provides

CMakeFiles/bow.dir/src/File.cpp.o.provides.build: CMakeFiles/bow.dir/src/File.cpp.o

# Object files for target bow
bow_OBJECTS = \
"CMakeFiles/bow.dir/src/ImageGroup.cpp.o" \
"CMakeFiles/bow.dir/src/Group.cpp.o" \
"CMakeFiles/bow.dir/src/main.cpp.o" \
"CMakeFiles/bow.dir/src/image.cpp.o" \
"CMakeFiles/bow.dir/src/BagOfWords.cpp.o" \
"CMakeFiles/bow.dir/src/File.cpp.o"

# External object files for target bow
bow_EXTERNAL_OBJECTS =

bow: CMakeFiles/bow.dir/src/ImageGroup.cpp.o
bow: CMakeFiles/bow.dir/src/Group.cpp.o
bow: CMakeFiles/bow.dir/src/main.cpp.o
bow: CMakeFiles/bow.dir/src/image.cpp.o
bow: CMakeFiles/bow.dir/src/BagOfWords.cpp.o
bow: CMakeFiles/bow.dir/src/File.cpp.o
bow: CMakeFiles/bow.dir/build.make
bow: /usr/local/lib/libopencv_xphoto.so.3.1.0
bow: /usr/local/lib/libopencv_xobjdetect.so.3.1.0
bow: /usr/local/lib/libopencv_ximgproc.so.3.1.0
bow: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
bow: /usr/local/lib/libopencv_tracking.so.3.1.0
bow: /usr/local/lib/libopencv_text.so.3.1.0
bow: /usr/local/lib/libopencv_surface_matching.so.3.1.0
bow: /usr/local/lib/libopencv_structured_light.so.3.1.0
bow: /usr/local/lib/libopencv_stereo.so.3.1.0
bow: /usr/local/lib/libopencv_saliency.so.3.1.0
bow: /usr/local/lib/libopencv_rgbd.so.3.1.0
bow: /usr/local/lib/libopencv_reg.so.3.1.0
bow: /usr/local/lib/libopencv_plot.so.3.1.0
bow: /usr/local/lib/libopencv_optflow.so.3.1.0
bow: /usr/local/lib/libopencv_line_descriptor.so.3.1.0
bow: /usr/local/lib/libopencv_hdf.so.3.1.0
bow: /usr/local/lib/libopencv_fuzzy.so.3.1.0
bow: /usr/local/lib/libopencv_face.so.3.1.0
bow: /usr/local/lib/libopencv_dpm.so.3.1.0
bow: /usr/local/lib/libopencv_dnn.so.3.1.0
bow: /usr/local/lib/libopencv_datasets.so.3.1.0
bow: /usr/local/lib/libopencv_ccalib.so.3.1.0
bow: /usr/local/lib/libopencv_bioinspired.so.3.1.0
bow: /usr/local/lib/libopencv_bgsegm.so.3.1.0
bow: /usr/local/lib/libopencv_aruco.so.3.1.0
bow: /usr/local/lib/libopencv_viz.so.3.1.0
bow: /usr/local/lib/libopencv_videostab.so.3.1.0
bow: /usr/local/lib/libopencv_videoio.so.3.1.0
bow: /usr/local/lib/libopencv_video.so.3.1.0
bow: /usr/local/lib/libopencv_superres.so.3.1.0
bow: /usr/local/lib/libopencv_stitching.so.3.1.0
bow: /usr/local/lib/libopencv_shape.so.3.1.0
bow: /usr/local/lib/libopencv_photo.so.3.1.0
bow: /usr/local/lib/libopencv_objdetect.so.3.1.0
bow: /usr/local/lib/libopencv_ml.so.3.1.0
bow: /usr/local/lib/libopencv_imgproc.so.3.1.0
bow: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
bow: /usr/local/lib/libopencv_highgui.so.3.1.0
bow: /usr/local/lib/libopencv_flann.so.3.1.0
bow: /usr/local/lib/libopencv_features2d.so.3.1.0
bow: /usr/local/lib/libopencv_core.so.3.1.0
bow: /usr/local/lib/libopencv_calib3d.so.3.1.0
bow: /usr/lib/x86_64-linux-gnu/libboost_system.so
bow: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
bow: /usr/local/lib/libopencv_text.so.3.1.0
bow: /usr/local/lib/libopencv_face.so.3.1.0
bow: /usr/local/lib/libopencv_ximgproc.so.3.1.0
bow: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
bow: /usr/local/lib/libopencv_shape.so.3.1.0
bow: /usr/local/lib/libopencv_video.so.3.1.0
bow: /usr/local/lib/libopencv_objdetect.so.3.1.0
bow: /usr/local/lib/libopencv_calib3d.so.3.1.0
bow: /usr/local/lib/libopencv_features2d.so.3.1.0
bow: /usr/local/lib/libopencv_ml.so.3.1.0
bow: /usr/local/lib/libopencv_highgui.so.3.1.0
bow: /usr/local/lib/libopencv_videoio.so.3.1.0
bow: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
bow: /usr/local/lib/libopencv_imgproc.so.3.1.0
bow: /usr/local/lib/libopencv_flann.so.3.1.0
bow: /usr/local/lib/libopencv_core.so.3.1.0
bow: CMakeFiles/bow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bow.dir/build: bow
.PHONY : CMakeFiles/bow.dir/build

CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/ImageGroup.cpp.o.requires
CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/Group.cpp.o.requires
CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/main.cpp.o.requires
CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/image.cpp.o.requires
CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/BagOfWords.cpp.o.requires
CMakeFiles/bow.dir/requires: CMakeFiles/bow.dir/src/File.cpp.o.requires
.PHONY : CMakeFiles/bow.dir/requires

CMakeFiles/bow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bow.dir/clean

CMakeFiles/bow.dir/depend:
	cd "/home/mateusz/magisterska/BagOfWOrds (kopia)" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/mateusz/magisterska/BagOfWOrds (kopia)" "/home/mateusz/magisterska/BagOfWOrds (kopia)" "/home/mateusz/magisterska/BagOfWOrds (kopia)" "/home/mateusz/magisterska/BagOfWOrds (kopia)" "/home/mateusz/magisterska/BagOfWOrds (kopia)/CMakeFiles/bow.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/bow.dir/depend

