# Minimum required CMake version
cmake_minimum_required(VERSION 2.8)
 add_compile_options(-std=c++11)
# Project name
project(object_categorization)


find_package( OpenCV 3 REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()




# Find the Boost installation, specifically the components 'system' and 'filesystem'
find_package(Boost COMPONENTS system filesystem REQUIRED)

# ${PROJECT_SOURCE_DIR} is the name of the root directory of the project
# TO_NATIVE_PATH converts the path ${PROJECT_SOURCE_DIR}/data/ to a full path and the file() command stores it in DATA_FOLDER
file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/data/" DATA_FOLDER)
# set TRAIN_FOLDER to DATA_FOLDER/train_images - this is where we will put our templates for constructing the vocabulary
set(TRAIN_FOLDER "${DATA_FOLDER}train_images/")
# set TEMPLATE_FOLDER to DATA_FOLDER/templates - this is where we will put our traininfg images, in folders organized by category
set(TEMPLATE_FOLDER "${DATA_FOLDER}templates/")

# set the configuration input file to ${PROJECT_SOURCE_DIR}/Config.h.in and the includable header file holding configuration information to ${PROJECT_SOURCE_DIR}/include/Config.h
configure_file("${PROJECT_SOURCE_DIR}/Config.h.in" "${PROJECT_SOURCE_DIR}/include/Config.h")

# Other directories where header files for linked libraries can be found
include_directories(${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include" ${Boost_INCLUDE_DIRS})
file(GLOB SOURCES src/*.cpp)
# executable produced as a result of compilation
add_executable(bow ${SOURCES})
# libraries to be linked with this executable - OpenCV and Boost (system and filesystem components)
target_link_libraries(bow ${OpenCV_LIBS}  ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
