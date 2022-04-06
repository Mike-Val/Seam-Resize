cmake_minimum_required(VERSION 3.9)
project(SeamResize VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

INCLUDE_DIRECTORIES(/usr/local/var/vcpkg/installed/x64-osx/include/)
LINK_DIRECTORIES(/usr/local/var/vcpkg/installed/x64-osx/lib/)
find_package(OpenCV REQUIRED)

# include(CTest)
# enable_testing()


add_executable(SeamResize main.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
