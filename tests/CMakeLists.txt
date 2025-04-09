# 添加所有测试子目录
file(GLOB TEST_SUBDIRS LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
foreach(test_dir ${TEST_SUBDIRS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/CMakeLists.txt)
        add_subdirectory(${test_dir})
    endif()
endforeach()
