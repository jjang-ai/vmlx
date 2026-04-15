add_executable(example ${CMAKE_CURRENT_LIST_DIR}/example.c)
target_link_libraries(example PUBLIC mlxc)

add_executable(example-float64 ${CMAKE_CURRENT_LIST_DIR}/example-float64.c)
target_link_libraries(example-float64 PUBLIC mlxc)

add_executable(example-grad ${CMAKE_CURRENT_LIST_DIR}/example-grad.c)
target_link_libraries(example-grad PUBLIC mlxc)

add_executable(example-safe-tensors
               ${CMAKE_CURRENT_LIST_DIR}/example-safe-tensors.c)
target_link_libraries(example-safe-tensors PUBLIC mlxc)

add_executable(example-metal-kernel
               ${CMAKE_CURRENT_LIST_DIR}/example-metal-kernel.c)
target_link_libraries(example-metal-kernel PUBLIC mlxc)

add_executable(example-closure ${CMAKE_CURRENT_LIST_DIR}/example-closure.c)
target_link_libraries(example-closure PUBLIC mlxc)

add_executable(example-export ${CMAKE_CURRENT_LIST_DIR}/example-export.c)
target_link_libraries(example-export PUBLIC mlxc)
