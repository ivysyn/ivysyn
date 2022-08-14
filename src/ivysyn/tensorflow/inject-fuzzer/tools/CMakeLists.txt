# THE LIST OF TOOLS AND THE CORRESPONDING SOURCE FILES
# ====================================================
set(INJECT_FUZZER_TOOLS
    inject-fuzzer
    inject-fuzzer-gpu-only
    inject-gettypes
    inject-validate
)


set(inject-fuzzer_SOURCES
    InjectFuzzerMain.cpp
  ../lib/InjectFuzzer.cpp
)

set(inject-fuzzer-gpu-only_SOURCES
    InjectFuzzerGPUOnlyMain.cpp
  ../lib/InjectFuzzerGPUOnly.cpp
)

set(inject-gettypes_SOURCES
    InjectGetTypesMain.cpp
  ../lib/InjectGetTypes.cpp
)


set(inject-validate_SOURCES
    InjectValidateMain.cpp
  ../lib/InjectValidate.cpp
)
# CONFIGURE THE TOOLS
# ===================
foreach( tool ${INJECT_FUZZER_TOOLS} )
    # Create a library corresponding to 'plugin'
    add_executable(
      ${tool}
      ${${tool}_SOURCES}
      )

    # Link with libclangTooling
    target_link_libraries(
      ${tool}
      clangTooling
    )

    # Configure include directories for 'tool'
    target_include_directories(
      ${tool}
      PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )
endforeach()
