vcpkg_buildpath_length_warning(37)

block(SCOPE_FOR VARIABLES PROPAGATE SOURCE_PATH)
set(VCPKG_BUILD_TYPE release) # header-only

vcpkg_from_gitlab(
    GITLAB_URL https://gitlab.com
    OUT_SOURCE_PATH SOURCE_PATH
    REPO spinicist/eigen
    REF f713f9cdcef3160f7ed9d56647f233fc5b7ad096
    SHA512 2f48bfa84768936b5f0ff145a95077240e04a2a6dab97dc70e8896f5cf975c089ea93148978d4aeaa1932de9b72708ebf7c3be2a9a3bf455e01cb639d50efada
    HEAD_REF master
    PATCHES
)

# vcpkg_from_git(
#     URL file:///Users/tobias/Code/eigen
#     OUT_SOURCE_PATH SOURCE_PATH
#     REF 78cb63a5d465d708be850c011d450d92d44d8434
# )

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
        -DEIGEN_BUILD_DOC=OFF
        -DEIGEN_BUILD_PKGCONFIG=ON
        "-DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/share/eigen3"
        "-DPKGCONFIG_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/lib/pkgconfig"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
endblock()

if(NOT VCPKG_BUILD_TYPE)
    file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/eigen3.pc" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig")
endif()
vcpkg_fixup_pkgconfig()

file(GLOB INCLUDES "${CURRENT_PACKAGES_DIR}/include/eigen3/*")
# Copy the eigen header files to conventional location for user-wide MSBuild integration
file(COPY ${INCLUDES} DESTINATION "${CURRENT_PACKAGES_DIR}/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING.README")
